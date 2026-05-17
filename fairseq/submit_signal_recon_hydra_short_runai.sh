#!/usr/bin/env bash
# RunAI — short Hydra signal-reconstruction smoke (100 updates, ~100-sample dataset).
#
# Baseline: spectralfm_full_train (single Adam + cosine LR). The smoke YAML
# (spectralfm_recon_loss_tr_signal_short) inherits from it and adds the
# recon-loss model fields + per-component init/freeze.
#
# Training (verify against your spec):
#   • Init: base_libri_official.pt for FE / LN / proj / transformer (shape-safe merge).
#   • Frozen: FE, LN, post_extract_proj, fe_recon_decoder, mask_emb, final_proj.
#   • Trainable: TransformerEncoder + trans_recon_decoder (MirrorDecoder path).
#   • LR: 1e-5 (single Adam; only trainable params actually update).
#   • W&B: project=spectralfm-runai-signal-recon. Logs regression (data2vec EMA)
#     + loss_recon_trans (L2 spectrogram recon) + loss_recon_fe (off but logged).
#
# Prerequisites on PVC (/storage/noy/SpectralFM):
#   1) Branch ``recon/2ae-basemerge`` pulled on the cluster (git) OR ``code/`` synced.
#   2) ``fairseq/base_libri_official.pt`` on PVC.
#   3) ``fairseq/data/nova_data/single_channel_100`` (dataset ROOT, contains train.tsv + wav/).
#   4) ``PYTHONPATH`` includes ``/storage/noy/SpectralFM/code`` (for recon_components loaders).
#   5) WANDB_API_KEY available — taken from your local env and passed to the pod.
#
# Usage:
#   bash fairseq/submit_signal_recon_hydra_short_runai.sh
#   runai logs -f sfm-sr-hydra-short
#
set -euo pipefail

IMAGE="${IMAGE:-noyhassid/spectralfm-lean:v6}"
PVC="${PVC:-storage}"
REPO="/storage/noy/SpectralFM"
FAIRSEQ="${REPO}/fairseq"
CODE="${REPO}/code"
CONDA_ENV="${CONDA_ENV:-spectralfm}"
CONFIG_DIR="examples/data2vec/config/audio/pretraining/recon_loss"
CONFIG_NAME="spectralfm_recon_loss_tr_signal_short"
JOB_NAME="${JOB_NAME:-sfm-sr-hydra-short}"
NODE_POOLS="${NODE_POOLS:-faculty,raja}"
GPU_MEM="${GPU_MEM:-40G}"

BASE_LIBRI="${BASE_LIBRI:-${REPO}/fairseq/base_libri_official.pt}"

# All RunAI data lives under /storage/noy/SpectralFM/fairseq/data/nova_data
# (NOT /mnt5; /mnt5 is local-only). Always pass task.data via CLI override to
# guarantee it wins over any inherited YAML default.
DATA_ROOT="${DATA_ROOT:-${REPO}/fairseq/data/nova_data/single_channel_100}"

# Hydra `# @package _group_` files don't deep-merge top-level keys like
# `hydra:`, `optimization:`, `lr_scheduler:` reliably across `defaults:`
# imports. Pin everything that matters via CLI overrides (always win).
RUN_TAG="$(date -u +%Y-%m-%d_%H-%M-%S)_tr_signal_short"
RUN_DIR="${REPO}/fairseq/outputs/recon_loss/${RUN_TAG}"
MAX_UPDATE="${MAX_UPDATE:-100}"
MAX_EPOCH="${MAX_EPOCH:-100}"
SAVE_INTERVAL_UPDATES="${SAVE_INTERVAL_UPDATES:-50}"
# 5% of MAX_UPDATE by default, capped to a sensible smoke value.
WARMUP_UPDATES="${WARMUP_UPDATES:-$(( MAX_UPDATE / 20 > 10 ? MAX_UPDATE / 20 : 10 ))}"

submit_one() {
  local STATUS
  STATUS=$(runai list jobs 2>/dev/null | awk -v name="$JOB_NAME" '$1 == name {print $2}' || true)
  if [[ "$STATUS" == "Running" ]]; then
    echo "Skipping $JOB_NAME (already Running)"
    return 0
  fi
  if [[ -n "$STATUS" ]]; then
    echo "Deleting stale $JOB_NAME (status: $STATUS)"
    runai delete job "$JOB_NAME" 2>/dev/null || true
    sleep 2
  fi

  local REMOTE
  REMOTE="set -euo pipefail; \
for f in ${BASE_LIBRI} ${CODE}/recon_components.py ${FAIRSEQ}/examples/data2vec/models/data2vec_audio.py; do \
  [[ -f \"\$f\" ]] || { echo \"ERROR: missing \$f on PVC (git pull recon/2ae-basemerge or kubectl cp code/)\"; exit 2; }; \
done; \
[[ -f ${DATA_ROOT}/train.tsv ]] || { echo \"ERROR: missing ${DATA_ROOT}/train.tsv on PVC\"; exit 3; }; \
cd ${FAIRSEQ} && \
export PYTHONPATH=${CODE}:${REPO}/fairseq:${REPO}/fairseq/examples CUDA_VISIBLE_DEVICES=0 && \
conda run --no-capture-output -n ${CONDA_ENV} \
  python fairseq_cli/hydra_train.py \
  --config-dir ${CONFIG_DIR} \
  --config-name ${CONFIG_NAME} \
  common.user_dir=${REPO}/fairseq/examples/ \
  common.wandb_project=spectralfm-runai-signal-recon \
  common.log_interval=10 \
  distributed_training.distributed_world_size=1 \
  dataset.disable_validation=true \
  dataset.batch_size=16 \
  task.data=${DATA_ROOT} \
  optimization.max_update=${MAX_UPDATE} \
  optimization.max_epoch=${MAX_EPOCH} \
  optimization.update_freq=[1] \
  optimization.lr=[1.0e-5] \
  lr_scheduler.warmup_updates=${WARMUP_UPDATES} \
  checkpoint.save_interval_updates=${SAVE_INTERVAL_UPDATES} \
  checkpoint.keep_interval_updates=2 \
  checkpoint.no_epoch_checkpoints=true \
  checkpoint.save_dir=${RUN_DIR}/checkpoints \
  hydra.run.dir=${RUN_DIR} \
  model.epoch_cosim_enable=false \
  model.recon_loss_type=l2 \
  model.recon_decoder_type=mirror \
  model.lambda_recon_fe=0.0 \
  model.lambda_recon_trans=1.0 \
  model.freeze_fe_v2=true \
  model.freeze_ln=true \
  model.freeze_proj=true \
  model.freeze_fe_recon_decoder=true \
  model.freeze_mask_emb=true \
  model.freeze_final_proj=true \
  model.freeze_transformer_v2=false \
  model.freeze_trans_recon_decoder=false \
  model.init_fe_ckpt=${BASE_LIBRI} \
  model.init_ln_ckpt=${BASE_LIBRI} \
  model.init_proj_ckpt=${BASE_LIBRI} \
  model.init_transformer_ckpt=${BASE_LIBRI} \
  model.tag_param_groups=false"

  # WandB env: prefer caller's WANDB_API_KEY; fall back to the key file on PVC.
  local WANDB_ARGS=()
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS+=( -e "WANDB_API_KEY=${WANDB_API_KEY}" )
  fi
  WANDB_ARGS+=( -e "WANDB_PROJECT=spectralfm-runai-signal-recon" )

  echo "Submitting $JOB_NAME  gpu=${GPU_MEM}  pools=${NODE_POOLS}"
  runai submit "$JOB_NAME" \
    --image "$IMAGE" \
    --gpu-memory "$GPU_MEM" \
    --node-pools "$NODE_POOLS" \
    --existing-pvc "claimname=${PVC},path=/storage" \
    "${WANDB_ARGS[@]}" \
    --command -- bash -c "$REMOTE"
}

echo "=== Hydra short signal-recon  job=${JOB_NAME} ==="
echo "W&B project: spectralfm-runai-signal-recon"
echo "Data root:   ${DATA_ROOT}"
echo "Init:        ${BASE_LIBRI}"
echo "Run dir:     ${RUN_DIR}"
echo "Max update:  ${MAX_UPDATE}  (save every ${SAVE_INTERVAL_UPDATES})"
echo "Trainable:   transformer + trans_recon_decoder @ lr=1e-5 (warmup=${WARMUP_UPDATES})"
echo ""
submit_one
echo "=== Done. Logs: runai logs -f ${JOB_NAME} ==="
