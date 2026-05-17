#!/usr/bin/env bash
# RunAI — short Hydra signal-reconstruction smoke (100 updates, ~100-sample dataset).
#
# Baseline: spectralfm_full_train (single Adam + cosine LR). The smoke YAML
# (spectralfm_recon_loss_tr_signal_short) inherits from it and adds the
# recon-loss model fields + per-component init/freeze.
#
# Training (verify against your spec):
#   • Init: Feb-15 2026 SpectralFM checkpoint_best_2026-02-15.pt loaded via the
#     legacy model_path path (fe_mode=train → full model.load_state_dict strict=False).
#     This is a 12-layer fairseq-layout SpectralFM ckpt — every key (FE × 5,
#     layer_norm, post_extract_proj, encoder × 12, final_proj, mask_emb) shape-matches.
#   • Frozen: FE, LN, post_extract_proj, fe_recon_decoder, mask_emb.
#   • Trainable: TransformerEncoder + final_proj + trans_recon_decoder.
#     train_only_fe=false (don't run freeze_all_except_feature_extractor).
#     freeze_*_v2=false now explicitly UNFREEZES (post May-17 patch to
#     _maybe_apply_recon_components), so we get an unambiguous trainable set.
#   • LR: 1e-5 (single Adam; only trainable params actually update).
#   • W&B: project=spectralfm-runai-signal-recon. Logs regression (data2vec EMA)
#     + loss_recon_trans (L2 spectrogram recon) + loss_recon_fe (off but logged).
#
# Prerequisites on PVC (/storage/noy/SpectralFM):
#   1) Branch ``recon/2ae-basemerge`` pulled on the cluster (git) OR ``code/`` synced.
#   2) ``checkpoints/checkpoint_best_2026-02-15.pt`` on PVC (rsync from local /mnt5).
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

PRETRAINED_CKPT="${PRETRAINED_CKPT:-${REPO}/checkpoints/checkpoint_best_2026-02-15.pt}"

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
# Peak LR (passed to cosine scheduler; warmup ramps 0→LR, then cosine decays LR→min_lr).
LR="${LR:-1.0e-5}"
# Batch-size convention from .cursor/rules/training-batch-size.mdc:
# bsz=512 + update_freq=4 ⇒ effective optimizer batch = 2048 on 1 GPU (matches the
# scale the Feb-15 SpectralFM warm-start was trained at). bsz=16 / uf=1 is for
# smoke tests only (the 100-step verify run). NUM_WORKERS bumped to 4 for bsz=512.
BATCH_SIZE="${BATCH_SIZE:-512}"
UPDATE_FREQ="${UPDATE_FREQ:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

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
for f in ${PRETRAINED_CKPT} ${CODE}/recon_components.py ${FAIRSEQ}/examples/data2vec/models/data2vec_audio.py; do \
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
  dataset.batch_size=${BATCH_SIZE} \
  dataset.num_workers=${NUM_WORKERS} \
  task.data=${DATA_ROOT} \
  optimization.max_update=${MAX_UPDATE} \
  optimization.max_epoch=${MAX_EPOCH} \
  optimization.update_freq=[${UPDATE_FREQ}] \
  optimization.lr=[${LR}] \
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
  model.model_path=${PRETRAINED_CKPT} \
  model.skip_pretrained_weights=false \
  model.fe_mode=train \
  model.train_only_fe=false \
  model.init_fe_ckpt=null \
  model.init_ln_ckpt=null \
  model.init_proj_ckpt=null \
  model.init_transformer_ckpt=null \
  model.init_fe_recon_decoder_ckpt=null \
  model.init_trans_recon_decoder_ckpt=null \
  model.freeze_fe_v2=true \
  model.freeze_ln=true \
  model.freeze_proj=true \
  model.freeze_fe_recon_decoder=true \
  model.freeze_mask_emb=true \
  model.freeze_transformer_v2=false \
  model.freeze_final_proj=false \
  model.freeze_trans_recon_decoder=false \
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
echo "Init:        ${PRETRAINED_CKPT}  (Feb-15 2026, full model_path load)"
echo "Run dir:     ${RUN_DIR}"
echo "Max update:  ${MAX_UPDATE}  (save every ${SAVE_INTERVAL_UPDATES})"
echo "Batch:       bsz=${BATCH_SIZE} update_freq=${UPDATE_FREQ}  (effective optimizer batch = $((BATCH_SIZE*UPDATE_FREQ)))"
echo "Trainable:   transformer + final_proj + trans_recon_decoder @ lr=${LR} (warmup=${WARMUP_UPDATES})"
echo ""
submit_one
echo "=== Done. Logs: runai logs -f ${JOB_NAME} ==="
