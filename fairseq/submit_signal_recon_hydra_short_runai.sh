#!/usr/bin/env bash
# RunAI — short Hydra signal-reconstruction smoke (100 updates, ~100-sample dataset).
#
# Training (verify against your spec):
#   • Init: base_libri_official.pt for FE / LN / proj / transformer (shape-safe merge).
#   • Frozen: FE, LN, post_extract_proj, fe_recon_decoder, mask_emb, final_proj.
#   • Trainable: TransformerEncoder + trans_recon_decoder (MirrorDecoder path).
#   • LR: 1e-5 on the ``transformer`` composite group (encoder + trans_recon_decoder joint).
#   • Losses logged to W&B: ``regression`` (data2vec EMA target) + ``recon_trans`` (L2 spectrogram recon).
#
# Prerequisites on PVC (/storage/noy/SpectralFM):
#   1) Branch ``recon/2ae-basemerge`` pulled on the cluster (git) OR ``code/`` synced.
#   2) ``fairseq/base_libri_official.pt`` on PVC.
#   3) ``fairseq/data/nova_data/single_channel_100`` (dataset ROOT, contains train.tsv + wav/).
#   4) ``PYTHONPATH`` includes ``/storage/noy/SpectralFM/code`` (for recon_components loaders).
#   5) WANDB_API_KEY in the image/job env.
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
  distributed_training.distributed_world_size=1 \
  dataset.disable_validation=true \
  task.data=${DATA_ROOT}"

  echo "Submitting $JOB_NAME  gpu=${GPU_MEM}  pools=${NODE_POOLS}"
  runai submit "$JOB_NAME" \
    --image "$IMAGE" \
    --gpu-memory "$GPU_MEM" \
    --node-pools "$NODE_POOLS" \
    --existing-pvc "claimname=${PVC},path=/storage" \
    --command -- bash -c "$REMOTE"
}

echo "=== Hydra short signal-recon  job=${JOB_NAME} ==="
echo "W&B project: spectralfm-runai-signal-recon"
echo "Data root:   ${DATA_ROOT}"
echo "Init:        ${BASE_LIBRI}"
echo "Trainable:   transformer + trans_recon_decoder @ lr=1e-5"
echo ""
submit_one
echo "=== Done. Logs: runai logs -f ${JOB_NAME} ==="
