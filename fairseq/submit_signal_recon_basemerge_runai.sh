#!/usr/bin/env bash
# RunAI — signal-reconstruction with explicit per-component pretrained inits:
#
#   FE          = Apr-28 FE-AE best  (fairseq/apr28_fe_recon_best.pt)        [frozen]
#   layer_norm  = Apr-28 FE-AE best  (same file)                              [frozen]
#   post_extract_proj = (no source in Apr-28 ckpt → random init)              [trainable]
#   transformer = base_libri data2vec_multi (fairseq/base_libri_official.pt)  [trainable, ViT→fairseq remap]
#   head        = random TransformerMirrorDecoder                              [trainable]
#
# Per-component LR (Adam param groups, cosine warmup decay applied per group):
#   transformer = 5e-5  (low — fine-tune pretrained Libri weights)
#   proj        = 5e-4  (high — random Linear 512→768)
#   head        = 5e-4  (high — random conv/LN/ConvTranspose stack)
#
# Grid (2 jobs):
#   steps ∈ {10000, 100000}
#
# Prerequisites on PVC (/storage/noy/SpectralFM):
#   1) Latest ``code/`` synced (uses new ``--init_*_ckpt`` / ``--lr_*`` / ``--freeze_*`` flags).
#   2) FE source ckpt at ``fairseq/apr28_fe_recon_best.pt``:
#        kubectl cp /mnt5/noy/SpectralFM/fairseq/apr28_fe_recon_best.pt <pod>:/storage/noy/SpectralFM/fairseq/
#   3) Base Libri ckpt already at ``fairseq/base_libri_official.pt`` on PVC.
#   4) Manifest ``train_runai_150k.tsv`` (same as 100k grid).
#
set -euo pipefail

IMAGE="${IMAGE:-noyhassid/spectralfm-lean:v6}"
PVC="${PVC:-storage}"
REPO="/storage/noy/SpectralFM"
SPEC_CODE="${SPEC_CODE:-${REPO}/code}"
FE_CKPT_INIT="${FE_CKPT_INIT:-${REPO}/fairseq/apr28_fe_recon_best.pt}"
TR_CKPT_INIT="${TR_CKPT_INIT:-${REPO}/fairseq/base_libri_official.pt}"
MANIFEST="${MANIFEST:-${REPO}/fairseq/data/nova_data/single_channel_one/train_runai_150k.tsv}"
OUT_PARENT="${OUT_PARENT:-${REPO}/fairseq/outputs/signal_recon_basemerge}"
STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
CONDA_ENV="${CONDA_ENV:-spectralfm}"
NODE_POOLS="${NODE_POOLS:-faculty,raja}"
GPU_MEM="${GPU_MEM:-40G}"
WANDB_PROJECT="${WANDB_PROJECT:-spectralfm-runai-signal-recon-basemerge}"

N_SAMPLES="${N_SAMPLES:-100000}"
BATCH="${BATCH:-512}"
GA="${GA:-4}"

LR_TRANSFORMER="${LR_TRANSFORMER:-5e-5}"
LR_PROJ="${LR_PROJ:-5e-4}"
LR_HEAD="${LR_HEAD:-5e-4}"

submit_one() {
  local JOB_NAME="$1"
  local REMOTE_CMD="$2"

  local STATUS
  STATUS=$(runai list jobs 2>/dev/null | awk -v name="$JOB_NAME" '$1 == name {print $2}' || true)
  if [[ "$STATUS" == "Running" ]]; then
    echo "Skipping $JOB_NAME (already Running)"; echo ""; return 0
  fi
  if [[ -n "$STATUS" ]]; then
    echo "Deleting stale $JOB_NAME (status: $STATUS)"
    runai delete job "$JOB_NAME" 2>/dev/null || true
    sleep 2
  fi

  local NODE_TYPE_ARGS=()
  if [[ -n "${RUNAI_NODE_TYPE:-}" ]]; then
    NODE_TYPE_ARGS=(--node-type "${RUNAI_NODE_TYPE}")
  fi
  echo "Submitting $JOB_NAME  gpu=${GPU_MEM}  pools=${NODE_POOLS}  node-type=${RUNAI_NODE_TYPE:-<unset>}"
  runai submit "$JOB_NAME" \
    --image "$IMAGE" \
    --gpu-memory "$GPU_MEM" \
    --node-pools "$NODE_POOLS" \
    "${NODE_TYPE_ARGS[@]}" \
    --existing-pvc "claimname=${PVC},path=/storage" \
    --command -- bash -c "$REMOTE_CMD"
  echo ""
  sleep 2
}

common_head="set -euo pipefail; \
TR=\"${SPEC_CODE}/train_reconstruction.py\"; \
for f in \"\$TR\" ${FE_CKPT_INIT} ${TR_CKPT_INIT}; do [[ -f \"\$f\" ]] || { echo \"ERROR: missing \$f on PVC\"; exit 2; }; done; \
cd ${REPO} && export PYTHONPATH=${REPO}/code:${REPO}/fairseq:${REPO}/fairseq/examples CUDA_VISIBLE_DEVICES=0"

wandb_args() {
  local name="$1"
  if [[ -n "${WANDB_PROJECT}" ]]; then
    echo -n "--wandb_project ${WANDB_PROJECT} --wandb_run_name ${name}"
  else
    echo -n ""
  fi
}

remote_basemerge() {
  local job="$1" steps="$2" warmup="$3"
  local OUT="${OUT_PARENT}/${STAMP}_${job}"
  local wb; wb="$(wandb_args "$job")"
  echo "${common_head} && mkdir -p \"${OUT}\" && conda run --no-capture-output -n ${CONDA_ENV} python3 \"\$TR\" \
    --mode train --recon_path transformer \
    --ckpt none \
    --init_fe_ckpt          ${FE_CKPT_INIT} \
    --init_ln_ckpt          ${FE_CKPT_INIT} \
    --init_proj_ckpt        ${FE_CKPT_INIT} \
    --init_transformer_ckpt ${TR_CKPT_INIT} \
    --freeze_fe_v2 --freeze_ln \
    --manifest ${MANIFEST} --n_samples ${N_SAMPLES} \
    --steps ${steps} --warmup ${warmup} \
    --batch_size ${BATCH} --grad_accum_steps ${GA} \
    --lr 1e-4 \
    --lr_transformer ${LR_TRANSFORMER} \
    --lr_proj        ${LR_PROJ} \
    --lr_head        ${LR_HEAD} \
    ${wb} --run_suffix basemerge_${STAMP} \
    --out_dir \"${OUT}\" --device cuda"
}

echo "=== Signal recon basemerge  STAMP=${STAMP}  OUT_PARENT=${OUT_PARENT} ==="
echo "FE init       = ${FE_CKPT_INIT}  (FE + LN; proj falls back to random)"
echo "TR init       = ${TR_CKPT_INIT}  (8 ViT blocks → first 8/12 fairseq layers)"
echo "LR(transformer/proj/head) = ${LR_TRANSFORMER} / ${LR_PROJ} / ${LR_HEAD}"
echo ""

submit_one "sfm-srbm-s10k"  "$(remote_basemerge sfm-srbm-s10k  10000  1000)"
submit_one "sfm-srbm-s100k" "$(remote_basemerge sfm-srbm-s100k 100000 10000)"

echo "=== Done: 2 jobs submitted (or skipped if already Running). ==="
echo "Logs: runai logs -f <job-name>"
