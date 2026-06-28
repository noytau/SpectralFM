#!/usr/bin/env bash
# RunAI — FE decoder TV regularization experiment.
#
# Warm-start: load all 3 heads + backbone from 3ae_norm_exp2_long.pt.
# Training:   ONLY head_fe is trained (head_proj + head_trans frozen).
# Loss:       λ_fe=1.0  λ_tv_fe=0.1  (λ_proj=0  λ_trans=0)
# Data:       single_channel_100 (short run, ~1000 steps)
# Normalize:  --normalize (F.layer_norm per sample, matching backbone pretraining)
#
# What is loaded / frozen / trained:
#   --ckpt              checkpoint_572_4000.pt      backbone base (May-18 run)
#   --init_fe_ckpt      apr28_fe_recon_best.pt      conv FE weights
#   --init_ln_ckpt      apr28_fe_recon_best.pt      post-FE LayerNorm
#   --init_transformer_ckpt base_libri_official.pt  8 ViT layers (0-7)
#   --init_head_fe_ckpt 3ae_norm_exp2_long.pt       FE decoder (warm-start, TRAINED)
#   --init_head_proj_ckpt 3ae_norm_exp2_long.pt     proj decoder (warm-start, FROZEN)
#   --init_head_trans_ckpt 3ae_norm_exp2_long.pt    trans decoder (warm-start, FROZEN)
#
#   FROZEN:  FE, LN, transformer backbone, proj, head_proj, head_trans
#   TRAINED: head_fe  (fine-tuned with MSE + TV regularization)

set -euo pipefail

IMAGE="${IMAGE:-noyhassid/spectralfm-lean:v6}"
PVC="${PVC:-storage}"
REPO="/storage/noy/SpectralFM"
SPEC_CODE="${SPEC_CODE:-${REPO}/code}"

# Backbone checkpoints
TR_BACKBONE_CKPT="${TR_BACKBONE_CKPT:-${REPO}/fairseq/outputs/recon_loss/2026-05-18_13-17-11_tr_signal_short/checkpoints/checkpoint_572_4000.pt}"
APR28_CKPT="${APR28_CKPT:-${REPO}/fairseq/apr28_fe_recon_best.pt}"
BASE_LIBRI_CKPT="${BASE_LIBRI_CKPT:-${REPO}/fairseq/base_libri_official.pt}"

# Warm-start heads from best exp2 long checkpoint
EXP2_LONG_CKPT="${EXP2_LONG_CKPT:-${REPO}/checkpoints/recon_runs_copied/3ae_norm_exp2_long.pt}"

# Data
MANIFEST="${MANIFEST:-${REPO}/fairseq/data/nova_data/single_channel_100/train.tsv}"
N_SAMPLES="${N_SAMPLES:-100}"

OUT_PARENT="${OUT_PARENT:-${REPO}/fairseq/outputs/signal_recon_tv_fe}"
STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
CONDA_ENV="${CONDA_ENV:-spectralfm}"
NODE_POOLS="${NODE_POOLS:-faculty,raja}"
GPU_MEM="${GPU_MEM:-40G}"
WANDB_PROJECT="${WANDB_PROJECT:-spectralfm-runai-signal-recon-tv-fe}"

STEPS="${STEPS:-1000}"
WARMUP="${WARMUP:-100}"
BATCH="${BATCH:-128}"
GA="${GA:-2}"
LR="${LR:-1e-4}"
LAMBDA_TV_FE="${LAMBDA_TV_FE:-0.1}"

JOB_NAME="${JOB_NAME:-sfm-tv-fe-lv${LAMBDA_TV_FE/./p}-s${STEPS}}"

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
  echo "Submitting $JOB_NAME  gpu=${GPU_MEM}  pools=${NODE_POOLS}"
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

OUT="${OUT_PARENT}/${STAMP}_${JOB_NAME}"

REMOTE_CMD="set -euo pipefail; \
cd ${REPO} && git pull --ff-only 2>&1 | tail -5; \
TR=\"${SPEC_CODE}/train_reconstruction.py\"; \
for f in \"\$TR\" \"${TR_BACKBONE_CKPT}\" \"${APR28_CKPT}\" \"${BASE_LIBRI_CKPT}\" \"${EXP2_LONG_CKPT}\" \"${MANIFEST}\"; do \
  [[ -f \"\$f\" ]] || { echo \"ERROR: missing \$f on PVC\"; exit 2; }; \
done; \
export PYTHONPATH=${REPO}/code:${REPO}/fairseq:${REPO}/fairseq/examples CUDA_VISIBLE_DEVICES=0 && \
mkdir -p \"${OUT}\" && \
conda run --no-capture-output -n ${CONDA_ENV} python3 \"\$TR\" \
    --mode train --recon_path transformer \
    --ckpt              ${TR_BACKBONE_CKPT} \
    --init_fe_ckpt      ${APR28_CKPT} \
    --init_ln_ckpt      ${APR28_CKPT} \
    --init_transformer_ckpt ${BASE_LIBRI_CKPT} \
    --random_init_proj \
    --freeze_fe_v2 --freeze_ln --freeze_proj --freeze_transformer \
    --init_head_fe_ckpt    ${EXP2_LONG_CKPT} \
    --init_head_proj_ckpt  ${EXP2_LONG_CKPT} \
    --init_head_trans_ckpt ${EXP2_LONG_CKPT} \
    --freeze_head_proj \
    --freeze_head_trans \
    --lambda_recon_fe   1.0 \
    --lambda_tv_fe      ${LAMBDA_TV_FE} \
    --lambda_recon_proj 0.0 \
    --lambda_recon_trans 0.0 \
    --monitor_recon_proj \
    --normalize \
    --manifest ${MANIFEST} --n_samples ${N_SAMPLES} \
    --steps ${STEPS} --warmup ${WARMUP} \
    --batch_size ${BATCH} --grad_accum_steps ${GA} \
    --lr ${LR} \
    --wandb_project ${WANDB_PROJECT} --wandb_run_name ${JOB_NAME}_${STAMP} \
    --run_suffix tv_fe_${STAMP} \
    --out_dir \"${OUT}\" --device cuda"

echo "=== TV-FE regularization experiment  STAMP=${STAMP} ==="
echo ""
echo "Job:        ${JOB_NAME}"
echo "Backbone:   ${TR_BACKBONE_CKPT}"
echo "Apr-28:     ${APR28_CKPT}"
echo "Libri:      ${BASE_LIBRI_CKPT}"
echo "Warm-start: ${EXP2_LONG_CKPT}  (all 3 heads)"
echo ""
echo "FROZEN:  FE, LN, transformer, proj, head_proj, head_trans"
echo "TRAINED: head_fe (MSE + λ_tv_fe=${LAMBDA_TV_FE} × TV)"
echo ""
echo "Steps=${STEPS}  Warmup=${WARMUP}  Batch=${BATCH}  GA=${GA}  LR=${LR}"
echo "N_SAMPLES=${N_SAMPLES}  Manifest=${MANIFEST}"
echo ""

submit_one "${JOB_NAME}" "${REMOTE_CMD}"

echo "=== Done.  Logs: runai logs -f ${JOB_NAME} ==="
