#!/usr/bin/env bash
# RunAI — 3-AE with F.layer_norm normalization (--normalize).
#
# 2 experiments × (short + long) = 4 jobs.
# Same backbone setup as the previous successful 3AE run (checkpoint_572_4000.pt).
#
# ─── Experiment 1: Apr-28 FE head frozen, train proj + trans decoders ────────
#   LOADS (frozen backbone):
#     --ckpt              checkpoint_572_4000.pt     (May-18 full model; base for all backbone weights)
#     --init_fe_ckpt      ckpt_lr0.0001_...w15.pt    (Apr-28: conv FE weights)
#     --init_ln_ckpt      ckpt_lr0.0001_...w15.pt    (Apr-28: post-FE LayerNorm)
#     --init_head_fe_ckpt ckpt_lr0.0001_...w15.pt    (Apr-28: FE MirrorDecoder head)
#     --init_transformer_ckpt base_libri_official.pt  (8 ViT blocks → encoder.layers.0-7)
#     --random_init_proj                              (proj re-randomized)
#   FROZEN:  FE, LN, transformer backbone, proj, head_fe
#   TRAINED: head_proj (TransformerMirrorDecoder from proj features)
#            head_trans (TransformerMirrorDecoder from transformer features)
#
# ─── Experiment 2: Train all 3 decoder heads ──────────────────────────────────
#   LOADS (frozen backbone):
#     same backbone init as Exp 1
#     --init_head_fe_ckpt ckpt_lr0.0001_...w15.pt    (Apr-28: FE head as init; NOT frozen)
#   FROZEN:  FE, LN, transformer backbone, proj
#   TRAINED: head_fe  (fine-tuned from Apr-28 init with normalize)
#            head_proj (TransformerMirrorDecoder)
#            head_trans (TransformerMirrorDecoder)
#
# Both experiments apply --normalize (F.layer_norm per sample, matching data2vec preprocessing).

set -euo pipefail

IMAGE="${IMAGE:-noyhassid/spectralfm-lean:v6}"
PVC="${PVC:-storage}"
REPO="/storage/noy/SpectralFM"
SPEC_CODE="${SPEC_CODE:-${REPO}/code}"

# Backbone checkpoints (same as previous successful 3AE run)
TR_BACKBONE_CKPT="${TR_BACKBONE_CKPT:-${REPO}/fairseq/outputs/recon_loss/2026-05-18_13-17-11_tr_signal_short/checkpoints/checkpoint_572_4000.pt}"
APR28_CKPT="${APR28_CKPT:-${REPO}/fairseq/apr28_fe_recon_best.pt}"
BASE_LIBRI_CKPT="${BASE_LIBRI_CKPT:-${REPO}/fairseq/base_libri_official.pt}"

# Data manifests
MANIFEST_SHORT="${MANIFEST_SHORT:-${REPO}/fairseq/data/nova_data/single_channel_100/train.tsv}"
MANIFEST_LONG="${MANIFEST_LONG:-${REPO}/fairseq/data/nova_data/single_channel_1k/train.tsv}"

OUT_PARENT="${OUT_PARENT:-${REPO}/fairseq/outputs/signal_recon_3ae_norm}"
STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
CONDA_ENV="${CONDA_ENV:-spectralfm}"
NODE_POOLS="${NODE_POOLS:-faculty,raja}"
GPU_MEM="${GPU_MEM:-40G}"
WANDB_PROJECT="${WANDB_PROJECT:-spectralfm-runai-signal-recon-3ae-norm}"

# Short run hyperparams
N_SHORT="${N_SHORT:-100}"
STEPS_SHORT="${STEPS_SHORT:-1000}"
WARMUP_SHORT="${WARMUP_SHORT:-100}"
BATCH_SHORT="${BATCH_SHORT:-128}"
GA_SHORT="${GA_SHORT:-2}"

# Long run hyperparams  (matches previous successful 3AE run: n990 s10k bsz512 warmup1000)
N_LONG="${N_LONG:-990}"
STEPS_LONG="${STEPS_LONG:-10000}"
WARMUP_LONG="${WARMUP_LONG:-1000}"
BATCH_LONG="${BATCH_LONG:-512}"
GA_LONG="${GA_LONG:-1}"

LR="${LR:-1e-4}"

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

common_head() {
  local manifest="$1"
  echo "set -euo pipefail; \
TR=\"${SPEC_CODE}/train_reconstruction.py\"; \
for f in \"\$TR\" \"${TR_BACKBONE_CKPT}\" \"${APR28_CKPT}\" \"${BASE_LIBRI_CKPT}\" \"${manifest}\"; do \
  [[ -f \"\$f\" ]] || { echo \"ERROR: missing \$f on PVC\"; exit 2; }; \
done; \
cd ${REPO} && export PYTHONPATH=${REPO}/code:${REPO}/fairseq:${REPO}/fairseq/examples CUDA_VISIBLE_DEVICES=0"
}

# Shared backbone init flags (same for both experiments)
backbone_init_flags() {
  echo "--ckpt ${TR_BACKBONE_CKPT} \
    --init_fe_ckpt          ${APR28_CKPT} \
    --init_ln_ckpt          ${APR28_CKPT} \
    --init_transformer_ckpt ${BASE_LIBRI_CKPT} \
    --random_init_proj \
    --freeze_fe_v2 --freeze_ln --freeze_proj --freeze_transformer"
}

# ── Experiment 1: Apr-28 FE head frozen, train proj + trans decoders ──────────
build_cmd_exp1() {
  local job="$1"
  local manifest="$2"
  local n="$3"
  local steps="$4"
  local warmup="$5"
  local batch="$6"
  local ga="$7"
  local OUT="${OUT_PARENT}/${STAMP}_${job}"
  local head="$(common_head "${manifest}")"
  local backbone="$(backbone_init_flags)"
  echo "${head} && mkdir -p \"${OUT}\" && \
conda run --no-capture-output -n ${CONDA_ENV} python3 \"\$TR\" \
    --mode train --recon_path transformer \
    ${backbone} \
    --init_head_fe_ckpt ${APR28_CKPT} \
    --freeze_head_fe \
    --lambda_recon_fe 0.0 \
    --lambda_recon_proj 1.0 \
    --lambda_recon_trans 1.0 \
    --monitor_recon_fe \
    --normalize \
    --manifest ${manifest} --n_samples ${n} \
    --steps ${steps} --warmup ${warmup} \
    --batch_size ${batch} --grad_accum_steps ${ga} \
    --lr ${LR} \
    --wandb_project ${WANDB_PROJECT} --wandb_run_name ${job}_${STAMP} \
    --run_suffix norm_exp1_${STAMP} \
    --out_dir \"${OUT}\" --device cuda"
}

# ── Experiment 2: All 3 decoder heads trained (head_fe init from Apr-28) ──────
build_cmd_exp2() {
  local job="$1"
  local manifest="$2"
  local n="$3"
  local steps="$4"
  local warmup="$5"
  local batch="$6"
  local ga="$7"
  local OUT="${OUT_PARENT}/${STAMP}_${job}"
  local head="$(common_head "${manifest}")"
  local backbone="$(backbone_init_flags)"
  echo "${head} && mkdir -p \"${OUT}\" && \
conda run --no-capture-output -n ${CONDA_ENV} python3 \"\$TR\" \
    --mode train --recon_path transformer \
    ${backbone} \
    --init_head_fe_ckpt ${APR28_CKPT} \
    --lambda_recon_fe 1.0 \
    --lambda_recon_proj 1.0 \
    --lambda_recon_trans 1.0 \
    --normalize \
    --manifest ${manifest} --n_samples ${n} \
    --steps ${steps} --warmup ${warmup} \
    --batch_size ${batch} --grad_accum_steps ${ga} \
    --lr ${LR} \
    --wandb_project ${WANDB_PROJECT} --wandb_run_name ${job}_${STAMP} \
    --run_suffix norm_exp2_${STAMP} \
    --out_dir \"${OUT}\" --device cuda"
}

echo "=== 3-AE normalize training  STAMP=${STAMP} ==="
echo ""
echo "Exp 1 — FE head frozen (Apr-28), train head_proj + head_trans"
echo "Exp 2 — All 3 heads trained (head_fe fine-tuned from Apr-28)"
echo ""
echo "Backbone: ${TR_BACKBONE_CKPT}"
echo "Apr-28:   ${APR28_CKPT}"
echo "Libri:    ${BASE_LIBRI_CKPT}"
echo "Short: N=${N_SHORT}  STEPS=${STEPS_SHORT}  WARMUP=${WARMUP_SHORT}  BATCH=${BATCH_SHORT}  GA=${GA_SHORT}"
echo "Long:  N=${N_LONG}   STEPS=${STEPS_LONG}   WARMUP=${WARMUP_LONG}  BATCH=${BATCH_LONG}  GA=${GA_LONG}"
echo ""

submit_one "sfm-3ae-norm-1-short" "$(build_cmd_exp1 sfm-3ae-norm-1-short "${MANIFEST_SHORT}" "${N_SHORT}" "${STEPS_SHORT}" "${WARMUP_SHORT}" "${BATCH_SHORT}" "${GA_SHORT}")"
submit_one "sfm-3ae-norm-1-long"  "$(build_cmd_exp1 sfm-3ae-norm-1-long  "${MANIFEST_LONG}"  "${N_LONG}"  "${STEPS_LONG}"  "${WARMUP_LONG}"  "${BATCH_LONG}"  "${GA_LONG}")"
submit_one "sfm-3ae-norm-2-short" "$(build_cmd_exp2 sfm-3ae-norm-2-short "${MANIFEST_SHORT}" "${N_SHORT}" "${STEPS_SHORT}" "${WARMUP_SHORT}" "${BATCH_SHORT}" "${GA_SHORT}")"
submit_one "sfm-3ae-norm-2-long"  "$(build_cmd_exp2 sfm-3ae-norm-2-long  "${MANIFEST_LONG}"  "${N_LONG}"  "${STEPS_LONG}"  "${WARMUP_LONG}"  "${BATCH_LONG}"  "${GA_LONG}")"

echo "=== Done.  Monitor: runai list jobs | grep sfm-3ae-norm ==="
