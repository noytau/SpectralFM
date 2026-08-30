#!/usr/bin/env bash
# T4 — Geoffrey-local launcher for the TV-FE recon experiment (one run).
# Mirrors the inner command of submit_signal_recon_tv_fe_runai.sh with /mnt5 paths.
# Set WANDB_PROJECT env to enable W&B logging.
# Usage: launch_tv_fe_geoffry.sh <gpu_idx> <lambda_tv_fe> [STEPS] [N_SAMPLES] [BATCH] [GA] [TAG]
set -euo pipefail

GPU="$1"; LTV="$2"
STEPS="${3:-2000}"; N_SAMPLES="${4:-950}"; BATCH="${5:-128}"; GA="${6:-4}"
TAG="${7:-tv${LTV/./p}}"
LR="${LR:-1e-4}"; WARMUP="${WARMUP:-200}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

REPO=/mnt5/noy/SpectralFM
TR_BACKBONE_CKPT=${REPO}/fairseq/outputs/recon_loss/2026-05-18_13-17-11_tr_signal_short/checkpoints/checkpoint_572_4000.pt
APR28_CKPT=${REPO}/fairseq/apr28_fe_recon_best.pt
BASE_LIBRI_CKPT=${REPO}/fairseq/base_libri_official.pt
EXP2_LONG_CKPT=${REPO}/checkpoints/recon_runs_copied/3ae_norm_exp2_long.pt
MANIFEST=${REPO}/fairseq/data/nova_data/single_channel_1k/train_mnt5.tsv

STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
OUT=${REPO}/fairseq/outputs/signal_recon_tv_fe_local/${STAMP}_${TAG}
mkdir -p "${OUT}"

WANDB_ARGS=()
if [[ -n "${WANDB_PROJECT}" ]]; then
  WANDB_ARGS=(--wandb_project "${WANDB_PROJECT}" --wandb_run_name "${TAG}_${STAMP}")
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH=${REPO}/code:${REPO}/fairseq:${REPO}/fairseq/examples
export CUDA_VISIBLE_DEVICES=${GPU}

exec /mnt5/noy/miniconda3/envs/spectralfm/bin/python3 ${REPO}/code/train_reconstruction.py \
    --mode train --recon_path transformer \
    --ckpt              ${TR_BACKBONE_CKPT} \
    --init_fe_ckpt      ${APR28_CKPT} \
    --init_ln_ckpt      ${APR28_CKPT} \
    --init_transformer_ckpt ${BASE_LIBRI_CKPT} \
    --init_proj_ckpt    ${EXP2_LONG_CKPT} \
    --freeze_fe_v2 --freeze_ln --freeze_proj --freeze_transformer \
    --init_head_fe_ckpt    ${EXP2_LONG_CKPT} \
    --init_head_proj_ckpt  ${EXP2_LONG_CKPT} \
    --init_head_trans_ckpt ${EXP2_LONG_CKPT} \
    --freeze_head_proj \
    --freeze_head_trans \
    --lambda_recon_fe   1.0 \
    --lambda_tv_fe      ${LTV} \
    --lambda_recon_proj 0.0 \
    --lambda_recon_trans 0.0 \
    --monitor_recon_proj \
    --normalize \
    --data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_1k/wav --n_samples ${N_SAMPLES} \
    --steps ${STEPS} --warmup ${WARMUP} \
    --batch_size ${BATCH} --grad_accum_steps ${GA} \
    --lr ${LR} \
    "${WANDB_ARGS[@]}" \
    --run_suffix ${TAG}_${STAMP} \
    --out_dir "${OUT}" --device cuda
