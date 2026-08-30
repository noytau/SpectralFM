#!/usr/bin/env bash
# T2 — Geoffrey-local launcher: projection-head experiment (proj TRAINED, random init).
# Same base recipe as launch_tv_fe_geoffry.sh but: proj random-init + trained,
# head_proj trained (lambda_proj=1), head_trans frozen, lambda_tv=0.
# Usage: launch_proj_mlp_geoffry.sh <gpu> <proj_type linear|mlp_gelu> <mlp_hidden> <TAG> [STEPS] [N] [BATCH] [GA]
set -euo pipefail

GPU="$1"; PROJ_TYPE="$2"; HIDDEN="$3"; TAG="$4"
STEPS="${5:-2000}"; N_SAMPLES="${6:-950}"; BATCH="${7:-128}"; GA="${8:-4}"
LR="${LR:-1e-4}"; WARMUP="${WARMUP:-200}"

REPO=/mnt5/noy/SpectralFM
TR_BACKBONE_CKPT=${REPO}/fairseq/outputs/recon_loss/2026-05-18_13-17-11_tr_signal_short/checkpoints/checkpoint_572_4000.pt
APR28_CKPT=${REPO}/fairseq/apr28_fe_recon_best.pt
BASE_LIBRI_CKPT=${REPO}/fairseq/base_libri_official.pt
EXP2_LONG_CKPT=${REPO}/checkpoints/recon_runs_copied/3ae_norm_exp2_long.pt
MANIFEST=${REPO}/fairseq/data/nova_data/single_channel_1k/train_mnt5.tsv

STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
OUT=${REPO}/fairseq/outputs/signal_recon_proj_mlp_local/${STAMP}_${TAG}
mkdir -p "${OUT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=${REPO}/code:${REPO}/fairseq:${REPO}/fairseq/examples
export CUDA_VISIBLE_DEVICES=${GPU}

exec /mnt5/noy/miniconda3/envs/spectralfm/bin/python3 ${REPO}/code/train_reconstruction.py \
    --mode train --recon_path transformer \
    --ckpt              ${TR_BACKBONE_CKPT} \
    --init_fe_ckpt      ${APR28_CKPT} \
    --init_ln_ckpt      ${APR28_CKPT} \
    --init_transformer_ckpt ${BASE_LIBRI_CKPT} \
    --random_init_proj \
    --post_extract_proj_type ${PROJ_TYPE} \
    --post_extract_proj_mlp_hidden ${HIDDEN} \
    --freeze_fe_v2 --freeze_ln --freeze_transformer \
    --init_head_fe_ckpt    ${EXP2_LONG_CKPT} \
    --init_head_proj_ckpt  ${EXP2_LONG_CKPT} \
    --init_head_trans_ckpt ${EXP2_LONG_CKPT} \
    --freeze_head_trans \
    --lambda_recon_fe   1.0 \
    --lambda_recon_proj 1.0 \
    --lambda_recon_trans 0.0 \
    --lambda_tv_fe      0.0 \
    --normalize \
    --data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_1k/wav --n_samples ${N_SAMPLES} \
    --steps ${STEPS} --warmup ${WARMUP} \
    --batch_size ${BATCH} --grad_accum_steps ${GA} \
    --lr ${LR} \
    --run_suffix ${TAG}_${STAMP} \
    --out_dir "${OUT}" --device cuda
