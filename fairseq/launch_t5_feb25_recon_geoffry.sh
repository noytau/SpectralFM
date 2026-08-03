#!/usr/bin/env bash
# T5 — 3AE reconstruction heads trained FROM SCRATCH on the frozen Feb-25 SSL backbone.
# Full backbone (FE+LN+proj+transformer) loaded via --ckpt from the fairseq SSL checkpoint;
# everything frozen; head_fe/head_proj/head_trans random-init and trained.
# Usage: launch_t5_feb25_recon_geoffry.sh <gpu> [TAG] [STEPS] [N] [BATCH] [GA]
set -euo pipefail

GPU="$1"; TAG="${2:-feb25_3ae_scratch}"
STEPS="${3:-3000}"; N_SAMPLES="${4:-950}"; BATCH="${5:-128}"; GA="${6:-4}"
LR="${LR:-1e-4}"; WARMUP="${WARMUP:-300}"

REPO=/mnt5/noy/SpectralFM
FEB25_CKPT=${REPO}/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt
MANIFEST=${REPO}/fairseq/data/nova_data/single_channel_1k/train_mnt5.tsv

STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
OUT=${REPO}/fairseq/outputs/signal_recon_feb25_local/${STAMP}_${TAG}
mkdir -p "${OUT}"

export PYTHONPATH=${REPO}/code:${REPO}/fairseq:${REPO}/fairseq/examples
export CUDA_VISIBLE_DEVICES=${GPU}

exec /mnt5/noy/miniconda3/envs/spectralfm/bin/python3 ${REPO}/code/train_reconstruction.py \
    --mode train --recon_path transformer \
    --ckpt ${FEB25_CKPT} \
    --freeze_fe_v2 --freeze_ln --freeze_proj --freeze_transformer \
    --lambda_recon_fe   1.0 \
    --lambda_recon_proj 1.0 \
    --lambda_recon_trans 1.0 \
    --normalize \
    --manifest ${MANIFEST} --n_samples ${N_SAMPLES} \
    --steps ${STEPS} --warmup ${WARMUP} \
    --batch_size ${BATCH} --grad_accum_steps ${GA} \
    --lr ${LR} \
    --run_suffix ${TAG}_${STAMP} \
    --out_dir "${OUT}" --device cuda
