#!/usr/bin/env bash
# FE autoencoder on the Feb-25 trained backbone — apr28_fe_recon_best recipe replica.
# apr28 recorded params: n_samples=1000, steps=10000, lr=1e-4, warmup=0 (tag lr0.0001_n1000_s10000),
# FE-AE path (encoder+LN+MirrorDecoder ALL trained, raw signals, no normalize).
# Difference vs apr28: FE+LN initialized from the Feb-25 SSL checkpoint instead of random.
# Usage: launch_feb25_fe_ae_geoffry.sh <gpu> [TAG]
set -euo pipefail
GPU="$1"; TAG="${2:-feb25_fe_ae_apr28recipe}"

REPO=/mnt5/noy/SpectralFM
FEB25_CKPT=${REPO}/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt
MANIFEST=${REPO}/fairseq/data/nova_data/single_channel_10k/train_mnt5.tsv

STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
OUT=${REPO}/fairseq/outputs/signal_recon_feb25_local/${STAMP}_${TAG}
mkdir -p "${OUT}"

export PYTHONPATH=${REPO}/code:${REPO}/fairseq:${REPO}/fairseq/examples
export CUDA_VISIBLE_DEVICES=${GPU}

exec /mnt5/noy/miniconda3/envs/spectralfm/bin/python3 ${REPO}/code/train_reconstruction.py \
    --mode train --recon_path fe --freeze_fe_v2 --freeze_ln \
    --ckpt ${FEB25_CKPT} \
    --data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav --n_samples 1000 \
    --steps 10000 --warmup 0 --lr 1e-4 \
    --batch_size 1000 --grad_accum_steps 4 \
    --wandb_project spectralfm-fe-recon-feb25 --wandb_run_name ${TAG}_${STAMP} \
    --run_suffix ${TAG}_${STAMP} \
    --out_dir "${OUT}" --device cuda
