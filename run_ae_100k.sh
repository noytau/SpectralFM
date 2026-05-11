#!/bin/bash
set -e

export WANDB_API_KEY="$(cat /storage/noy/.wandb_key 2>/dev/null || echo $WANDB_API_KEY)"

cd /storage/noy/SpectralFM

python -u debug_recon_encoding.py \
  --mode train \
  --ckpt none \
  --manifest /storage/noy/SpectralFM/fairseq/data/nova_data/single_channel_all/train.tsv \
  --lr 1e-4 \
  --n_samples 100000 \
  --steps 50000 \
  --warmup 15 \
  --wandb_project spectralfm-autoencoder \
  --out_dir autoencoder_experiments
