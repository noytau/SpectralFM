#!/bin/bash
set -e

export WANDB_API_KEY="$(echo $WANDB_API_KEY)"

cd /storage/noy/SpectralFM

python -u debug_recon_encoding.py \
  --mode train \
  --ckpt none \
  --manifest /storage/noy/SpectralFM/fairseq/data/nova_data/single_channel_one/train.tsv \
  --lr 1e-4 \
  --n_samples 950000 \
  --steps 50000 \
  --warmup 15 \
  --wandb_project spectralfm-autoencoder \
  --out_dir autoencoder_experiments
