#!/bin/bash
set -e
cd /mnt5/noy/SpectralFM

OUT="autoencoder_experiments"
mkdir -p $OUT

echo "═══════════════════════════════════════════"
echo "  TRAIN 12 autoencoder experiments"
echo "═══════════════════════════════════════════"

for LR in 1e-3 1e-4; do
  for N in 100 1000 10000; do
    for S in 1000 10000; do
      echo ""
      echo "──── TRAIN: lr=$LR  n=$N  steps=$S ────"
      python debug_recon_encoding.py \
        --mode train \
        --lr $LR \
        --n_samples $N \
        --steps $S \
        --out_dir $OUT
    done
  done
done

echo ""
echo "═══ TRAINING DONE ═══"
echo "Checkpoints:"
ls -lh $OUT/ckpt_*.pt
echo ""
echo "To analyze, run:"
echo "  python debug_recon_encoding.py --mode analyze --ckpt_ae $OUT/<ckpt_file>.pt --k 5"
