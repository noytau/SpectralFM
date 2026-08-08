#!/bin/bash
set -e

WORKDIR="/storage/noy/SpectralFM"
cd "$WORKDIR"

DATA="/storage/noy/data/single_channel_all/wavs"
OUT="autoencoder_experiments"
mkdir -p "$OUT"

LR="1e-4"

echo "═══════════════════════════════════════════"
echo "  TRAIN 6 autoencoder experiments (lr=$LR)"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo "  FE init: random (standalone build)"
echo "═══════════════════════════════════════════"

for N in 100 1000 10000; do
  for S in 1000 10000; do
    TAG="lr0.0001_n${N}_s${S}"
    CKPT_FILE="$OUT/ckpt_${TAG}.pt"

    if [ -f "$CKPT_FILE" ]; then
      echo ""
      echo "──── SKIP: $TAG (checkpoint exists) ────"
      continue
    fi

    echo ""
    echo "──── TRAIN: lr=$LR  n=$N  steps=$S ────"
    python -u debug_recon_encoding.py \
      --mode train \
      --ckpt none \
      --data_dir "$DATA" \
      --lr $LR \
      --n_samples $N \
      --steps $S \
      --out_dir "$OUT"
  done
done

echo ""
echo "═══ TRAINING DONE ═══"
echo "Checkpoints:"
ls -lh "$OUT"/ckpt_*.pt 2>/dev/null || echo "(none)"
