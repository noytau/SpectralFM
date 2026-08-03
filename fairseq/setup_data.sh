#!/bin/bash
# Setup data manifests for SpectralFM training (Geoffrey + RunAI).
#
# Run this on Geoffrey. It creates train.tsv / valid.tsv for each subset
# at the paths that RunAI expects (/storage/ prefix in the TSV root).
#
# Usage:
#   bash setup_data.sh
#
# After running, verify with:
#   head -3 /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_all/train.tsv

set -e

PYBIN="/mnt5/noy/miniconda3/envs/spectralfm_env/bin/python3"
SCRIPT="$(dirname "$0")/create_manifests.py"

# Source wav files (Geoffrey local path)
WAV_DIR="/mnt5/noy/fairseq/data/single_channel_1m/wavs"

# Output base (Geoffrey local path = RunAI /storage/ path, just different prefix)
OUT_BASE="/mnt5/noy/SpectralFM/fairseq/data/nova_data"

# The root path written into the TSV (must use /storage/ so RunAI can find the files)
RUNAI_ROOT="/storage/noy/fairseq/data/single_channel_1m/wavs"

echo "=== Checking wav source ==="
WAV_COUNT=$(ls "$WAV_DIR" | wc -l)
echo "Found $WAV_COUNT wav files in $WAV_DIR"

if [ "$WAV_COUNT" -eq 0 ]; then
    echo "ERROR: No wav files found in $WAV_DIR"
    exit 1
fi

echo ""
echo "=== Creating manifests for each subset ==="

# ── single_channel_all: use all available wav files ────────────────────────
echo ""
echo "--- single_channel_all (all ${WAV_COUNT} files) ---"
"$PYBIN" "$SCRIPT" \
    --wav_dir    "$WAV_DIR" \
    --out_dir    "$OUT_BASE/single_channel_all" \
    --runai_root "$RUNAI_ROOT" \
    --valid_frac 0.001

# ── single_channel_one: ~100k files ───────────────────────────────────────
echo ""
echo "--- single_channel_one (100k files) ---"
"$PYBIN" "$SCRIPT" \
    --wav_dir    "$WAV_DIR" \
    --out_dir    "$OUT_BASE/single_channel_one" \
    --runai_root "$RUNAI_ROOT" \
    --valid_frac 0.01 \
    --max_train  100000

# ── single_channel_10k: 10k files ─────────────────────────────────────────
echo ""
echo "--- single_channel_10k (10k files) ---"
"$PYBIN" "$SCRIPT" \
    --wav_dir    "$WAV_DIR" \
    --out_dir    "$OUT_BASE/single_channel_10k" \
    --runai_root "$RUNAI_ROOT" \
    --valid_frac 0.01 \
    --max_train  10000

echo ""
echo "=== All manifests created ==="
echo ""
echo "Summary:"
for subset in single_channel_all single_channel_one single_channel_10k; do
    DIR="$OUT_BASE/$subset"
    if [ -f "$DIR/train.tsv" ]; then
        TRAIN_LINES=$(wc -l < "$DIR/train.tsv")
        VALID_LINES=$(wc -l < "$DIR/valid.tsv")
        ROOT=$(head -1 "$DIR/train.tsv")
        echo "  $subset: train=$((TRAIN_LINES-1)) valid=$((VALID_LINES-1)) root=$ROOT"
    fi
done

echo ""
echo "=== Verify a wav file is reachable ==="
# Check that the TSV root's first file actually exists on Geoffrey
FIRST_FILE=$(sed -n '2p' "$OUT_BASE/single_channel_all/train.tsv" | cut -f1)
GEOFFRY_ROOT="${RUNAI_ROOT/\/storage\//\/mnt5\/}"
CHECK_PATH="$GEOFFRY_ROOT/$FIRST_FILE"
if [ -f "$CHECK_PATH" ]; then
    echo "OK: $CHECK_PATH exists"
else
    echo "WARNING: $CHECK_PATH not found — verify /mnt5 = /storage mapping"
fi

echo ""
echo "Done. To submit RunAI jobs:"
echo "  bash $(dirname "$0")/../sweep_dataset.sh"
