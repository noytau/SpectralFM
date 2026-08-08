#!/usr/bin/env bash
# Copy checkpoint_best.pt (or latest checkpoint_N_STEP.pt) from every
# fe_vs_transformer_collapse run into a single flat directory, renamed as:
#   <wandb_run_name>_<YYYY-MM-DD>_<HH-MM-SS>.pt

SRC_BASE="/storage/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse"
DST_DIR="/storage/noy/SpectralFM/checkpoints/runai"

mkdir -p "$DST_DIR"

for RUN_DIR in "$SRC_BASE"/*/; do
    [[ -d "$RUN_DIR" ]] || continue

    CKPT_DIR="$RUN_DIR/checkpoints"
    [[ -d "$CKPT_DIR" ]] || { echo "No checkpoints/ in $RUN_DIR — skipping"; continue; }

    # Prefer checkpoint_best.pt; fall back to the latest checkpoint_N_STEP.pt
    if [[ -f "$CKPT_DIR/checkpoint_best.pt" ]]; then
        SRC="$CKPT_DIR/checkpoint_best.pt"
    else
        SRC=$(ls -t "$CKPT_DIR"/checkpoint_*.pt 2>/dev/null | head -1)
        [[ -n "$SRC" ]] || { echo "No .pt files in $CKPT_DIR — skipping"; continue; }
    fi

    # Extract date_time from directory name (first token: YYYY-MM-DD_HH-MM-SS)
    DIRNAME=$(basename "$RUN_DIR")
    DATETIME=$(echo "$DIRNAME" | grep -oP '^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}')

    # Get wandb_run_name from the training log (set via common.wandb_run_name override)
    LOG="$RUN_DIR/hydra_train.log"
    RUN_NAME=""
    if [[ -f "$LOG" ]]; then
        RUN_NAME=$(grep -m1 "wandb_run_name" "$LOG" \
                   | grep -oP "'wandb_run_name': '[^']+'" \
                   | grep -oP "(?<=': ')[^']+")
    fi

    # Fall back to directory suffix if log is missing
    if [[ -z "$RUN_NAME" ]]; then
        RUN_NAME=$(echo "$DIRNAME" | sed 's/^[0-9-]*_[0-9-]*_//')
    fi

    DST="$DST_DIR/${RUN_NAME}_${DATETIME}.pt"
    echo "Copying $(basename "$SRC")  →  $(basename "$DST")"
    cp "$SRC" "$DST"
done

echo ""
echo "Done. Files in $DST_DIR:"
ls -lh "$DST_DIR"
