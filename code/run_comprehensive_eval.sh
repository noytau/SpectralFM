#!/usr/bin/env bash
# =============================================================================
# run_comprehensive_eval.sh
#
# Runs 4 evaluation types for a checkpoint directory in one go:
#   1. Reconstruction          eval_fe_decoder.py
#                              → per-variant (Linear/MLP-512/MLP-512-256) FE and
#                                Transformer decoder reconstruction plots
#   2. Structured similarity   evaluation_runner.py  (cosine similarity of subsets)
#   3. Label regression        evaluation_runner.py  (linear probe on labeled data)
#   4. Noise robustness        evaluation_runner.py
#
# Output hierarchy
# ----------------
#   {OUTPUT_DIR}/{TIMESTAMP}/
#     reconstruction/
#       {run_name}/plots/            per-checkpoint plots
#       comparison_bar_chart.png
#       fe_vs_transformer_architecture_multi_model.png
#     runner/
#       {runner_timestamp}/
#         plots/
#         data/
#         eval_report_*.csv
#     summary.txt
#
# Usage
# -----
#   bash code/run_comprehensive_eval.sh
#   bash code/run_comprehensive_eval.sh /path/to/other/checkpoint_dir
#
# Override variables at the top of this file or pass CKPT_DIR as $1.
# =============================================================================

set -euo pipefail

# ── Paths (edit these as needed) ─────────────────────────────────────────────
CKPT_DIR="${1:-/mnt5/noy/SpectralFM/checkpoints/runai/fe_vs_transformer_collapse_with_var_loss}"
OUTPUT_DIR="/mnt5/noy/SpectralFM/code/eval_results/var_loss_eval"
NOVA_DATA="/mnt5/noy/SpectralFM/fairseq/data/nova_data"
EVAL_DATA="$NOVA_DATA/single_channel_10k"      # used by reconstruction (1k train / 1k eval)
LABELED_DATA="$NOVA_DATA/labeled_data"         # used by label_regression
CODE_DIR="/mnt5/noy/SpectralFM/code"

# ── Decoder settings ─────────────────────────────────────────────────────────
DECODER_VARIANTS="0 512 512:256"               # Linear, MLP-512, MLP-512-256
MAX_RECON_SAMPLES=2000                         # 1k train / 1k eval split
RECON_EPOCHS=200

# ── Create timestamped root ───────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ROOT="$OUTPUT_DIR/$TIMESTAMP"
mkdir -p "$ROOT"

# Tee all output to a log file as well
exec > >(tee "$ROOT/run.log") 2>&1

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SpectralFM Comprehensive Evaluation                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Checkpoint dir : $CKPT_DIR"
echo "  Output root    : $ROOT"
echo "  Timestamp      : $TIMESTAMP"
echo ""
echo "  Methods:"
echo "    [1] reconstruction         → $ROOT/reconstruction/"
echo "    [2] structured_similarity  ┐"
echo "    [3] label_regression       ├→ $ROOT/runner/"
echo "    [4] noise_robustness       ┘"
echo ""

# ── Helper: print step header ─────────────────────────────────────────────────
step() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  STEP $1"
    echo "══════════════════════════════════════════════════════════════"
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Reconstruction (FE decoder)
# ─────────────────────────────────────────────────────────────────────────────
step "1/4  Reconstruction (eval_fe_decoder.py)"

RECON_DIR="$ROOT/reconstruction"
mkdir -p "$RECON_DIR"

python "$CODE_DIR/eval_fe_decoder.py" \
    --checkpoint_dir "$CKPT_DIR" \
    --eval_data_dir  "$EVAL_DATA" \
    --max_eval_samples "$MAX_RECON_SAMPLES" \
    --decoder_variants $DECODER_VARIANTS \
    --include_embedding_decoder \
    --epochs "$RECON_EPOCHS" \
    --output_dir "$RECON_DIR"

echo "[✓] Reconstruction done → $RECON_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# STEPS 2–4 — evaluation_runner.py  (structured_similarity, label_regression,
#                                    noise_robustness)
# evaluation_runner.py creates its own timestamped subdir inside --output_dir,
# so we point it at $ROOT/runner/ which keeps it separate from reconstruction/.
# ─────────────────────────────────────────────────────────────────────────────
step "2–4/4  Structured similarity + Label regression + Noise robustness"

RUNNER_DIR="$ROOT/runner"
mkdir -p "$RUNNER_DIR"

python "$CODE_DIR/evaluation_runner.py" \
    --checkpoint_dir  "$CKPT_DIR" \
    --output_dir      "$RUNNER_DIR" \
    --eval_methods    structured_similarity label_regression noise_robustness \
    --nova_data_dir   "$NOVA_DATA" \
    --labeled_data_dir "$LABELED_DATA" \
    --best_only

echo "[✓] Runner methods done → $RUNNER_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
step "Summary"

SUMMARY="$ROOT/summary.txt"
{
    echo "SpectralFM Comprehensive Evaluation — $TIMESTAMP"
    echo "======================================================"
    echo ""
    echo "Checkpoint dir : $CKPT_DIR"
    echo "Output root    : $ROOT"
    echo ""
    echo "Checkpoints evaluated:"
    ls "$CKPT_DIR"/*.pt 2>/dev/null | while read p; do echo "  $(basename "$p")"; done
    echo ""
    echo "Outputs:"
    echo "  reconstruction/                   FE decoder (Linear / MLP-512 / MLP-512-256)"
    echo "    comparison_bar_chart.png"
    echo "    fe_vs_transformer_architecture_multi_model.png"
    echo "    decoder_variants_cross_checkpoint.png"
    echo "    {run_name}/plots/"
    echo "      reconstruction_triple_{variant}.png"
    echo "      reconstruction_all_variants.png"
    echo "      fe_vs_transformer_by_architecture.png"
    echo ""

    # Find the runner timestamp subdir
    RUNNER_TS_DIR=$(ls -d "$RUNNER_DIR"/*/  2>/dev/null | sort -r | head -1)
    if [ -n "$RUNNER_TS_DIR" ]; then
        echo "  runner/$(basename "$RUNNER_TS_DIR")/"
        echo "    plots/"
        ls "$RUNNER_TS_DIR/plots/" 2>/dev/null | while read f; do echo "      $f"; done
        echo ""
        # Print CSV summary if available
        CSV=$(ls "$RUNNER_TS_DIR"/*.csv 2>/dev/null | head -1)
        if [ -n "$CSV" ]; then
            echo "  eval_report: $(basename "$CSV")"
            python3 -c "
import csv, sys
with open('$CSV') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    if rows:
        run_col = 'run_name'
        metrics = ['metric_label_reg_best_r2', 'metric_best_loss']
        header = f'  {\"Run\":<40}' + ''.join(f'  {m.split(\"_\",2)[-1]:>18}' for m in metrics if m in rows[0])
        print(header)
        print('  ' + '-'*(len(header)-2))
        for row in rows:
            line = f'  {row.get(run_col, \"?\"):<40}'
            for m in metrics:
                if m in row:
                    try:    line += f'  {float(row[m]):>18.4f}'
                    except: line += f'  {row.get(m, \"N/A\"):>18}'
            print(line)
" 2>/dev/null || true
        fi
    fi
    echo ""
    echo "Total wall time: $((SECONDS / 60)) min $((SECONDS % 60)) s"
} | tee "$SUMMARY"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All done.  Results: $ROOT"
echo "╚══════════════════════════════════════════════════════════════╝"
