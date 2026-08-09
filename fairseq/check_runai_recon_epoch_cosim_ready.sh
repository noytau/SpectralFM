#!/usr/bin/env bash
# Verify fairseq tree is ready for recon_loss + epoch_cosim before RunAI submit.
# Run on: RunAI shell (default FAIRSEQ_ROOT) or locally with FAIRSEQ_ROOT=/mnt5/noy/SpectralFM/fairseq
#
# Usage:
#   ./check_runai_recon_epoch_cosim_ready.sh
#   FAIRSEQ_ROOT=/mnt5/noy/SpectralFM/fairseq ./check_runai_recon_epoch_cosim_ready.sh
#
set -euo pipefail

FAIRSEQ_ROOT="${1:-${FAIRSEQ_ROOT:-/storage/noy/SpectralFM/fairseq}}"
ERR=0

say_ok()   { echo "  OK   $*"; }
say_fail() { echo "  FAIL $*"; ERR=1; }

need_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    say_ok "exists: $f"
  else
    say_fail "missing file: $f"
  fi
}

need_grep() {
  local file="$1" pat="$2" desc="$3"
  if [[ -f "$file" ]] && grep -q "$pat" "$file"; then
    say_ok "$desc"
  else
    say_fail "$desc (file=$file pattern=$pat)"
  fi
}

echo "=== FAIRSEQ_ROOT=$FAIRSEQ_ROOT ==="
echo ""

echo "--- Required files ---"
need_file "$FAIRSEQ_ROOT/examples/data2vec/models/data2vec_audio.py"
need_file "$FAIRSEQ_ROOT/examples/data2vec/cosim_epoch_utils.py"
need_file "$FAIRSEQ_ROOT/fairseq/tasks/audio_pretraining.py"
need_file "$FAIRSEQ_ROOT/fairseq_cli/train.py"
YAML="$FAIRSEQ_ROOT/examples/data2vec/config/audio/pretraining/recon_loss/spectralfm_recon_loss.yaml"
need_file "$YAML"
NPY="$FAIRSEQ_ROOT/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy"
need_file "$NPY"
echo ""

echo "--- Content checks ---"
need_grep "$YAML" "epoch_cosim_enable" "YAML mentions epoch_cosim"
need_grep "$FAIRSEQ_ROOT/fairseq/tasks/audio_pretraining.py" "maybe_run_epoch_cosim" "audio_pretraining loads epoch cosim"
need_grep "$FAIRSEQ_ROOT/fairseq_cli/train.py" "task.end_epoch" "train.py calls task.end_epoch"
need_grep "$FAIRSEQ_ROOT/examples/data2vec/models/data2vec_audio.py" "extract_cosim_epoch_features" "data2vec_audio has extract_cosim_epoch_features"
need_grep "$FAIRSEQ_ROOT/examples/data2vec/cosim_epoch_utils.py" "def maybe_run_epoch_cosim" "cosim_epoch_utils has maybe_run_epoch_cosim"
echo ""

echo "--- epoch_cosim_subset_path in YAML vs file on disk ---"
if [[ -f "$YAML" ]]; then
  SUBSET=$(grep -E '^\s*epoch_cosim_subset_path:' "$YAML" | head -1 | sed -E 's/.*epoch_cosim_subset_path:\s*//;s/\s*#.*//;s/\s+$//' | tr -d '"' || true)
  if [[ -z "${SUBSET:-}" ]]; then
    say_fail "could not parse epoch_cosim_subset_path from YAML"
  else
    echo "  YAML epoch_cosim_subset_path=$SUBSET"
    if [[ -f "$SUBSET" ]]; then
      say_ok "subset file exists at path from YAML"
    elif [[ "$SUBSET" == /storage/* && "$FAIRSEQ_ROOT" == /mnt5/* ]]; then
      ALT="${SUBSET//\/storage\//\/mnt5\/}"
      if [[ -f "$ALT" ]]; then
        say_ok "subset exists at local mirror $ALT (YAML uses /storage for RunAI)"
      else
        say_fail "subset missing: $SUBSET (and local $ALT)"
      fi
    else
      say_fail "subset file NOT found at YAML path (copy .npy or fix YAML)"
    fi
  fi
fi
echo ""

echo "--- NumPy subset readable ---"
if [[ -f "$NPY" ]]; then
  if python3 -c "import numpy as np; a=np.load('$NPY'); assert len(a)>0; print('  OK   npy len=', len(a))" 2>/dev/null; then
    :
  else
    say_fail "cannot load $NPY with numpy"
  fi
else
  say_fail "skip numpy check (npy missing)"
fi
echo ""

if [[ "$ERR" -eq 0 ]]; then
  echo "=== All checks passed. Safe to submit (submit_recon_loss_experiments.sh). ==="
  exit 0
else
  echo "=== Some checks failed — fix above, then re-run. See recon_loss/HOW_TO_RUN.md ==="
  exit 1
fi
