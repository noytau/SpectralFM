#!/usr/bin/env bash
# Copy structured_similarity_single_channel_all.npy into a RunAI shell pod at /storage/...
# Prerequisites: kubectl works (KUBECONFIG set), source file exists on this machine.
#
# Usage:
#   ./copy_epoch_cosim_subset_to_shell.sh
#   ./copy_epoch_cosim_subset_to_shell.sh runai-raja spectralfm-shell-0-6
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAIRSEQ_ROOT="$SCRIPT_DIR"

SRC="${SRC:-$FAIRSEQ_ROOT/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy}"
DST="/storage/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy"

if [[ ! -f "$SRC" ]]; then
  echo "Missing: $SRC"
  echo "Generate with code/precompute_epoch_cosim_indices.py (see recon_loss/HOW_TO_RUN.md)"
  exit 1
fi

if [[ -n "${1:-}" && -n "${2:-}" ]]; then
  NS="$1"
  POD="$2"
else
  LINE="$(kubectl get pods -A 2>/dev/null | awk '/spectralfm-shell/ && /Running/ {print $1, $2; exit}')"
  if [[ -z "${LINE:-}" ]]; then
    echo "Could not find a Running pod matching 'spectralfm-shell'. Set namespace and pod explicitly:"
    echo "  $0 <namespace> <pod-name>"
    echo "Example: $0 runai-raja spectralfm-shell-0-6"
    exit 1
  fi
  NS="$(echo "$LINE" | awk '{print $1}')"
  POD="$(echo "$LINE" | awk '{print $2}')"
fi

echo "Using pod: $NS/$POD"
echo "  SRC: $SRC"
echo "  DST: $DST"

kubectl exec -n "$NS" "$POD" -- mkdir -p "$(dirname "$DST")"
kubectl cp "$SRC" "${POD}:${DST}" -n "$NS"
kubectl exec -n "$NS" "$POD" -- ls -la "$DST"
echo "Done."
