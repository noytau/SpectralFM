#!/usr/bin/env bash
# Copy finished signal-recon grid checkpoints from RunAI PVC into local experiments tree.
#
# 1) In-cluster: copy only run dirs that contain *.pt from
#    /storage/.../fairseq/outputs/signal_recon_100k_grid/ →
#    /storage/.../autoencoder_experiments/signal_recon_100k_grid_runai/<run_dir>/
# 2) To this machine: kubectl cp that tree into REPO/autoencoder_experiments/
#
# Requires: kubectl + kube context (e.g. export KUBECONFIG=~/.kube/config)
# Pick any Running training pod that mounts the ``storage`` PVC at /storage (default: first sfm-sr-tr-*).
#
# Usage:
#   bash fairseq/copy_signal_recon_grid_ckpts_from_runai.sh
#   RUNAI_NS=runai-raja RUNAI_POD=sfm-sr-tr-s100k-lr1e4-0-0 bash fairseq/copy_signal_recon_grid_ckpts_from_runai.sh
set -euo pipefail

REPO="${REPO:-/mnt5/noy/SpectralFM}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
export KUBECONFIG
NS="${RUNAI_NS:-runai-raja}"
POD="${RUNAI_POD:-}"
if [[ -z "$POD" ]]; then
  POD="$(kubectl get pods -n "$NS" --field-selector=status.phase=Running -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep -E '^sfm-sr-' | head -1 || true)"
fi
if [[ -z "$POD" ]]; then
  echo "ERROR: set RUNAI_POD to a Running pod with /storage (namespace $NS), or submit a sfm-sr-* job." >&2
  exit 1
fi

REMOTE_GRID="/storage/noy/SpectralFM/fairseq/outputs/signal_recon_100k_grid"
REMOTE_DST="/storage/noy/SpectralFM/autoencoder_experiments/signal_recon_100k_grid_runai"
LOCAL_DST="${REPO}/autoencoder_experiments/signal_recon_100k_grid_runai"

echo "Using pod $NS/$POD"
cat << 'SCRIPT' | kubectl exec -i -n "$NS" "$POD" -- bash
set -euo pipefail
SRC="/storage/noy/SpectralFM/fairseq/outputs/signal_recon_100k_grid"
DST="/storage/noy/SpectralFM/autoencoder_experiments/signal_recon_100k_grid_runai"
mkdir -p "$DST"
for d in "$SRC"/*/; do
  shopt -s nullglob
  pts=( "$d"*.pt )
  shopt -u nullglob
  (( ${#pts[@]} == 0 )) && continue
  name="$(basename "${d%/}")"
  mkdir -p "$DST/$name"
  cp -a "${pts[@]}" "$DST/$name/"
  echo "[copied] $name"
done
ls -la "$DST"
SCRIPT

mkdir -p "$(dirname "$LOCAL_DST")"
echo "kubectl cp → $LOCAL_DST"
kubectl cp -n "$NS" "$POD:$REMOTE_DST" "$LOCAL_DST"
echo "Done."
