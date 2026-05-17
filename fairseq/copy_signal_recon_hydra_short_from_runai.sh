#!/usr/bin/env bash
# Copy the latest Hydra short signal-recon run from RunAI PVC → local checkpoints/runai.
#
# Usage:
#   bash fairseq/copy_signal_recon_hydra_short_from_runai.sh
#   NS=runai-raja POD=spectralfm-shell-g bash fairseq/copy_signal_recon_hydra_short_from_runai.sh
#
set -euo pipefail

NS="${NS:-runai-raja}"
POD="${POD:-spectralfm-shell-g}"
SRC_GLOB="/storage/noy/SpectralFM/fairseq/outputs/recon_loss/*_tr_signal_short"
LOCAL_DST="/mnt5/noy/SpectralFM/checkpoints/runai/hydra_tr_signal_short"

if ! kubectl get pod -n "$NS" "$POD" &>/dev/null; then
  echo "ERROR: pod $NS/$POD not found. Set NS/POD (e.g. spectralfm-shell-g)."
  exit 1
fi

mkdir -p "$LOCAL_DST"

cat << SCRIPT | kubectl exec -i -n "$NS" "$POD" -- bash
set -euo pipefail
RUN_DIR=\$(ls -td ${SRC_GLOB} 2>/dev/null | head -1 || true)
if [[ -z "\$RUN_DIR" ]]; then
  echo "ERROR: no run dir matching ${SRC_GLOB}"
  exit 2
fi
CKPT_DIR="\$RUN_DIR/checkpoints"
echo "RUN_DIR=\$RUN_DIR"
if [[ -f "\$CKPT_DIR/checkpoint_best.pt" ]]; then
  SRC="\$CKPT_DIR/checkpoint_best.pt"
else
  SRC=\$(ls -t "\$CKPT_DIR"/checkpoint_[0-9]*.pt 2>/dev/null | head -1)
fi
[[ -f "\$SRC" ]] || { echo "ERROR: no checkpoint in \$CKPT_DIR"; ls -la "\$CKPT_DIR" 2>/dev/null; exit 3; }
STAMP=\$(basename "\$RUN_DIR" | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}' || echo unknown)
DST="/storage/noy/SpectralFM/checkpoints/runai/hydra_tr_signal_short_\${STAMP}.pt"
mkdir -p /storage/noy/SpectralFM/checkpoints/runai
cp "\$SRC" "\$DST"
echo "COPIED \$SRC -> \$DST"
ls -lh "\$DST"
SCRIPT

LATEST=$(ls -t /mnt5/noy/SpectralFM/checkpoints/runai/hydra_tr_signal_short_*.pt 2>/dev/null | head -1 || true)
if [[ -n "$LATEST" ]]; then
  echo "Local (after NFS sync): $LATEST"
else
  echo "Note: copy ran in-cluster; wait for NFS sync or kubectl cp the file from \$DST above."
fi
