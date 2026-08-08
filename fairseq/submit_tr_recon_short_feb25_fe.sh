#!/usr/bin/env bash
# RunAI — short transformer signal-reconstruction (code/train_reconstruction.py).
#
# FE weights: Feb-25 long-train checkpoint on PVC (evaluated as strong single-ch run in EVALUATION_FLOW).
#   /storage/noy/SpectralFM/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt
# Mode: --ckpt none --fe_ckpt <above> --freeze_fe  →  pretrained conv FE + extractor LN frozen; train transformer + head.
#
# Train: 1000 samples, 1000 steps, --batch_size 512 --grad_accum_steps 4 (micro-batch 128, effective 512/update).
#
# **PVC must include the repo `code/` tree** (RunAI often only has `fairseq/` under SpectralFM). If the job exits
# with "can't open file .../code/train_reconstruction.py", copy from your workstation, e.g.:
#   kubectl cp /mnt5/noy/SpectralFM/code <NS>/<POD>:/storage/noy/SpectralFM/
# or at minimum:  .../code/train_reconstruction.py  →  .../storage/noy/SpectralFM/code/
#
# GPU (transformer path — required): schedule **only** on **NVIDIA A5000 or A6000** class nodes.
#   Default batch (--batch_size 512 --grad_accum_steps 4) needs the VRAM / bandwidth of those GPUs;
#   do **not** rely on pools that can place the job on consumer cards (e.g. RTX 2080 Ti) — expect OOM or invalid runs.
#   1) Point NODE_POOLS only at pools your admins guarantee are A5000/A6000 (default: faculty,raja — verify for your tenant).
#   2) Optional **pin** via ``RUNAI_NODE_TYPE=A6000`` or ``A5000`` (must match cluster ``--node-type`` labels).
#      If **unset**, no ``--node-type`` is passed — use when A6000-only affinity keeps the job Pending; still keep
#      NODE_POOLS limited to A5000/A6000 pools so you do not land on consumer GPUs.
#
# Usage:
#   bash fairseq/submit_tr_recon_short_feb25_fe.sh
#   runai logs -f sfm-tr-recon-1k-feb25fe

set -euo pipefail

IMAGE="${IMAGE:-noyhassid/spectralfm-lean:v6}"
PVC="${PVC:-storage}"
JOB_NAME="${JOB_NAME:-sfm-tr-recon-1k-feb25fe}"

REPO="/storage/noy/SpectralFM"
SPEC_CODE="${SPEC_CODE:-${REPO}/code}"
FE_CKPT="${FE_CKPT:-${REPO}/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt}"
MANIFEST="${MANIFEST:-${REPO}/fairseq/data/nova_data/single_channel_one/train_runai_150k.tsv}"
OUT_PARENT="${OUT_PARENT:-${REPO}/fairseq/outputs/tr_recon_short_feb25fe}"
STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
OUT_DIR="${OUT_PARENT}/${STAMP}"

CONDA_ENV="${CONDA_ENV:-spectralfm}"
NODE_POOLS="${NODE_POOLS:-faculty,raja}"
GPU_MEM="${GPU_MEM:-40G}"

NODE_TYPE_ARGS=()
if [[ -n "${RUNAI_NODE_TYPE:-}" ]]; then
  NODE_TYPE_ARGS=(--node-type "${RUNAI_NODE_TYPE}")
fi

STATUS=$(runai list jobs 2>/dev/null | awk -v name="$JOB_NAME" '$1 == name {print $2}' || true)
if [[ "$STATUS" == "Running" ]]; then
  echo "Job $JOB_NAME already Running — not resubmitting."
  exit 0
fi
if [[ -n "$STATUS" ]]; then
  echo "Deleting stale job $JOB_NAME (status: $STATUS)"
  runai delete job "$JOB_NAME" 2>/dev/null || true
  sleep 2
fi

# One line for RunAI -- bash -c (no nested quote breaks)
REMOTE="set -euo pipefail; TR=\"${SPEC_CODE}/train_reconstruction.py\"; if [[ ! -f \"\$TR\" ]]; then echo \"ERROR: missing \$TR on PVC. Sync e.g.: kubectl cp /mnt5/noy/SpectralFM/code <pod>:${REPO}/\"; exit 2; fi; cd ${REPO} && export PYTHONPATH=${REPO}/code:${REPO}/fairseq:${REPO}/fairseq/examples CUDA_VISIBLE_DEVICES=0 && mkdir -p ${OUT_DIR} && conda run --no-capture-output -n ${CONDA_ENV} python3 \"\$TR\" --mode train --recon_path transformer --ckpt none --fe_ckpt ${FE_CKPT} --freeze_fe --manifest ${MANIFEST} --n_samples 1000 --steps 1000 --batch_size 512 --grad_accum_steps 4 --lr 1e-4 --warmup 100 --wandb_project spectralfm-runai-tr-recon-short --wandb_run_name tr_1k_feb25fe_freezeFE_bs512_ga4 --run_suffix feb25_runai_fe_${STAMP} --out_dir ${OUT_DIR} --device cuda"

if [[ -z "${RUNAI_NODE_TYPE:-}" ]]; then
  echo "[i] RUNAI_NODE_TYPE unset → no --node-type pin (NODE_POOLS=${NODE_POOLS}). Set RUNAI_NODE_TYPE=A5000|A6000 to pin."
fi
echo "Submitting $JOB_NAME  gpu-memory=${GPU_MEM}  node-pools=${NODE_POOLS}  node-type=${RUNAI_NODE_TYPE:-<none>}"
echo "OUT_DIR=${OUT_DIR}"
echo "FE_CKPT=${FE_CKPT}"

runai submit "$JOB_NAME" \
  --image "$IMAGE" \
  --gpu-memory "$GPU_MEM" \
  --node-pools "$NODE_POOLS" \
  "${NODE_TYPE_ARGS[@]}" \
  --existing-pvc "claimname=${PVC},path=/storage" \
  --command -- bash -c "$REMOTE"

echo ""
echo "Submitted.  runai logs -f ${JOB_NAME}"
