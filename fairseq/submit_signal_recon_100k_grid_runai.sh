#!/usr/bin/env bash
# RunAI — 8× long signal-reconstruction runs via code/train_reconstruction.py (lazy manifest).
#
# Grid (2 × 2 × 2 = 8 jobs)
#   A) Transformer: frozen conv FE + extractor LN from Feb-25 Fairseq ckpt; train encoder + head
#      (--recon_path transformer --ckpt none --fe_ckpt … --freeze_fe)
#   B) FE path: CNN FE + LN + MirrorDecoder; FE (+LN) init from same Fairseq ckpt; full AE trains
#      (--recon_path fe --ckpt …)
#   Shared: n_samples=100000, steps ∈ {10000, 100000}, cosine LR (warmup 10% of steps),
#           lr ∈ {1e-4, 5e-4}, batch 512 / grad_accum 4, lazy manifest under single_channel_one.
#
# Prerequisites on PVC (/storage/noy/SpectralFM)
#   1) Repo **code/** tree (RunAI often only has fairseq/). If missing:
#        kubectl cp /mnt5/noy/SpectralFM/code <NS>/<shell-pod>:/storage/noy/SpectralFM/
#   2) Feb-25 FE source checkpoint (both modes use it):
#        …/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt
#   3) Manifest on PVC: ``train_runai_150k.tsv`` (first line = ``/storage/.../single_channel_one/wavs``; 150k
#      ``spectra*.wav`` rows). The checked-in ``train.tsv`` in that folder targets ``/mnt5/noy/.../single_channel_1m``
#      and different filenames — it will **fail on RunAI** with ``soundfile.LibsndfileError``.
#   4) W&B: set WANDB_API_KEY in the image/job env, or omit logging by exporting WANDB_PROJECT=""
#   5) Transformer jobs: use **professional** GPUs only (A5000/A6000 for default batch). **Scheduling:**
#        - Leave ``RUNAI_NODE_TYPE`` **unset** (default) to **omit** ``--node-type`` and let ``NODE_POOLS`` pick any
#          free GPU in those pools — avoids indefinite Pending when only a few nodes carry ``run.ai/type=A6000``.
#        - To **pin** a SKU, set ``RUNAI_NODE_TYPE=A5000`` or ``A6000`` (must match cluster label values).
#
# Usage (from any machine with runai CLI + project context):
#   bash fairseq/submit_signal_recon_100k_grid_runai.sh
#   RUNAI_NODE_TYPE=A5000 bash fairseq/submit_signal_recon_100k_grid_runai.sh   # optional hard pin
#   watch -n 30 'runai list jobs -p raja | grep sfm-sr-'
#
set -euo pipefail

IMAGE="${IMAGE:-noyhassid/spectralfm-lean:v6}"
PVC="${PVC:-storage}"
REPO="/storage/noy/SpectralFM"
SPEC_CODE="${SPEC_CODE:-${REPO}/code}"
FE_CKPT="${FE_CKPT:-${REPO}/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt}"
MANIFEST="${MANIFEST:-${REPO}/fairseq/data/nova_data/single_channel_one/train_runai_150k.tsv}"
OUT_PARENT="${OUT_PARENT:-${REPO}/fairseq/outputs/signal_recon_100k_grid}"
STAMP="$(date -u +%Y%m%d_%H%M%SZ)"
CONDA_ENV="${CONDA_ENV:-spectralfm}"
NODE_POOLS="${NODE_POOLS:-faculty,raja}"
GPU_MEM="${GPU_MEM:-40G}"
WANDB_PROJECT="${WANDB_PROJECT:-spectralfm-runai-signal-recon-100k}"

N_SAMPLES="${N_SAMPLES:-100000}"
BATCH="${BATCH:-512}"
GA="${GA:-4}"

submit_one() {
  local JOB_NAME="$1"
  local REMOTE_CMD="$2"
  local ADD_NODE_TYPE="${3:-0}"

  local STATUS
  STATUS=$(runai list jobs 2>/dev/null | awk -v name="$JOB_NAME" '$1 == name {print $2}' || true)
  if [[ "$STATUS" == "Running" ]]; then
    echo "Skipping $JOB_NAME (already Running)"
    echo ""
    return 0
  fi
  if [[ -n "$STATUS" ]]; then
    echo "Deleting stale $JOB_NAME (status: $STATUS)"
    runai delete job "$JOB_NAME" 2>/dev/null || true
    sleep 2
  fi

  local NODE_TYPE_ARGS=()
  if [[ "$ADD_NODE_TYPE" == "1" ]]; then
    if [[ -n "${RUNAI_NODE_TYPE:-}" ]]; then
      NODE_TYPE_ARGS=(--node-type "${RUNAI_NODE_TYPE}")
    else
      echo "[i] $JOB_NAME (transformer): RUNAI_NODE_TYPE unset → no --node-type pin (NODE_POOLS=${NODE_POOLS})."
    fi
  fi

  echo "Submitting $JOB_NAME  gpu=${GPU_MEM}  pools=${NODE_POOLS}  node-type=${RUNAI_NODE_TYPE:-<unset>}"
  runai submit "$JOB_NAME" \
    --image "$IMAGE" \
    --gpu-memory "$GPU_MEM" \
    --node-pools "$NODE_POOLS" \
    "${NODE_TYPE_ARGS[@]}" \
    --existing-pvc "claimname=${PVC},path=/storage" \
    --command -- bash -c "$REMOTE_CMD"
  echo ""
  sleep 2
}

# --- Remote bodies: OUT, wandb, run_suffix expanded on host; TR path checked inside pod. ---
common_head="set -euo pipefail; TR=\"${SPEC_CODE}/train_reconstruction.py\"; if [[ ! -f \"\$TR\" ]]; then echo \"ERROR: missing \$TR on PVC. Sync: kubectl cp /mnt5/noy/SpectralFM/code <pod>:${REPO}/\"; exit 2; fi; if [[ ! -f \"${FE_CKPT}\" ]]; then echo \"ERROR: missing FE ckpt ${FE_CKPT}\"; exit 2; fi; cd ${REPO} && export PYTHONPATH=${REPO}/code:${REPO}/fairseq:${REPO}/fairseq/examples CUDA_VISIBLE_DEVICES=0"

wandb_args() {
  local name="$1"
  if [[ -n "${WANDB_PROJECT}" ]]; then
    echo -n "--wandb_project ${WANDB_PROJECT} --wandb_run_name ${name}"
  else
    echo -n ""
  fi
}

# Transformer jobs (4): fe_ckpt + freeze_fe; cosine via warmup>0
tr_remote() {
  local job="$1" steps="$2" lr="$3" warmup="$4"
  local OUT="${OUT_PARENT}/${STAMP}_${job}"
  local wb
  wb="$(wandb_args "$job")"
  echo "${common_head} && mkdir -p \"${OUT}\" && conda run --no-capture-output -n ${CONDA_ENV} python3 \"\$TR\" --mode train --recon_path transformer --ckpt none --fe_ckpt ${FE_CKPT} --freeze_fe --manifest ${MANIFEST} --n_samples ${N_SAMPLES} --steps ${steps} --batch_size ${BATCH} --grad_accum_steps ${GA} --lr ${lr} --warmup ${warmup} ${wb} --run_suffix feb25grid_${STAMP} --out_dir \"${OUT}\" --device cuda"
}

# FE CNN+decoder jobs (4): pretrained FE init via --ckpt (Fairseq .pt)
fe_remote() {
  local job="$1" steps="$2" lr="$3" warmup="$4"
  local OUT="${OUT_PARENT}/${STAMP}_${job}"
  local wb
  wb="$(wandb_args "$job")"
  echo "${common_head} && mkdir -p \"${OUT}\" && conda run --no-capture-output -n ${CONDA_ENV} python3 \"\$TR\" --mode train --recon_path fe --ckpt ${FE_CKPT} --manifest ${MANIFEST} --n_samples ${N_SAMPLES} --steps ${steps} --batch_size ${BATCH} --grad_accum_steps ${GA} --lr ${lr} --warmup ${warmup} ${wb} --run_suffix feb25grid_${STAMP} --out_dir \"${OUT}\" --device cuda"
}

echo "=== Signal recon 100k grid  STAMP=${STAMP}  OUT_PARENT=${OUT_PARENT} ==="
echo "FE_CKPT=${FE_CKPT}"
echo "MANIFEST=${MANIFEST}"
echo ""

# steps 10k / warmup 1k
submit_one "sfm-sr-tr-s10k-lr1e4"  "$(tr_remote sfm-sr-tr-s10k-lr1e4 10000 1e-4 1000)"  1
submit_one "sfm-sr-tr-s10k-lr5e4"  "$(tr_remote sfm-sr-tr-s10k-lr5e4 10000 5e-4 1000)"  1
submit_one "sfm-sr-fe-s10k-lr1e4"  "$(fe_remote sfm-sr-fe-s10k-lr1e4 10000 1e-4 1000)"  0
submit_one "sfm-sr-fe-s10k-lr5e4"  "$(fe_remote sfm-sr-fe-s10k-lr5e4 10000 5e-4 1000)"  0

# steps 100k / warmup 10k
submit_one "sfm-sr-tr-s100k-lr1e4" "$(tr_remote sfm-sr-tr-s100k-lr1e4 100000 1e-4 10000)" 1
submit_one "sfm-sr-tr-s100k-lr5e4" "$(tr_remote sfm-sr-tr-s100k-lr5e4 100000 5e-4 10000)" 1
submit_one "sfm-sr-fe-s100k-lr1e4" "$(fe_remote sfm-sr-fe-s100k-lr1e4 100000 1e-4 10000)" 0
submit_one "sfm-sr-fe-s100k-lr5e4" "$(fe_remote sfm-sr-fe-s100k-lr5e4 100000 5e-4 10000)" 0

echo "=== Done: 8 jobs submitted (or skipped if already Running). ==="
echo "Logs: runai logs -f <job-name>"
