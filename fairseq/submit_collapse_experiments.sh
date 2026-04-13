#!/usr/bin/env bash
# =============================================================================
# RunAI submit script for fe_vs_transformer_collapse experiments
#
# SETUP — fill in these two variables before running:
#
#   IMAGE : get from your existing shell pod:
#       runai describe job spectralfm-shell-g | grep -i image
#
#   PVC   : get from your existing shell pod:
#       runai describe job spectralfm-shell-g | grep -i pvc
#
# Then run:   bash submit_collapse_experiments.sh
# Watch:      watch -n 30 'runai list jobs'
# =============================================================================

IMAGE="noyhassid/spectralfm-lean:v6"   # e.g. nvcr.io/nvidia/pytorch:24.01-py3
PVC="storage"         # e.g. spectralfm-pvc

FAIRSEQ_ROOT="/storage/noy/SpectralFM/fairseq"
CONDA_ENV="spectralfm"
CONFIG_DIR="examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse"
CONFIG_NAME="spectralfm_collapse_ablation"

# Shared base command prefix
BASE_CMD="cd ${FAIRSEQ_ROOT} && conda run --no-capture-output -n ${CONDA_ENV} \
  python fairseq_cli/hydra_train.py \
  --config-dir ${CONFIG_DIR} \
  --config-name ${CONFIG_NAME}"

# Short-run overrides (1k steps) — batch_size/update_freq set per-experiment below
SHORT_OVERRIDES="optimization.max_update=1000 \
  lr_scheduler.max_update=1000 \
  task.data=${FAIRSEQ_ROOT}/data/nova_data/single_channel_one \
  dataset.disable_validation=true \
  checkpoint.save_interval_updates=500 \
  checkpoint.keep_interval_updates=1"

# Long-run overrides (20k steps) — batch_size/update_freq set per-experiment below
LONG_OVERRIDES="optimization.max_update=20000 \
  lr_scheduler.max_update=20000 \
  task.data=${FAIRSEQ_ROOT}/data/nova_data/single_channel_one \
  dataset.disable_validation=true \
  checkpoint.save_interval_updates=2000 \
  checkpoint.keep_interval_updates=2"

# Common model flags per experiment
# fe-train: batch=512 on A6000 (48GB) uses ~26 GB — confirmed working
FE_TRAIN="model.fe_mode=train model.freeze_encoder=false model.lambda_var=0.0 model.lambda_cov=0.0 dataset.batch_size=512 optimization.update_freq=[4]"
# fe-identity: bypass the CNN FE entirely — feed raw frames directly to the Transformer.
#   All data files are 245 frames. CNN FE (k=3×4, k=5 s=5) compresses 245→T'=47 tokens.
#   IdentityFEProjection has NO striding, so without capping max_sample_size the
#   Transformer would receive T=245 tokens instead of T'=47 — a 5× longer sequence,
#   meaning 27× more attention memory (O(T²)), which OOMs even on A6000 at batch=32.
#
#   Fix (Option A): cap max_sample_size=47 so the Transformer sees exactly 47 tokens
#   in both fe-train and fe-identity. The comparison is then fair:
#     fe-train:    245 frames → CNN FE (k=3×4, k=5 s=5) → T'=47 rich representations → Transformer
#     fe-identity:  47 frames → linear proj (no stride) →  T=47 raw projections → Transformer
#   T' derivation: 245→243→241→239→237→floor((237-5)/5+1)=47
#   Same token count = same memory footprint = same batch_size=512 and update_freq=[4].
FE_IDENTITY="model.fe_mode=identity model.freeze_encoder=false model.lambda_var=0.0 model.lambda_cov=0.0 dataset.batch_size=512 optimization.update_freq=[4] task.max_sample_size=47 task.min_sample_size=10"
# trans-frozen: encoder frozen → no transformer activations stored for backward → light.
#   batch=512 fine on A6000 (needs A6000; OOMs on A5000 24GB).
FE_TRAIN_TRANS_FROZEN="model.fe_mode=train model.freeze_encoder=true model.lambda_var=0.0 model.lambda_cov=0.0 dataset.batch_size=512 optimization.update_freq=[4]"

submit_job() {
    local JOB_NAME="$1"
    local CMD="$2"

    # Skip if already Running — don't touch a healthy job
    local STATUS
    STATUS=$(runai list jobs 2>/dev/null | awk -v name="$JOB_NAME" '$1 == name {print $2}')
    if [[ "$STATUS" == "Running" ]]; then
        echo "Skipping $JOB_NAME (already Running)"
        echo ""
        return
    fi

    # Delete stale job (OOMing / Failed / Pending) before resubmitting
    if [[ -n "$STATUS" ]]; then
        echo "Deleting stale $JOB_NAME (status: $STATUS)"
        runai delete job "$JOB_NAME" 2>/dev/null || true
        sleep 2
    fi

    echo "Submitting: $JOB_NAME"
    runai submit "$JOB_NAME" \
        --image "$IMAGE" \
        --gpu-memory 40G \
        --node-pools faculty,raja \
        --existing-pvc "claimname=${PVC},path=/storage" \
        --command -- bash -c "$CMD"
    echo ""
}

# =============================================================================
# SHORT EXPERIMENTS — 1k steps, single_channel_100
# Expected duration: ~20–40 min each
# =============================================================================

echo "=== Submitting SHORT experiments (1k steps) ==="

submit_job "sfm-short-fe-train" \
  "${BASE_CMD} ${FE_TRAIN} ${SHORT_OVERRIDES} \
   common.wandb_run_name=short_fe-train_trans-train_base"

submit_job "sfm-short-fe-identity" \
  "${BASE_CMD} ${FE_IDENTITY} ${SHORT_OVERRIDES} \
   common.wandb_run_name=short_fe-identity_trans-train_base"

submit_job "sfm-short-trans-frozen" \
  "${BASE_CMD} ${FE_TRAIN_TRANS_FROZEN} ${SHORT_OVERRIDES} \
   common.wandb_run_name=short_fe-train_trans-frozen_base"

# =============================================================================
# LONG EXPERIMENTS — 20k steps, single_channel_one, bs=512 eff=2048
# Expected duration: ~8–14 hours each
# =============================================================================

echo "=== Submitting LONG experiments (20k steps) ==="

submit_job "sfm-long-fe-train" \
  "${BASE_CMD} ${FE_TRAIN} ${LONG_OVERRIDES} \
   common.wandb_run_name=long_fe-train_trans-train_base"

submit_job "sfm-long-fe-identity" \
  "${BASE_CMD} ${FE_IDENTITY} ${LONG_OVERRIDES} \
   common.wandb_run_name=long_fe-identity_trans-train_base"

submit_job "sfm-long-trans-frozen" \
  "${BASE_CMD} ${FE_TRAIN_TRANS_FROZEN} ${LONG_OVERRIDES} \
   common.wandb_run_name=long_fe-train_trans-frozen_base"

echo "=== All 6 jobs submitted ==="
echo ""
echo "Monitor with:"
echo "  watch -n 30 'runai list jobs'"
echo "  runai logs sfm-short-fe-train --tail 20"
echo ""
echo "WandB project: spectralfm_fe_vs_transformer_collapse"
