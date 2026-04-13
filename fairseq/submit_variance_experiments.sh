#!/usr/bin/env bash
# =============================================================================
# RunAI submit script — Regularization-loss ablations (6 long runs, 20k steps)
#
# Variance loss:    loss = data2vec_loss + 1.0 * variance_loss(embeddings, gamma=1.0)
# Uniformity loss:  loss = data2vec_loss + 1.0 * uniformity_loss(embeddings)
#
# Both sets run the same 3 architecture variants:
#   1. fe-identity_trans-train  (fe_mode=identity, freeze_encoder=false)
#   2. fe-train_trans-train     (fe_mode=train,    freeze_encoder=false)
#   3. fe-train_trans-frozen    (fe_mode=train,    freeze_encoder=true)
#
# Dataset: single_channel_all, 20k steps
#
# Usage:
#   bash submit_variance_experiments.sh
# Watch:
#   watch -n 30 'runai list jobs'
# =============================================================================

IMAGE="noyhassid/spectralfm-lean:v6"
PVC="storage"

FAIRSEQ_ROOT="/storage/noy/SpectralFM/fairseq"
CONDA_ENV="spectralfm"
CONFIG_DIR="examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse"
CONFIG_NAME="spectralfm_collapse_ablation"

BASE_CMD="cd ${FAIRSEQ_ROOT} && conda run --no-capture-output -n ${CONDA_ENV} \
  python fairseq_cli/hydra_train.py \
  --config-dir ${CONFIG_DIR} \
  --config-name ${CONFIG_NAME}"

# Long-run overrides: 20k steps, single_channel_one dataset
LONG_OVERRIDES="optimization.max_update=20000 \
  lr_scheduler.max_update=20000 \
  task.data=${FAIRSEQ_ROOT}/data/nova_data/single_channel_all \
  dataset.disable_validation=true \
  checkpoint.save_interval_updates=2000 \
  checkpoint.keep_interval_updates=2"

# Variance-loss knobs
VAR_LOSS="model.lambda_var=1.0 model.var_gamma=1.0 model.lambda_cov=0.0 model.lambda_uniform=0.0"

# Uniformity-loss knobs
UNIFORM_LOSS="model.lambda_var=0.0 model.lambda_cov=0.0 model.lambda_uniform=1.0"

# Per-experiment model flags (same batch/update_freq as baseline collapse runs)
FE_IDENTITY_VAR="model.fe_mode=identity model.freeze_encoder=false \
  dataset.batch_size=512 optimization.update_freq=[4] \
  task.max_sample_size=47 task.min_sample_size=10 \
  ${VAR_LOSS}"

FE_TRAIN_VAR="model.fe_mode=train model.freeze_encoder=false \
  dataset.batch_size=512 optimization.update_freq=[4] \
  ${VAR_LOSS}"

FE_TRAIN_TRANS_FROZEN_VAR="model.fe_mode=train model.freeze_encoder=true \
  dataset.batch_size=512 optimization.update_freq=[4] \
  ${VAR_LOSS}"

submit_job() {
    local JOB_NAME="$1"
    local CMD="$2"

    local STATUS
    STATUS=$(runai list jobs 2>/dev/null | awk -v name="$JOB_NAME" '$1 == name {print $2}')
    if [[ "$STATUS" == "Running" ]]; then
        echo "Skipping $JOB_NAME (already Running)"
        echo ""
        return
    fi

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

echo "=== Submitting LONG variance-loss experiments (20k steps, lambda_var=1.0, gamma=1.0) ==="

# 1. fe-identity / transformer-train
submit_job "sfm-long-id-var" \
  "${BASE_CMD} ${FE_IDENTITY_VAR} ${LONG_OVERRIDES} \
   common.wandb_run_name=long_fe-identity_trans-train_lv1.0"

# 2. fe-train / transformer-train
submit_job "sfm-long-train-var" \
  "${BASE_CMD} ${FE_TRAIN_VAR} ${LONG_OVERRIDES} \
   common.wandb_run_name=long_fe-train_trans-train_lv1.0"

# 3. fe-train / transformer-frozen
submit_job "sfm-long-frozen-var" \
  "${BASE_CMD} ${FE_TRAIN_TRANS_FROZEN_VAR} ${LONG_OVERRIDES} \
   common.wandb_run_name=long_fe-train_trans-frozen_lv1.0"

# =============================================================================
# UNIFORMITY LOSS — 3 long jobs, lambda_uniform=1.0
# =============================================================================

FE_IDENTITY_UNIFORM="model.fe_mode=identity model.freeze_encoder=false \
  dataset.batch_size=512 optimization.update_freq=[4] \
  task.max_sample_size=47 task.min_sample_size=10 \
  ${UNIFORM_LOSS}"

FE_TRAIN_UNIFORM="model.fe_mode=train model.freeze_encoder=false \
  dataset.batch_size=512 optimization.update_freq=[4] \
  ${UNIFORM_LOSS}"

FE_TRAIN_TRANS_FROZEN_UNIFORM="model.fe_mode=train model.freeze_encoder=true \
  dataset.batch_size=512 optimization.update_freq=[4] \
  ${UNIFORM_LOSS}"

echo "=== Submitting LONG uniformity-loss experiments (20k steps, lambda_uniform=1.0) ==="

# 4. fe-identity / transformer-train
submit_job "sfm-long-id-unif" \
  "${BASE_CMD} ${FE_IDENTITY_UNIFORM} ${LONG_OVERRIDES} \
   common.wandb_run_name=long_fe-identity_trans-train_lu1.0"

# 5. fe-train / transformer-train
submit_job "sfm-long-train-unif" \
  "${BASE_CMD} ${FE_TRAIN_UNIFORM} ${LONG_OVERRIDES} \
   common.wandb_run_name=long_fe-train_trans-train_lu1.0"

# 6. fe-train / transformer-frozen
submit_job "sfm-long-frozen-unif" \
  "${BASE_CMD} ${FE_TRAIN_TRANS_FROZEN_UNIFORM} ${LONG_OVERRIDES} \
   common.wandb_run_name=long_fe-train_trans-frozen_lu1.0"

echo "=== All 6 regularization-loss jobs submitted ==="
echo ""
echo "Monitor with:"
echo "  watch -n 30 'runai list jobs'"
echo ""
echo "Variance jobs:    sfm-long-id-var / sfm-long-train-var / sfm-long-frozen-var"
echo "Uniformity jobs:  sfm-long-id-unif / sfm-long-train-unif / sfm-long-frozen-unif"
echo ""
echo "WandB project: spectralfm_fe_vs_transformer_collapse"
