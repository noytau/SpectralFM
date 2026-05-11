#!/usr/bin/env bash
# RunAI — Reconstruction loss experiments. Shared settings: recon_loss/spectralfm_recon_loss.yaml
#
#   bash submit_recon_loss_experiments.sh
#
# Logs:
#   - RunAI:  runai logs -f <job-name>   (stdout/stderr from fairseq json logs)
#   - W&B:    project spectralfm_recon_loss (common.wandb_* in YAML); ensure WANDB_API_KEY in the cluster/image
#
# GPU: 40G per job (--gpu-memory 40G). Node pools: faculty,raja
#
IMAGE="noyhassid/spectralfm-lean:v6"
PVC="storage"

FAIRSEQ_ROOT="/storage/noy/SpectralFM/fairseq"
CONDA_ENV="spectralfm"
CONFIG_DIR="examples/data2vec/config/audio/pretraining/recon_loss"
CONFIG_NAME="spectralfm_recon_loss"

# Epoch cosim: full structured panel (~100 samples, same as short_smoke_struct100_wandb / HOW_TO_RUN mode B).
# Requires structured_similarity_full.json on PVC (precompute_epoch_cosim_indices.py --structured_entries_json).
RUNAI_DEFAULTS="dataset.disable_validation=true model.lambda_recon_fe=1.0 model.epoch_cosim_structured_entries_path=/storage/noy/SpectralFM/fairseq/examples/data2vec/config/audio/pretraining/recon_loss/structured_similarity_full.json distributed_training.distributed_world_size=1"

# One physical GPU per job (Fairseq DDP disabled via world_size=1 above).
BASE_CMD="cd ${FAIRSEQ_ROOT} && export CUDA_VISIBLE_DEVICES=0 && conda run --no-capture-output -n ${CONDA_ENV} \
  python fairseq_cli/hydra_train.py \
  --config-dir ${CONFIG_DIR} \
  --config-name ${CONFIG_NAME}"

submit_job() {
    local JOB_NAME="$1"
    local GPU_MEM="$2"
    local EXTRA="$3"

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

    echo "Submitting: $JOB_NAME  (gpu-memory=${GPU_MEM})"
    runai submit "$JOB_NAME" \
        --image "$IMAGE" \
        --gpu-memory "$GPU_MEM" \
        --node-pools faculty,raja \
        --existing-pvc "claimname=${PVC},path=/storage" \
        --command -- bash -c "${BASE_CMD} ${RUNAI_DEFAULTS} ${EXTRA}"
    echo ""
}

echo "=== Submitting reconstruction loss experiments Exp1–Exp3 (batch=512, update_freq=4), 40G GPU ==="
echo ""

# Exp1: YAML defaults (λ_fe=1, λ_trans=0, freeze_encoder=false)
submit_job "sfm-recon-exp1-fe" "40G" \
  "common.wandb_run_name=recon_exp1_fe-recon_pretrained_lr1e-5"

submit_job "sfm-recon-exp2-fe-tr" "40G" \
  "model.lambda_recon_trans=1.0 common.wandb_run_name=recon_exp2_fe-tr-recon_pretrained_lr1e-5"

submit_job "sfm-recon-exp3-frozen" "40G" \
  "model.freeze_encoder=true common.wandb_run_name=recon_exp3_fe-recon_frozen-trans"

# Exp4 (optional): train FE only — uncomment to submit a 4th job
# submit_job "sfm-recon-exp4-train-fe-only" "40G" \
#   "model.train_only_fe=true common.wandb_run_name=recon_exp4_train-fe-only_lambda-fe1"

echo "=== Done ==="
echo "WandB project: spectralfm_recon_loss  |  watch jobs:  watch -n 30 'runai list jobs'"
