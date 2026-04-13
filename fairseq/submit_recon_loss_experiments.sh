#!/usr/bin/env bash
# RunAI — Reconstruction loss experiments. Shared settings: recon_loss/spectralfm_recon_loss.yaml
#   bash submit_recon_loss_experiments.sh
#
IMAGE="noyhassid/spectralfm-lean:v6"
PVC="storage"

FAIRSEQ_ROOT="/storage/noy/SpectralFM/fairseq"
CONDA_ENV="spectralfm"
CONFIG_DIR="examples/data2vec/config/audio/pretraining/recon_loss"
CONFIG_NAME="spectralfm_recon_loss"

# Epoch cosim: enable via YAML model.epoch_cosim_* only when PVC data2vec_audio.py dataclass includes those fields.
# Override stale PVC: skip dry-run over full train-as-valid; recon λ if YAML on /storage is old.
RUNAI_DEFAULTS="dataset.disable_validation=true model.lambda_recon_fe=1.0"

BASE_CMD="cd ${FAIRSEQ_ROOT} && conda run --no-capture-output -n ${CONDA_ENV} \
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

echo "=== Submitting reconstruction loss experiments Exp1–Exp4 (batch=512, update_freq=4) ==="
echo ""

# Exp1: YAML defaults (λ_fe=1, λ_trans=0, freeze_encoder=false)
submit_job "sfm-recon-exp1-fe" "40G" \
  "common.wandb_run_name=recon_exp1_fe-recon_pretrained_lr1e-5"

submit_job "sfm-recon-exp2-fe-tr" "40G" \
  "model.lambda_recon_trans=1.0 common.wandb_run_name=recon_exp2_fe-tr-recon_pretrained_lr1e-5"

submit_job "sfm-recon-exp3-frozen" "40G" \
  "model.freeze_encoder=true common.wandb_run_name=recon_exp3_fe-recon_frozen-trans"

# Exp4: train feature extractor stack only (transformer frozen by train_only_fe); FE + trans recon λ unchanged
submit_job "sfm-recon-exp4-train-fe-only" "40G" \
  "model.train_only_fe=true common.wandb_run_name=recon_exp4_train-fe-only_lambda-fe1"

echo "=== Done ==="
echo "WandB: spectralfm_recon_loss"
