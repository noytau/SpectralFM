#!/usr/bin/env bash
# RunAI — Reconstruction decoder architecture experiments (6 jobs, ~1k steps each)
#
#   bash submit_recon_decoder_experiments.sh
#
# All jobs use the reconstruction-loss-experiments branch on PVC.
# Dataset: single_channel_one (950k samples), effective batch=2048, ~464 iters/epoch.
# 1000 updates ≈ 2 epochs ≈ ~2M samples trained on.
# YAML default lr is [0.0001] (1e-4). Only lr1e5 overrides it.

IMAGE="noyhassid/spectralfm-lean:v6"
PVC="storage"

FAIRSEQ_ROOT="/storage/noy/SpectralFM/fairseq"
CONDA_ENV="spectralfm"
CONFIG_DIR="examples/data2vec/config/audio/pretraining/recon_loss"
CONFIG_NAME="spectralfm_recon_loss"

SHARED_OVERRIDES="\
  task.data=/storage/noy/SpectralFM/fairseq/data/nova_data/single_channel_one \
  common.user_dir=/storage/noy/SpectralFM/fairseq/examples/ \
  model.model_path=/storage/noy/SpectralFM/fairseq/base_libri_official.pt \
  model.epoch_cosim_subset_path=/storage/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy \
  model.epoch_cosim_structured_entries_path=/storage/noy/SpectralFM/fairseq/examples/data2vec/config/audio/pretraining/recon_loss/structured_similarity_full.json \
  dataset.disable_validation=true \
  distributed_training.distributed_world_size=1 \
  optimization.max_update=1000 \
  lr_scheduler.max_update=1000 \
  lr_scheduler.warmup_updates=100 \
  checkpoint.no_epoch_checkpoints=true \
  checkpoint.save_interval_updates=500 \
  checkpoint.keep_interval_updates=2"

# One physical GPU per job.
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
        echo "⏭  Skipping $JOB_NAME (already Running)"
        echo ""
        return
    fi

    if [[ -n "$STATUS" ]]; then
        echo "🗑  Deleting stale $JOB_NAME (status: $STATUS)"
        runai delete job "$JOB_NAME" 2>/dev/null || true
        sleep 2
    fi

    echo "▶  Submitting: $JOB_NAME  (gpu-memory=${GPU_MEM})"
    runai submit "$JOB_NAME" \
        --image "$IMAGE" \
        --gpu-memory "$GPU_MEM" \
        --node-pools faculty,raja \
        --existing-pvc "claimname=${PVC},path=/storage" \
        --command -- bash -c "${BASE_CMD} ${SHARED_OVERRIDES} ${EXTRA}"
    echo ""
}

echo "============================================================"
echo " Reconstruction Decoder Experiments (6 jobs, 1k steps each)"
echo " Dataset: single_channel_one (950k samples)"
echo " Effective batch: 2048 → ~464 iters/epoch → ~2 epochs"
echo " Samples trained on: ~2M"
echo "============================================================"
echo ""

# YAML default: lr=[0.0001], recon_decoder_type=mlp, recon_loss_type=l1
# Only override what differs per experiment.

# ── Exp 1: Conv1d decoder (temporal upsampling, no mean-pool) ──
submit_job "sfm-recon-conv1d" "40G" \
  "model.recon_decoder_type=conv1d \
   common.wandb_run_name=recon_conv1d_lr1e-4 \
   hydra.run.dir=/storage/noy/SpectralFM/fairseq/outputs/recon_loss/recon_conv1d_lr1e-4"

# ── Exp 2: Interpolation decoder (lightweight temporal, no mean-pool) ──
submit_job "sfm-recon-interp" "40G" \
  "model.recon_decoder_type=interp \
   common.wandb_run_name=recon_interp_lr1e-4 \
   hydra.run.dir=/storage/noy/SpectralFM/fairseq/outputs/recon_loss/recon_interp_lr1e-4"

# ── Exp 3: Flatten decoder (all timesteps → MLP, no mean-pool) ──
submit_job "sfm-recon-flat" "40G" \
  "model.recon_decoder_type=flat \
   common.wandb_run_name=recon_flat_lr1e-4 \
   hydra.run.dir=/storage/noy/SpectralFM/fairseq/outputs/recon_loss/recon_flat_lr1e-4"

# ── Exp 4: Deeper MLP decoder (keep mean-pool, wider hidden layers) ──
submit_job "sfm-recon-deep-mlp" "40G" \
  "model.recon_decoder_type=mlp \
   model.recon_decoder_hidden=1024,512 \
   common.wandb_run_name=recon_deep_mlp_lr1e-4 \
   hydra.run.dir=/storage/noy/SpectralFM/fairseq/outputs/recon_loss/recon_deep_mlp_lr1e-4"

# ── Exp 5: Default MLP + lr=1e-5 (slower FE adaptation) ──
# Brackets must be escaped to survive RunAI's bash -c chain
submit_job "sfm-recon-lr1e5" "40G" \
  "model.recon_decoder_type=mlp \
   optimization.lr=\\[0.00001\\] \
   common.wandb_run_name=recon_mlp_lr1e-5 \
   hydra.run.dir=/storage/noy/SpectralFM/fairseq/outputs/recon_loss/recon_mlp_lr1e-5"

# ── Exp 6: Default MLP + L2 loss ──
submit_job "sfm-recon-l2" "40G" \
  "model.recon_decoder_type=mlp \
   model.recon_loss_type=l2 \
   common.wandb_run_name=recon_mlp_l2_lr1e-4 \
   hydra.run.dir=/storage/noy/SpectralFM/fairseq/outputs/recon_loss/recon_mlp_l2_lr1e-4"

echo "============================================================"
echo " All jobs submitted. Track on WandB: spectralfm_recon_loss"
echo " Monitor: watch -n 30 'runai list jobs | grep sfm-recon'"
echo "============================================================"
