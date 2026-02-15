#!/bin/bash
# Training script for base_libri model on LibriSpeech test-clean dataset
# This trains a model compatible with our evaluation code

set -e

# Navigate to fairseq directory
cd /mnt5/noy/SpectralFM/fairseq

# Dataset path (already prepared with valid.tsv)
DATA_DIR="/mnt5/noy/SpectralFM/fairseq/data/librispeech_test_clean"

# Output directory for checkpoints
OUTPUT_DIR="/mnt5/noy/SpectralFM/fairseq/outputs/base_libri_test_clean_$(date +%Y%m%d_%H%M%S)"

# Config file
CONFIG_FILE="examples/data2vec/config/audio/pretraining/spectralfm_full_train.yaml"

echo "=========================================="
echo "Training base_libri on LibriSpeech test-clean"
echo "=========================================="
echo "Dataset: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Config: ${CONFIG_FILE}"
echo ""

# Check if dataset exists
if [ ! -f "${DATA_DIR}/valid.tsv" ]; then
    echo "ERROR: Dataset not found at ${DATA_DIR}/valid.tsv"
    echo "Please run: python /mnt5/noy/SpectralFM/code/prepare_librispeech_test_clean.py"
    exit 1
fi

# Create train.tsv from valid.tsv (for training, we'll use the same data)
# In a real scenario, you'd want separate train/valid splits
if [ ! -f "${DATA_DIR}/train.tsv" ]; then
    echo "[+] Creating train.tsv from valid.tsv (using same data for training)..."
    cp "${DATA_DIR}/valid.tsv" "${DATA_DIR}/train.tsv"
fi

# Training command
# Override config parameters for LibriSpeech test-clean:
# - task.data: point to our dataset
# - task.max_sample_size: adjust for LibriSpeech (test-clean has longer files)
# - task.min_sample_size: minimum sample size
# - distributed_training.distributed_world_size: adjust based on available GPUs
# - checkpoint.save_dir: output directory

python fairseq_cli/hydra_train.py \
    -m \
    --config-dir examples/data2vec/config/audio/pretraining \
    --config-name spectralfm_full_train \
    task.data="${DATA_DIR}" \
    task.max_sample_size=320000 \
    task.min_sample_size=32000 \
    task.normalize=true \
    distributed_training.distributed_world_size=1 \
    checkpoint.save_dir="${OUTPUT_DIR}" \
    common.wandb_project="spectralfm_base_libri_test_clean" \
    common.wandb_run_name="base_libri_test_clean_$(date +%Y%m%d_%H%M%S)" \
    optimization.max_update=10000 \
    optimization.max_epoch=10 \
    lr_scheduler.warmup_updates=1000 \
    dataset.batch_size=32 \
    dataset.num_workers=4 \
    model.conv_feature_layers='[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)]' \
    model.mask_prob=0.5 \
    model.mask_length=5

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo ""
echo "To evaluate, use:"
echo "  python /mnt5/noy/SpectralFM/code/evaluation_runner.py \\"
echo "    --checkpoint ${OUTPUT_DIR}/checkpoints/checkpoint_best.pt \\"
echo "    --eval_methods validation_loss \\"
echo "    --eval_data_dir ${DATA_DIR}"
echo "=========================================="
