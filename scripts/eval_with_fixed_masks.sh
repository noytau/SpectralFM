#!/bin/bash
# Evaluate validation loss on 100 random samples from train.tsv using fixed masks

set -e

# Paths (RunAI storage)
CHECKPOINT_PATH="/storage/noy/SpectralFM/fairseq/outputs/2026-02-15/20-51-24/checkpoints/checkpoint_best.pt"
MASK_MEMORY_PATH="/storage/noy/SpectralFM/fairseq/outputs/mask_memorize/mask_memory.pt"
DATA_DIR="/storage/noy/SpectralFM/fairseq/data/nova_data/base_libri_100"
TRAIN_MANIFEST="${DATA_DIR}/train.tsv"
SUBSET_MANIFEST="${DATA_DIR}/valid_subset_100.tsv"
FAIRSEQ_DIR="/storage/noy/SpectralFM/fairseq"

echo "=================================================================================="
echo "EVALUATION WITH FIXED MASKS"
echo "=================================================================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Mask Memory: ${MASK_MEMORY_PATH}"
echo "Dataset: ${DATA_DIR}"
echo "=================================================================================="

# Create subset manifest with 100 random samples
cd ${FAIRSEQ_DIR}
python3 << 'PYTHON_SCRIPT'
import random
import sys

train_manifest = '/storage/noy/SpectralFM/fairseq/data/nova_data/base_libri_100/train.tsv'
subset_manifest = '/storage/noy/SpectralFM/fairseq/data/nova_data/base_libri_100/valid_subset_100.tsv'

with open(train_manifest, 'r') as f:
    lines = f.readlines()

header = lines[0]
data_lines = lines[1:]

random.seed(42)
sampled_lines = random.sample(data_lines, min(100, len(data_lines)))

with open(subset_manifest, 'w') as f:
    f.write(header)
    for line in sampled_lines:
        f.write(line)

print(f"Created subset manifest: {subset_manifest}")
print(f"  Original samples: {len(data_lines)}")
print(f"  Subset samples: {len(sampled_lines)}")
PYTHON_SCRIPT

# Run evaluation using fairseq_cli.hydra_validate with fixed masks
cd ${FAIRSEQ_DIR}

python3 -m fairseq_cli.hydra_validate \
    --config-dir examples/data2vec/config/audio/pretraining \
    --config-name spectralfm_full_train \
    common_eval.path=${CHECKPOINT_PATH} \
    task.data=${DATA_DIR} \
    dataset.valid_subset=valid_subset_100 \
    dataset.batch_size=8 \
    common.fp16=false \
    common.user_dir=${FAIRSEQ_DIR}/examples/data2vec \
    +model.mask_memory_save_path=${MASK_MEMORY_PATH} \
    2>&1 | tee /storage/noy/SpectralFM/eval_fixed_masks.log

echo ""
echo "=================================================================================="
echo "Evaluation completed. Check /storage/noy/SpectralFM/eval_fixed_masks.log"
echo "=================================================================================="
