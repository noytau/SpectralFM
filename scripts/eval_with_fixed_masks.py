#!/usr/bin/env python3
"""
Evaluate validation loss on 100 random samples from train.tsv using fixed masks.

Usage:
    python scripts/eval_with_fixed_masks.py
"""

import os
import sys
import torch
import random
import numpy as np
from pathlib import Path

# Setup paths (RunAI uses /storage instead of /mnt5)
sys.path.insert(0, '/storage/noy/SpectralFM/fairseq')
os.environ['USER_DIR'] = '/storage/noy/SpectralFM/fairseq/examples/data2vec'

from fairseq import checkpoint_utils, tasks, utils
from fairseq.data import data_utils
from fairseq.tasks import FairseqTask
from omegaconf import open_dict
import logging

# Import user module to register models
from omegaconf import DictConfig
user_dir_cfg = DictConfig({'user_dir': '/storage/noy/SpectralFM/fairseq/examples/data2vec'})
utils.import_user_module(user_dir_cfg)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_subset_manifest(source_manifest, target_manifest, num_samples=100, seed=42):
    """Create a subset manifest with random samples."""
    with open(source_manifest, 'r') as f:
        lines = f.readlines()
    
    # First line is header (root path)
    header = lines[0]
    data_lines = lines[1:]
    
    # Randomly sample
    random.seed(seed)
    sampled_lines = random.sample(data_lines, min(num_samples, len(data_lines)))
    
    # Write subset manifest
    with open(target_manifest, 'w') as f:
        f.write(header)  # Write root path
        for line in sampled_lines:
            f.write(line)
    
    logger.info(f"Created subset manifest: {target_manifest}")
    logger.info(f"  Original samples: {len(data_lines)}")
    logger.info(f"  Subset samples: {len(sampled_lines)}")
    return len(sampled_lines), sampled_lines


def evaluate_with_fixed_masks():
    """Evaluate validation loss using fixed masks."""
    
    # Paths (RunAI storage)
    checkpoint_path = '/storage/noy/SpectralFM/fairseq/outputs/2026-02-15/20-51-24/checkpoints/checkpoint_best.pt'
    mask_memory_path = '/storage/noy/SpectralFM/fairseq/outputs/mask_memorize/mask_memory.pt'
    data_dir = '/storage/noy/SpectralFM/fairseq/data/nova_data/base_libri_100'
    train_manifest = f'{data_dir}/train.tsv'
    
    # Create temporary subset manifest (as valid_subset_100.tsv for fairseq)
    subset_manifest = f'{data_dir}/valid_subset_100.tsv'
    num_samples, sampled_lines = create_subset_manifest(train_manifest, subset_manifest, num_samples=100, seed=42)
    
    print("=" * 80)
    print("EVALUATION WITH FIXED MASKS")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mask Memory: {mask_memory_path}")
    print(f"Dataset: {data_dir}")
    print(f"Subset: {num_samples} samples from train.tsv")
    print("=" * 80)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides=None,
    )
    model = models[0]
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
        logger.info("Model moved to GPU")
    
    # Load mask memory
    logger.info(f"Loading mask memory from {mask_memory_path}")
    if hasattr(model, 'load_mask_memory'):
        if os.path.exists(mask_memory_path):
            success = model.load_mask_memory(mask_memory_path)
            if success:
                model.enable_mask_memory()
                num_masks = len(model._mask_memory) if hasattr(model, '_mask_memory') else 0
                logger.info(f"[+] Loaded {num_masks} masks from mask memory")
            else:
                logger.warning("[!] Failed to load mask memory, will use seed-based masking")
        else:
            logger.warning(f"[!] Mask memory file not found: {mask_memory_path}")
    else:
        logger.warning("[!] Model does not support mask memory")
    
    # Update config to use subset as validation
    with open_dict(saved_cfg):
        saved_cfg.task.data = data_dir
        saved_cfg.dataset.valid_subset = 'valid_subset_100'  # Use our subset
        saved_cfg.common.fp16 = False  # Use fp32 for stable evaluation
        saved_cfg.dataset.batch_size = 8  # Smaller batch for evaluation
    
    # Rebuild task with updated config
    task = tasks.setup_task(saved_cfg.task)
    
    # Load the subset dataset
    logger.info("Loading dataset...")
    task.load_dataset('valid_subset_100', combine=False, epoch=1)
    dataset = task.dataset('valid_subset_100')
    
    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()
    
    # Evaluation loop
    logger.info("Starting evaluation...")
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    total_tokens = 0
    
    # Use task's validation iterator
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=saved_cfg.dataset.max_tokens if saved_cfg.dataset.max_tokens else None,
        max_sentences=saved_cfg.dataset.batch_size,
        max_positions=None,
        ignore_invalid_inputs=saved_cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=saved_cfg.dataset.required_batch_size_multiple,
        seed=saved_cfg.common.seed,
        num_workers=saved_cfg.dataset.num_workers,
        epoch=1,
        data_buffer_size=saved_cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    ).next_epoch_itr(shuffle=False)
    
    with torch.no_grad():
        for i, sample in enumerate(itr):
            if torch.cuda.is_available():
                sample = data_utils.move_to_cuda(sample)
            
            # Forward pass
            loss, sample_size, logging_output = criterion(model, sample)
            
            # Accumulate
            total_loss += loss.item() * sample_size
            total_samples += sample.get('nsentences', 0)
            total_tokens += sample_size
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1} batches, current loss: {loss.item():.6f}")
    
    # Calculate average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples evaluated: {total_samples}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average validation loss: {avg_loss:.6f}")
    print(f"Mask memory used: {hasattr(model, '_mask_memory') and model._mask_memory is not None}")
    print("=" * 80)
    
    # Cleanup (optional - comment out to keep manifest for inspection)
    # if os.path.exists(subset_manifest):
    #     logger.info(f"Cleaning up temporary manifest: {subset_manifest}")
    #     os.remove(subset_manifest)
    
    return avg_loss


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        loss = evaluate_with_fixed_masks()
        print(f"\n✓ Evaluation completed successfully. Loss: {loss:.6f}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
