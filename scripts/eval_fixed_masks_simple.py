#!/usr/bin/env python3
"""
Simple evaluation script that loads checkpoint and mask memory, then runs validation.
"""

import os
import sys
import torch

# Setup paths
sys.path.insert(0, '/storage/noy/SpectralFM/fairseq')
os.environ['USER_DIR'] = '/storage/noy/SpectralFM/fairseq/examples/data2vec'

from fairseq import checkpoint_utils, tasks
from fairseq_cli import validate
from omegaconf import DictConfig, OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    checkpoint_path = '/storage/noy/SpectralFM/fairseq/outputs/2026-02-15/20-51-24/checkpoints/checkpoint_best.pt'
    mask_memory_path = '/storage/noy/SpectralFM/fairseq/outputs/mask_memorize/mask_memory.pt'
    data_dir = '/storage/noy/SpectralFM/fairseq/data/nova_data/base_libri_100'
    
    print("=" * 80)
    print("EVALUATION WITH FIXED MASKS")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mask Memory: {mask_memory_path}")
    print(f"Dataset: {data_dir}")
    print("=" * 80)
    
    # Create a minimal config for validation
    cfg = OmegaConf.create({
        'common_eval': {'path': checkpoint_path},
        'task': {'data': data_dir, '_name': 'audio_pretraining'},
        'dataset': {
            'valid_subset': 'valid_subset_100',
            'batch_size': 8,
            'max_tokens': None,
        },
        'common': {
            'fp16': False,
            'cpu': False,
            'user_dir': '/storage/noy/SpectralFM/fairseq/examples/data2vec',
        },
        'distributed_training': {
            'distributed_world_size': 1,
            'device_id': 0,
        },
        'checkpoint': {'checkpoint_suffix': ''},
        'criterion': {'_name': 'model'},
    })
    
    # Import only what we need (avoid importing all tasks which has errors)
    # Directly import the model and task modules
    sys.path.insert(0, '/storage/noy/SpectralFM/fairseq/examples/data2vec')
    # Import model directly (bypasses __init__.py)
    from examples.data2vec.models import data2vec_audio  # Register model
    # Import task directly (bypasses __init__.py)  
    import importlib.util
    task_spec = importlib.util.spec_from_file_location(
        "audio_pretraining",
        "/storage/noy/SpectralFM/fairseq/examples/data2vec/tasks/audio_pretraining.py"
    )
    audio_pretraining = importlib.util.module_from_spec(task_spec)
    task_spec.loader.exec_module(audio_pretraining)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    overrides = {'task': {'data': data_dir}}
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides=overrides,
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
                logger.warning("[!] Failed to load mask memory")
        else:
            logger.warning(f"[!] Mask memory file not found: {mask_memory_path}")
    else:
        logger.warning("[!] Model does not support mask memory")
    
    # Update config for validation
    saved_cfg.dataset.valid_subset = 'valid_subset_100'
    saved_cfg.dataset.batch_size = 8
    saved_cfg.common.fp16 = False
    
    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()
    
    # Run validation
    logger.info("Starting validation...")
    validate.validate(cfg, model, task, criterion, saved_cfg)


if __name__ == "__main__":
    main()
