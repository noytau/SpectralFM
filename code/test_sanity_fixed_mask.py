"""
Sanity test script for fixed mask training and evaluation.

This script validates the loss calculation by:
1. Training on a small 100-sample subset with reproducible masking
2. Saving mask metadata (seed, sample IDs, mask configuration)
3. Evaluating on the EXACT same samples with the EXACT same masks
4. Computing Data2Vec loss (student vs EMA teacher) to compare train vs eval loss

Expected result: Training and evaluation loss should be very similar (within 5-10%)
if the loss calculation is correct. NOT zero - see plan for explanation.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf, DictConfig, open_dict
from typing import Optional, Dict, List, Any

# Add fairseq to path
_FAIRSEQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fairseq")
if _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)

from fairseq import tasks
from fairseq.trainer import Trainer
from model_loader import load_fairseq_checkpoint
from test_utils import (
    create_subset_dataset,
    load_subset_dataset,
    create_fixed_mask_indices,
    generate_mask_hash,
    save_mask_metadata,
    load_mask_metadata,
    load_saved_masks,
    get_mask_for_batch,
    compute_reproducible_mask_indices,
    compute_data2vec_loss,
)


def create_training_config(
    subset_data_path: str,
    output_dir: str,
    mask_start: int = 10,
    mask_end: int = 25,
    max_update: int = 100,
    batch_size: int = 8
) -> DictConfig:
    """
    Create a training configuration for the sanity test.
    
    Args:
        subset_data_path: Path to subset dataset directory
        output_dir: Directory to save checkpoints
        mask_start: Start index for fixed mask
        mask_end: End index for fixed mask
        max_update: Maximum number of updates (small for quick test)
        batch_size: Batch size
    
    Returns:
        OmegaConf DictConfig for training
    """
    # Base config structure
    cfg = OmegaConf.create({
        "common": {
            "fp16": False,  # Disable fp16 for sanity test
            "log_format": "json",
            "log_interval": 10,
            "tensorboard_logdir": os.path.join(output_dir, "tb"),
            "user_dir": os.path.join(os.path.dirname(os.path.dirname(__file__)), "fairseq", "examples", "data2vec"),
            "seed": 42,
            "tpu": False,  # Required for task config interpolation
            "empty_cache_freq": 0,  # Disable empty cache for sanity test
        },
        "checkpoint": {
            "save_dir": os.path.join(output_dir, "checkpoints"),
            "save_interval": 1,
            "save_interval_updates": 50,
            "keep_interval_updates": 1,
            "no_epoch_checkpoints": True,
        },
        "task": {
            "_name": "audio_pretraining",
            "data": subset_data_path,
            "max_sample_size": 500,
            "min_sample_size": 200,
            "normalize": True,
            # tpu and seed will be interpolated from common.tpu and common.seed
        },
        "dataset": {
            "num_workers": 2,
            "batch_size": batch_size,
            "skip_invalid_size_inputs_valid_test": True,
            "required_batch_size_multiple": 1,
        },
        "distributed_training": {
            "distributed_world_size": 1,  # Single GPU for sanity test
            "ddp_backend": "no_c10d",
        },
        "criterion": {
            "_name": "model",
            "log_keys": ["ema_decay", "target_var", "pred_var"],
        },
        "optimization": {
            "max_update": max_update,
            "max_epoch": 1,
            "lr": [0.0005],
            "update_freq": [1],
            "clip_norm": 0.0,  # Disable gradient clipping for sanity test
        },
        "optimizer": {
            "_name": "adam",
            "adam_betas": "(0.9,0.98)",
            "adam_eps": 1e-06,
            "weight_decay": 0.01,
        },
        "lr_scheduler": {
            "_name": "fixed",
        },
        "ema": {
            "store_ema": False,  # Disable EMA for sanity test (simpler)
            "ema_decay": 0.9999,
            "ema_start_update": 0,
            "ema_update_freq": 1,
            "ema_fp32": False,
        },
        "model": {
            "_name": "data2vec_audio",
            "extractor_mode": "layer_norm",
            "encoder_layerdrop": 0.05,
            "dropout_input": 0.0,
            "dropout_features": 0.0,
            "feature_grad_mult": 1.0,
            "encoder_embed_dim": 768,
            "conv_feature_layers": "[(512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 5, 5)]",
            "mask_prob": 0.65,  # Still set mask_prob > 0, but we'll override with fixed mask
            "mask_length": 10,
            "loss_beta": 0,
            "loss_scale": None,
            "instance_norm_target_layer": True,
            "average_top_k_layers": 8,
            "pos_conv_depth": 5,
            "conv_pos": 95,
            "ema_decay": 0.999,
            "ema_end_decay": 0.9999,
            "ema_anneal_end_step": 30000,
            "ema_transformer_only": True,
            "ema_layers_only": True,
            "require_same_masks": True,
            "mask_dropout": 0,
        },
    })
    
    # Create config with struct mode disabled to allow flexibility
    # (some fields might not exist in all model configs)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(cfg.model, False)
    
    # Note: base_layers is not in Data2VecAudioConfig dataclass, so we can't set it here.
    # The trainer uses getattr(self.cfg.model, "base_layers", 0) which will return 0 if missing.
    # However, if the attribute exists but is None, we need to handle it after model is built.
    
    # Resolve interpolations (like ${common.tpu}, ${common.seed})
    # This ensures that task.tpu and task.seed are properly resolved
    try:
        OmegaConf.resolve(cfg)
    except Exception as e:
        print(f"[!] Warning: Could not resolve all config interpolations: {e}")
        print("[!] Continuing anyway...")
    
    return cfg


def set_fixed_mask_on_model(model, mask_start: int, mask_end: int):
    """
    Set fixed mask indices on the model.
    
    Args:
        model: The Data2VecAudioModel instance
        mask_start: Start index for fixed mask
        mask_end: End index for fixed mask
    """
    model._fixed_mask_start = mask_start
    model._fixed_mask_end = mask_end
    print(f"[+] Set fixed mask on model: indexes {mask_start}-{mask_end}")


def train_with_fixed_mask(
    subset_data_path: str,
    output_dir: str,
    mask_start: int = 10,
    mask_end: int = 25,
    max_update: int = 100,
    batch_size: int = 8
):
    """
    Train model with fixed mask indices.
    
    Args:
        subset_data_path: Path to subset dataset
        output_dir: Directory to save checkpoints
        mask_start: Start index for fixed mask
        mask_end: End index for fixed mask
        max_update: Maximum number of updates
        batch_size: Batch size
    """
    print(f"[+] Starting training with fixed mask (indexes {mask_start}-{mask_end})")
    
    # Create config
    cfg = create_training_config(
        subset_data_path, output_dir, mask_start, mask_end, max_update, batch_size
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint.save_dir, exist_ok=True)
    
    # Create a minimal training loop that sets the fixed mask on the model
    print("[+] Setting up training with fixed mask...")
    
    # Setup task and model
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    
    # Fix base_layers issue: ensure it's 0 (not None) to avoid trainer errors
    # Do this after model is built so we can modify the config
    if hasattr(cfg.model, "base_layers") and getattr(cfg.model, "base_layers", None) is None:
        with open_dict(cfg.model):
            cfg.model.base_layers = 0
    
    # Set fixed mask
    set_fixed_mask_on_model(model, mask_start, mask_end)
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create trainer
    trainer = Trainer(cfg, task, model, criterion)
    
    # Load dataset
    task.load_dataset("train")
    train_dataset = task.dataset("train")
    
    # Create batch iterator
    batch_itr = task.get_batch_iterator(
        dataset=train_dataset,
        max_tokens=None,  # Use batch_size instead
        max_sentences=cfg.dataset.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=1,
    )
    
    # Get epoch iterator
    epoch_itr = batch_itr.next_epoch_itr(shuffle=True)
    
    # Training loop
    print(f"[+] Training for {max_update} updates...")
    print(f"[+] Dataset size: {len(train_dataset)}")
    print(f"[+] Batch iterator type: {type(batch_itr)}")
    print(f"[+] Epoch iterator type: {type(epoch_itr)}")
    
    num_updates = 0
    for batch_idx, sample in enumerate(epoch_itr):
        if num_updates >= max_update:
            break
        
        # Debug: check sample format on first batch
        if batch_idx == 0:
            print(f"[+] Debug: First batch type: {type(sample)}")
            if isinstance(sample, dict):
                print(f"[+] Debug: Batch keys: {list(sample.keys())}")
                if "net_input" in sample:
                    net_input = sample['net_input']
                    print(f"[+] Debug: net_input type: {type(net_input)}")
                    print(f"[+] Debug: net_input repr (first 500 chars): {repr(net_input)[:500]}")
                    if isinstance(net_input, dict):
                        print(f"[+] Debug: net_input keys: {list(net_input.keys())}")
                        if "source" in net_input:
                            print(f"[+] Debug: source type: {type(net_input['source'])}")
                            print(f"[+] Debug: source shape: {net_input['source'].shape if hasattr(net_input['source'], 'shape') else 'N/A'}")
                    elif isinstance(net_input, str):
                        print(f"[!] ERROR: net_input is a string! This is the problem.")
                        print(f"[!] Full net_input value: {net_input}")
                        print(f"[!] This suggests the dataset collater is not working correctly.")
                        # Don't continue if net_input is wrong
                        raise ValueError(f"net_input must be a dict, got string: {net_input[:200]}")
            elif isinstance(sample, str):
                print(f"[!] ERROR: Sample is a string! Value: {sample[:200]}")
            elif isinstance(sample, (list, tuple)):
                print(f"[+] Debug: Sample is list/tuple with {len(sample)} items")
                if len(sample) > 0:
                    print(f"[+] Debug: First item type: {type(sample[0])}")
                    print(f"[+] Debug: First item value: {sample[0]}")
        
        # Ensure sample is a dict (batch iterator should return collated batches)
        if not isinstance(sample, dict):
            print(f"[!] Error: sample is not a dict, got {type(sample)}")
            print(f"[!] Sample value: {sample}")
            if isinstance(sample, str):
                print(f"[!] Sample is a string! This suggests a dataset/collater issue.")
                print(f"[!] Full sample: {sample}")
            elif isinstance(sample, (list, tuple)):
                print(f"[!] Sample is a list/tuple with {len(sample)} items")
                if len(sample) > 0:
                    print(f"[!] First item type: {type(sample[0])}")
                    print(f"[!] First item: {sample[0]}")
            raise TypeError(f"Expected dict, got {type(sample)}: {sample}")
        
        # Forward and backward
        try:
            # Additional check: verify net_input is a dict before passing to trainer
            if isinstance(sample, dict) and "net_input" in sample:
                net_input = sample["net_input"]
                if not isinstance(net_input, dict):
                    print(f"[!] ERROR: net_input is not a dict! Type: {type(net_input)}")
                    print(f"[!] net_input value: {net_input}")
                    print(f"[!] This suggests the collater is not working correctly.")
                    # Try to fix it - maybe it's a string representation?
                    if isinstance(net_input, str):
                        print(f"[!] net_input is a string. This is wrong.")
                        raise ValueError(f"net_input should be a dict, got string: {net_input[:200]}")
                    raise TypeError(f"net_input should be a dict, got {type(net_input)}")
                # Verify net_input has the expected structure
                if "source" not in net_input:
                    print(f"[!] WARNING: net_input missing 'source' key. Keys: {list(net_input.keys())}")
            
            # Final verification before passing to trainer
            if not isinstance(sample, dict):
                raise TypeError(f"Sample must be a dict, got {type(sample)}")
            if "net_input" not in sample:
                raise KeyError("Sample missing 'net_input' key")
            net_input = sample["net_input"]
            if not isinstance(net_input, dict):
                raise TypeError(f"net_input must be a dict, got {type(net_input)}: {net_input}")
            
            # Move all tensors in sample to the correct device
            # The trainer should handle this, but let's ensure it's done
            sample_on_device = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample_on_device[k] = v.to(device)
                elif isinstance(v, dict):
                    sample_on_device[k] = {}
                    for k2, v2 in v.items():
                        if isinstance(v2, torch.Tensor):
                            sample_on_device[k][k2] = v2.to(device)
                        else:
                            sample_on_device[k][k2] = v2
                else:
                    sample_on_device[k] = v
            
            # Trainer expects a list of samples, not a single sample
            logging_output = trainer.train_step([sample_on_device])
            num_updates += 1
            
            if num_updates % 10 == 0:
                # Extract loss from logging_output
                loss = logging_output.get("loss", 0.0)
                if torch.is_tensor(loss):
                    loss = loss.item()
                print(f"  Update {num_updates}/{max_update}, Loss: {loss:.4f}")
        except Exception as e:
            print(f"[!] Error in train_step: {e}")
            print(f"[!] Sample type: {type(sample)}")
            if isinstance(sample, dict):
                print(f"[!] Sample keys: {list(sample.keys())}")
                if "net_input" in sample:
                    net_input = sample["net_input"]
                    print(f"[!] net_input type: {type(net_input)}")
                    print(f"[!] net_input value (first 500 chars): {str(net_input)[:500]}")
            import traceback
            traceback.print_exc()
            raise
    
    # Save checkpoint
    checkpoint_path = os.path.join(cfg.checkpoint.save_dir, "checkpoint_last.pt")
    trainer.save_checkpoint(checkpoint_path, {"num_updates": num_updates})
    print(f"[+] Saved checkpoint to: {checkpoint_path}")
    
    return checkpoint_path


def evaluate_with_fixed_mask(
    checkpoint_path: str,
    subset_data_path: str,
    output_dir: str,
    mask_start: int = 10,
    mask_end: int = 25
):
    """
    Evaluate model with fixed mask indices (legacy - embedding comparison).
    
    Args:
        checkpoint_path: Path to checkpoint file
        subset_data_path: Path to subset dataset
        output_dir: Directory to save evaluation results
        mask_start: Start index for fixed mask
        mask_end: End index for fixed mask
    """
    print(f"[+] Starting evaluation with fixed mask (indexes {mask_start}-{mask_end})")
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg, checkpoint_info = load_fairseq_checkpoint(checkpoint_path, device)
    
    # Set fixed mask on loaded model
    set_fixed_mask_on_model(model, mask_start, mask_end)
    
    # Load dataset
    task_cfg = OmegaConf.create({
        "_name": "audio_pretraining",
        "data": subset_data_path,
        "max_sample_size": 500,
        "min_sample_size": 200,
        "normalize": True,
    })
    
    task = tasks.setup_task(task_cfg)
    task.load_dataset("train")
    train_dataset = task.dataset("train")
    
    # Get samples
    samples = []
    for i in range(min(100, len(train_dataset))):
        samples.append(train_dataset[i])
    
    print(f"[+] Evaluating on {len(samples)} samples...")
    
    # Evaluation
    model.eval()
    results = []
    
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            source = sample["source"]
            sample_id = sample.get("id", idx)
            
            # Prepare input
            data = source.to(device)
            if data.dim() == 1:
                data = data.unsqueeze(0)  # [1, seq_len]
            
            # Get ground truth embeddings (unmasked)
            gt_result = model.extract_features(data, padding_mask=None, mask=False)
            gt_embeddings = gt_result["x"]  # [1, T, embed_dim]
            
            T = gt_embeddings.shape[1]  # Sequence length after feature extraction
            
            # Create fixed mask for this sequence
            fixed_mask = create_fixed_mask_indices(1, T, mask_start, mask_end)
            mask_indices = torch.from_numpy(fixed_mask).to(device)
            
            # Get masked embeddings
            masked_result = model.forward(
                data,
                padding_mask=None,
                mask=True,
                features_only=True,
                mask_indices=fixed_mask
            )
            masked_embeddings = masked_result["x"]  # [1, T, embed_dim]
            
            # Compute metrics at masked positions
            if mask_indices.any():
                gt_masked = gt_embeddings[mask_indices]  # [num_masked, embed_dim]
                pred_masked = masked_embeddings[mask_indices]  # [num_masked, embed_dim]
                
                # Cosine similarity
                cos_sim = F.cosine_similarity(
                    gt_masked, pred_masked, dim=-1
                ).mean().item()
                
                # MSE
                mse = F.mse_loss(pred_masked, gt_masked).item()
                
                # L1
                l1 = F.l1_loss(pred_masked, gt_masked).item()
                
                results.append({
                    "sample_id": int(sample_id),
                    "seq_len": T,
                    "num_masked": mask_indices.sum().item(),
                    "cos_sim": cos_sim,
                    "mse": mse,
                    "l1": l1,
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "eval_results.csv")
    results_df.to_csv(results_path, index=False)
    
    print(f"[+] Evaluation complete!")
    print(f"    - Results saved to: {results_path}")
    if len(results) > 0:
        print(f"    - Mean cosine similarity: {results_df['cos_sim'].mean():.4f}")
        print(f"    - Mean MSE: {results_df['mse'].mean():.4f}")
        print(f"    - Mean L1: {results_df['l1'].mean():.4f}")
    
    return results_path


def evaluate_data2vec_loss(
    checkpoint_path: str,
    subset_data_path: str,
    output_dir: str,
    seed: int = 42,
    mask_prob: float = 0.4,
    mask_length: int = 10,
    batch_size: int = 16,
    mask_metadata_path: Optional[str] = None,
    saved_mask_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate model using the actual Data2Vec loss computation.
    
    This computes the same loss as training: MSE between student predictions
    and EMA teacher targets at masked positions.
    
    Args:
        checkpoint_path: Path to checkpoint file
        subset_data_path: Path to subset dataset
        output_dir: Directory to save evaluation results
        seed: Random seed for reproducible masking
        mask_prob: Mask probability (should match training)
        mask_length: Mask span length (should match training)
        batch_size: Batch size for evaluation
        mask_metadata_path: Optional path to saved mask metadata for exact reproduction
        saved_mask_dir: Optional path to directory containing saved mask files from training
    
    Returns:
        Dictionary with evaluation results including loss values
    """
    print(f"\n{'='*60}")
    print(f"DATA2VEC LOSS VALIDATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {subset_data_path}")
    print(f"Seed: {seed}, Mask prob: {mask_prob}, Mask length: {mask_length}")
    if saved_mask_dir:
        print(f"Using saved masks from: {saved_mask_dir}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Loading checkpoint on device: {device}")
    model, model_cfg, checkpoint_info = load_fairseq_checkpoint(checkpoint_path, device)
    
    print(f"[+] Checkpoint info:")
    print(f"    - Epoch: {checkpoint_info.get('epoch', 'N/A')}")
    print(f"    - Updates: {checkpoint_info.get('num_updates', 'N/A')}")
    
    # Load dataset
    task_cfg = OmegaConf.create({
        "_name": "audio_pretraining",
        "data": subset_data_path,
        "max_sample_size": 500,
        "min_sample_size": 200,
        "normalize": True,
    })
    
    task = tasks.setup_task(task_cfg)
    task.load_dataset("train")
    train_dataset = task.dataset("train")
    
    num_samples = len(train_dataset)
    print(f"[+] Loaded dataset with {num_samples} samples")
    
    # Load mask metadata if provided
    if mask_metadata_path and os.path.exists(mask_metadata_path):
        print(f"[+] Loading mask metadata from: {mask_metadata_path}")
        mask_metadata = load_mask_metadata(mask_metadata_path)
        seed = mask_metadata.get("seed", seed)
        mask_prob = mask_metadata.get("mask_prob", mask_prob)
        mask_length = mask_metadata.get("mask_length", mask_length)
    
    # Load saved masks if provided
    saved_masks = None
    if saved_mask_dir and os.path.exists(saved_mask_dir):
        saved_masks = load_saved_masks(saved_mask_dir)
        print(f"[+] Will use {len(saved_masks)} saved masks from training")
    
    # Collect samples and compute loss
    model.eval()
    
    all_losses = []
    all_sample_sizes = []
    batch_results = []
    
    # Process in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"[+] Evaluating {num_samples} samples in {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_sample_ids = list(range(start_idx, end_idx))
        
        # Get batch samples
        batch_sources = []
        for sid in batch_sample_ids:
            sample = train_dataset[sid]
            batch_sources.append(sample["source"])
        
        # Pad and stack
        max_len = max(s.shape[0] for s in batch_sources)
        padded_sources = []
        for s in batch_sources:
            if s.shape[0] < max_len:
                pad = torch.zeros(max_len - s.shape[0])
                s = torch.cat([s, pad])
            padded_sources.append(s)
        
        sources = torch.stack(padded_sources).to(device)
        current_batch_size = sources.shape[0]
        
        # Get sequence length after feature extraction
        with torch.no_grad():
            features = model.feature_extractor(sources)
            seq_length = features.shape[2]  # After conv: (B, C, T)
        
        # Get masks - either from saved training masks or compute reproducibly
        mask_indices = get_mask_for_batch(
            saved_masks=saved_masks,
            batch_idx=batch_idx,
            batch_size=current_batch_size,
            sequence_length=seq_length,
            mask_prob=mask_prob,
            mask_length=mask_length,
            seed=seed,
            sample_indices=batch_sample_ids
        )
        
        # Compute Data2Vec loss
        try:
            loss_result = compute_data2vec_loss(
                model=model,
                source=sources,
                mask_indices=mask_indices,
                device=device
            )
            
            all_losses.append(loss_result["regression_loss"])
            all_sample_sizes.append(loss_result["sample_size"])
            
            batch_results.append({
                "batch_idx": batch_idx,
                "sample_ids": batch_sample_ids,
                "regression_loss": loss_result["regression_loss"],
                "normalized_loss": loss_result["normalized_loss"],
                "sample_size": loss_result["sample_size"],
                "target_var": loss_result["target_var"],
                "pred_var": loss_result["pred_var"],
                "num_masked": mask_indices.sum(),
            })
            
            if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
                print(f"  Batch {batch_idx+1}/{num_batches}: "
                      f"loss={loss_result['normalized_loss']:.6f}, "
                      f"target_var={loss_result['target_var']:.4f}, "
                      f"pred_var={loss_result['pred_var']:.4f}")
                
        except Exception as e:
            print(f"[!] Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute aggregate statistics
    total_loss = sum(all_losses)
    total_sample_size = sum(all_sample_sizes)
    avg_loss = total_loss / total_sample_size if total_sample_size > 0 else 0.0
    
    results = {
        "checkpoint_path": checkpoint_path,
        "data_path": subset_data_path,
        "num_samples": num_samples,
        "num_batches": len(batch_results),
        "seed": seed,
        "mask_prob": mask_prob,
        "mask_length": mask_length,
        "total_loss": total_loss,
        "total_sample_size": total_sample_size,
        "average_loss": avg_loss,
        "batch_results": batch_results,
        "evaluation_time": datetime.now().isoformat(),
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "data2vec_loss_validation.json")
    
    # Convert batch_results for JSON serialization
    serializable_results = results.copy()
    serializable_results["batch_results"] = [
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
         for k, v in br.items()}
        for br in batch_results
    ]
    
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    # Also save as CSV for easy analysis
    batch_df = pd.DataFrame(batch_results)
    batch_csv_path = os.path.join(output_dir, "data2vec_loss_by_batch.csv")
    batch_df.to_csv(batch_csv_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples evaluated: {num_samples}")
    print(f"Total sample size (masked positions): {total_sample_size}")
    print(f"Total loss: {total_loss:.6f}")
    print(f"Average loss (per masked position): {avg_loss:.6f}")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Batch results saved to: {batch_csv_path}")
    print(f"{'='*60}\n")
    
    return results


def compare_train_eval_loss(
    training_log_path: str,
    eval_results_path: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Compare training loss with evaluation loss to validate loss calculation.
    
    Args:
        training_log_path: Path to training log file (hydra_train.log or tensorboard)
        eval_results_path: Path to evaluation results JSON
        output_dir: Directory to save comparison results
    
    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*60}")
    print(f"COMPARING TRAINING VS EVALUATION LOSS")
    print(f"{'='*60}")
    
    # Load evaluation results
    with open(eval_results_path, "r") as f:
        eval_results = json.load(f)
    
    eval_loss = eval_results["average_loss"]
    
    # Try to parse training loss from log
    train_loss = None
    if os.path.exists(training_log_path):
        try:
            with open(training_log_path, "r") as f:
                lines = f.readlines()
            
            # Look for loss values in JSON log format
            for line in reversed(lines):
                if '"loss"' in line or "'loss'" in line:
                    try:
                        log_entry = json.loads(line.strip())
                        if "loss" in log_entry:
                            train_loss = float(log_entry["loss"])
                            break
                    except:
                        pass
        except Exception as e:
            print(f"[!] Could not parse training log: {e}")
    
    comparison = {
        "eval_loss": eval_loss,
        "train_loss": train_loss,
        "difference": None,
        "percentage_diff": None,
        "validation_passed": None,
    }
    
    if train_loss is not None:
        comparison["difference"] = abs(eval_loss - train_loss)
        comparison["percentage_diff"] = (comparison["difference"] / train_loss) * 100 if train_loss > 0 else None
        
        # Validation passes if within 10%
        if comparison["percentage_diff"] is not None:
            comparison["validation_passed"] = comparison["percentage_diff"] < 10.0
    
    # Save comparison
    comparison_path = os.path.join(output_dir, "loss_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    # Print results
    print(f"Evaluation loss: {eval_loss:.6f}")
    print(f"Training loss: {train_loss if train_loss else 'N/A'}")
    if comparison["difference"] is not None:
        print(f"Difference: {comparison['difference']:.6f} ({comparison['percentage_diff']:.2f}%)")
        print(f"Validation: {'PASSED' if comparison['validation_passed'] else 'FAILED'}")
    print(f"{'='*60}\n")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Sanity test with fixed mask for validating Data2Vec loss calculation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full validation (train + eval) on 100 samples
  python test_sanity_fixed_mask.py --mode validate \\
      --original_data /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_100
  
  # Evaluate existing checkpoint with Data2Vec loss
  python test_sanity_fixed_mask.py --mode eval_loss \\
      --checkpoint /path/to/checkpoint.pt \\
      --subset_data /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_100
  
  # Evaluate using EXACT same masks saved during training
  python test_sanity_fixed_mask.py --mode validate \\
      --checkpoint /path/to/checkpoint.pt \\
      --subset_data /path/to/data \\
      --saved_mask_dir /path/to/saved_masks
  
  # Legacy mode: train with fixed mask positions (indexes 10-25)
  python test_sanity_fixed_mask.py --mode train \\
      --original_data /path/to/data --mask_start 10 --mask_end 25
"""
    )
    parser.add_argument("--mode", type=str, 
                       choices=["train", "eval", "both", "eval_loss", "validate"],
                       default="validate",
                       help="Mode: train, eval (legacy), both, eval_loss (Data2Vec loss), validate (full)")
    parser.add_argument("--original_data", type=str, default=None,
                       help="Path to original dataset directory (must contain train.tsv)")
    parser.add_argument("--subset_size", type=int, default=100,
                       help="Number of samples in subset (default: 100)")
    parser.add_argument("--output_dir", type=str, default="./outputs/sanity_test_fixed_mask",
                       help="Output directory for checkpoints and results")
    parser.add_argument("--mask_start", type=int, default=10,
                       help="Start index for fixed mask (default: 10) - legacy mode only")
    parser.add_argument("--mask_end", type=int, default=25,
                       help="End index for fixed mask (default: 25) - legacy mode only")
    parser.add_argument("--mask_prob", type=float, default=0.4,
                       help="Mask probability (default: 0.4)")
    parser.add_argument("--mask_length", type=int, default=10,
                       help="Mask span length (default: 10)")
    parser.add_argument("--max_update", type=int, default=100,
                       help="Maximum number of training updates (default: 100)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint for evaluation")
    parser.add_argument("--subset_data", type=str, default=None,
                       help="Path to subset dataset (if already exists)")
    parser.add_argument("--training_log", type=str, default=None,
                       help="Path to training log for loss comparison")
    parser.add_argument("--saved_mask_dir", type=str, default=None,
                       help="Path to directory containing saved mask files from training")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine subset data path
    if args.subset_data is not None:
        subset_data_path = args.subset_data
        print(f"[+] Using provided subset dataset: {subset_data_path}")
    elif args.original_data is not None:
        print("[+] Creating subset dataset...")
        subset_data_path, selected_indices = create_subset_dataset(
            args.original_data,
            subset_size=args.subset_size,
            seed=args.seed,
            output_dir=os.path.join(args.output_dir, "subset_data")
        )
        # Save selected indices for reference
        indices_path = os.path.join(args.output_dir, "selected_indices.json")
        with open(indices_path, "w") as f:
            json.dump(selected_indices, f)
        print(f"[+] Saved selected indices to: {indices_path}")
    else:
        # Default to the 100-sample dataset
        subset_data_path = "/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_100"
        if not os.path.exists(subset_data_path):
            raise ValueError("No dataset specified. Use --original_data or --subset_data")
        print(f"[+] Using default subset dataset: {subset_data_path}")
    
    # Handle different modes
    if args.mode == "validate":
        # Full validation: evaluate with Data2Vec loss on existing or trained checkpoint
        print("\n" + "="*60)
        print("RUNNING FULL LOSS VALIDATION")
        print("="*60 + "\n")
        
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            # Look for checkpoint in output directory
            possible_paths = [
                os.path.join(args.output_dir, "checkpoints", "checkpoint_last.pt"),
                os.path.join(args.output_dir, "checkpoints", "checkpoint_best.pt"),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    checkpoint_path = p
                    break
            
            if checkpoint_path is None:
                print("[!] No checkpoint found. Please train first or provide --checkpoint")
                print("[!] You can train using: python fairseq_cli/hydra_train.py "
                      "--config-dir examples/data2vec/config/audio/pretraining "
                      "--config-name sanity_test_fixed_mask")
                return
        
        # Run Data2Vec loss evaluation
        eval_results = evaluate_data2vec_loss(
            checkpoint_path=checkpoint_path,
            subset_data_path=subset_data_path,
            output_dir=args.output_dir,
            seed=args.seed,
            mask_prob=args.mask_prob,
            mask_length=args.mask_length,
            batch_size=args.batch_size,
            saved_mask_dir=args.saved_mask_dir,
        )
        
        # Try to compare with training loss if log is available
        if args.training_log:
            compare_train_eval_loss(
                training_log_path=args.training_log,
                eval_results_path=os.path.join(args.output_dir, "data2vec_loss_validation.json"),
                output_dir=args.output_dir
            )
    
    elif args.mode == "eval_loss":
        # Just evaluate Data2Vec loss on checkpoint
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required for eval_loss mode")
        
        evaluate_data2vec_loss(
            checkpoint_path=checkpoint_path,
            subset_data_path=subset_data_path,
            output_dir=args.output_dir,
            seed=args.seed,
            mask_prob=args.mask_prob,
            mask_length=args.mask_length,
            batch_size=args.batch_size,
            saved_mask_dir=args.saved_mask_dir,
        )
    
    elif args.mode in ["train", "both"]:
        # Legacy: train with fixed mask positions
        checkpoint_path = train_with_fixed_mask(
            subset_data_path,
            args.output_dir,
            mask_start=args.mask_start,
            mask_end=args.mask_end,
            max_update=args.max_update,
            batch_size=args.batch_size
        )
        
        if args.mode == "both":
            evaluate_with_fixed_mask(
                checkpoint_path,
                subset_data_path,
                args.output_dir,
                mask_start=args.mask_start,
                mask_end=args.mask_end
            )
    
    elif args.mode == "eval":
        # Legacy: evaluate with fixed mask positions
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        evaluate_with_fixed_mask(
            checkpoint_path,
            subset_data_path,
            args.output_dir,
            mask_start=args.mask_start,
            mask_end=args.mask_end
        )


if __name__ == "__main__":
    main()
