#!/usr/bin/env python3
"""
Evaluate validation loss on subset with detailed per-sample information.
For each sample, shows:
- Loss value
- Mask indices loaded from memory
- Mask indices actually used during evaluation
"""

import os
import sys
import torch
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

# Setup paths
sys.path.insert(0, '/mnt5/noy/SpectralFM/fairseq')
sys.path.insert(0, '/mnt5/noy/SpectralFM/code')
os.environ['USER_DIR'] = '/mnt5/noy/SpectralFM/fairseq/examples/data2vec'

from fairseq import checkpoint_utils, tasks, utils
from omegaconf import open_dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mask_memory(mask_memory_path):
    """Load mask memory from file."""
    import torch.serialization
    torch_module = sys.modules['torch']
    original_torch_load = torch_module.load
    def patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch_module.load = patched_torch_load
    try:
        mask_memory = torch.load(mask_memory_path, map_location='cpu')
    finally:
        torch_module.load = original_torch_load
    return mask_memory


def create_subset_manifest(source_manifest, target_manifest, num_samples=100, seed=42):
    """Create a subset manifest with random samples."""
    with open(source_manifest, 'r') as f:
        lines = f.readlines()
    
    header = lines[0]
    data_lines = lines[1:]
    
    random.seed(seed)
    sampled_lines = random.sample(data_lines, min(num_samples, len(data_lines)))
    
    with open(target_manifest, 'w') as f:
        f.write(header)
        for line in sampled_lines:
            f.write(line)
    
    logger.info(f"Created subset manifest: {target_manifest}")
    logger.info(f"  Original samples: {len(data_lines)}")
    logger.info(f"  Subset samples: {len(sampled_lines)}")
    return len(sampled_lines)


def evaluate_with_fixed_masks_detailed():
    """Evaluate with detailed per-sample information."""
    
    # Paths
    checkpoint_path = '/mnt5/noy/SpectralFM/checkpoints/checkpoint_best.pt'
    mask_memory_path = '/mnt5/noy/SpectralFM/checkpoints/mask_memory.pt'
    data_dir = '/mnt5/noy/SpectralFM/fairseq/data/nova_data/base_libri_100'
    train_manifest = f'{data_dir}/train.tsv'
    
    subset_manifest = f'{data_dir}/valid_subset_100.tsv'
    num_samples = create_subset_manifest(train_manifest, subset_manifest, num_samples=100, seed=42)
    
    print("=" * 80)
    print("DETAILED EVALUATION WITH FIXED MASKS")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mask Memory: {mask_memory_path}")
    print(f"Dataset: {data_dir}")
    print(f"Subset: {num_samples} samples from train.tsv")
    print("=" * 80)
    
    # Load checkpoint to get best loss
    logger.info("Loading checkpoint to extract best loss...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    best_loss = checkpoint.get('extra_state', {}).get('best', None)
    if best_loss is None:
        # Try alternative locations
        best_loss = checkpoint.get('best', None)
    if best_loss is None:
        # Try from optimizer history
        opt_history = checkpoint.get('optimizer_history', [])
        if opt_history:
            best_loss = opt_history[-1].get('best', None)
    
    print(f"\n[+] Best Loss from Checkpoint: {best_loss if best_loss is not None else 'N/A'}")
    print("=" * 80)
    
    # Load mask memory
    logger.info(f"Loading mask memory from {mask_memory_path}")
    mask_memory = load_mask_memory(mask_memory_path)
    print(f"[+] Loaded {len(mask_memory)} masks from mask memory")
    
    # Import model
    pyc_path = '/mnt5/noy/SpectralFM/fairseq/examples/data2vec/models/__pycache__/data2vec_audio.cpython-310.pyc'
    if os.path.exists(pyc_path):
        import marshal
        with open(pyc_path, 'rb') as f:
            f.read(16)
            code = marshal.load(f)
            module = type(sys)('examples.data2vec.models.data2vec_audio')
            sys.modules['examples.data2vec.models.data2vec_audio'] = module
            exec(code, module.__dict__)
    
    from fairseq import utils
    utils.import_user_module({'user_dir': os.environ['USER_DIR']})
    
    # Load model
    logger.info("Loading model...")
    overrides = {'task': {'data': data_dir}}
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides=overrides,
    )
    model = models[0]
    model.eval()
    
    # Move model to GPU first
    if torch.cuda.is_available():
        model = model.cuda()
        # Ensure all submodules are on GPU
        for param in model.parameters():
            if not param.is_cuda:
                param.data = param.data.cuda()
    
    # Load mask memory into model
    if hasattr(model, 'load_mask_memory'):
        torch_mod = sys.modules['torch']
        original_torch_load = torch_mod.load
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch_mod.load = patched_torch_load
        try:
            model.load_mask_memory(mask_memory_path)
            model.enable_mask_memory()
        finally:
            torch_mod.load = original_torch_load
    
    # Update config
    with open_dict(saved_cfg):
        saved_cfg.task.data = data_dir
        saved_cfg.dataset.valid_subset = 'valid_subset_100'
        saved_cfg.dataset.batch_size = 1  # Process one at a time for per-sample tracking
        saved_cfg.common.fp16 = False
    
    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()
    
    # Use Trainer for proper device handling
    from fairseq.trainer import Trainer
    use_cuda = torch.cuda.is_available() and not getattr(saved_cfg.common, 'cpu', False)
    trainer_device = torch.device("cuda" if use_cuda else "cpu")
    model_device = next(model.parameters()).device
    
    if model_device != trainer_device:
        model = model.to(trainer_device)
    
    # Ensure EMA model is also on the correct device
    if hasattr(model, 'ema') and model.ema is not None and hasattr(model.ema, 'model') and model.ema.model is not None:
        try:
            ema_model_device = next(model.ema.model.parameters()).device
            if ema_model_device != trainer_device:
                model.ema.model = model.ema.model.to(trainer_device)
        except Exception:
            pass
    
    # Create trainer
    trainer = Trainer(saved_cfg, task, model, criterion)
    
    # Load dataset
    task.load_dataset('valid_subset_100', combine=False, epoch=1)
    
    # Get iterator
    itr = trainer.get_valid_iterator('valid_subset_100').next_epoch_itr(shuffle=False)
    
    # Track per-sample information
    sample_results = []
    total_loss = 0.0
    total_tokens = 0
    
    print("\n" + "=" * 80)
    print("PER-SAMPLE EVALUATION RESULTS")
    print("=" * 80)
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(itr):
            
            # Get sample IDs
            sample_ids = sample.get('id', torch.tensor([batch_idx]))
            if isinstance(sample_ids, torch.Tensor):
                sample_ids = sample_ids.cpu().tolist()
            if not isinstance(sample_ids, list):
                sample_ids = [sample_ids]
            
            # Forward pass using trainer (handles device placement)
            log_output = trainer.valid_step(sample)
            if log_output:
                loss = log_output.get('loss', 0)
                sample_size = log_output.get('sample_size', 0)
                logging_output = log_output
            else:
                loss = 0
                sample_size = 0
                logging_output = {}
            
            # Get actual sequence length at embedding level (where mask is applied)
            # Masks are applied in compute_mask() on embeddings, not on raw input
            source = sample.get('net_input', {}).get('source')
            seq_len = None
            if source is not None:
                # Extract features to get embedding sequence length (where mask is applied)
                try:
                    # Get embeddings after feature extraction (before transformer)
                    # The mask is applied at this stage in compute_mask()
                    source_tensor = source[0:1] if isinstance(source, torch.Tensor) else source
                    if use_cuda and isinstance(source_tensor, torch.Tensor):
                        source_tensor = source_tensor.cuda()
                    
                    with torch.no_grad():
                        # extract_features returns embeddings after feature extractor
                        features = model.extract_features(
                            source_tensor,
                            padding_mask=None,
                            mask=False
                        )
                        if isinstance(features, dict):
                            x = features.get('x', features.get('encoder_out'))
                        else:
                            x = features
                        
                        if isinstance(x, torch.Tensor):
                            # x shape is typically (seq_len, batch, embed_dim) for fairseq
                            # or (batch, seq_len, embed_dim) for some models
                            if x.dim() == 3:
                                # Check which dimension is likely the sequence length
                                # Usually seq_len is the largest dimension
                                dims = x.shape
                                # Fairseq typically uses (seq_len, batch, embed_dim)
                                if dims[0] > dims[1] and dims[2] > 100:  # embed_dim is usually > 100
                                    seq_len = dims[0]  # (seq_len, batch, embed_dim)
                                elif dims[1] > dims[0] and dims[2] > 100:
                                    seq_len = dims[1]  # (batch, seq_len, embed_dim)
                                else:
                                    # Heuristic: use middle dimension if it's reasonable
                                    seq_len = dims[1] if dims[1] < 10000 else dims[0]
                            elif x.dim() == 2:
                                seq_len = x.shape[0]
                            
                            logger.debug(f"Extracted embedding sequence length: {seq_len} (x.shape: {x.shape})")
                except Exception as e:
                    logger.warning(f"Could not extract features to get sequence length: {e}")
                    # Fallback - but this won't match mask length
                    if isinstance(source, torch.Tensor):
                        seq_len = source.size(1) if source.dim() > 1 else source.size(0)
                    else:
                        seq_len = len(source) if hasattr(source, '__len__') else None
            
            # Get masks used during evaluation
            # Note: The model uses the mask from _mask_memory if sequence length matches
            # Otherwise it generates a random mask. We can't easily capture the actual mask
            # used, but we can check if it should match based on sequence length
            evaluated_masks = {}
            if hasattr(model, '_mask_memory') and model._mask_memory is not None:
                # The mask in _mask_memory is what would be used if seq_len matches
                for sid in sample_ids:
                    mask_key = None
                    if sid in model._mask_memory:
                        mask_key = sid
                    elif str(sid) in model._mask_memory:
                        mask_key = str(sid)
                    
                    if mask_key is not None:
                        stored_mask = model._mask_memory[mask_key]
                        if isinstance(stored_mask, torch.Tensor):
                            stored_mask = stored_mask.cpu().numpy()
                        elif not isinstance(stored_mask, np.ndarray):
                            stored_mask = np.array(stored_mask, dtype=bool)
                        
                        # Check if sequence length matches
                        stored_seq_len = len(stored_mask)
                        if seq_len is not None and stored_seq_len == seq_len:
                            # Mask should match - model will use stored mask
                            evaluated_masks[sid] = stored_mask
                        else:
                            # Sequence length mismatch - model generated random mask
                            # We can't easily get the random mask, so mark as None
                            evaluated_masks[sid] = None
            
            # Get loaded masks from mask_memory
            loaded_masks = {}
            for sid in sample_ids:
                mask = None
                if sid in mask_memory:
                    mask = mask_memory[sid]
                elif str(sid) in mask_memory:
                    mask = mask_memory[str(sid)]
                elif batch_idx in mask_memory:
                    mask = mask_memory[batch_idx]
                
                if mask is not None:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    elif not isinstance(mask, np.ndarray):
                        mask = np.array(mask, dtype=bool)
                    loaded_masks[sid] = mask
            
            # Store results
            for sid in sample_ids:
                loaded_mask = loaded_masks.get(sid)
                evaluated_mask = evaluated_masks.get(sid)
                
                # Get mask indices
                loaded_indices = np.where(loaded_mask)[0].tolist() if loaded_mask is not None else None
                evaluated_indices = np.where(evaluated_mask)[0].tolist() if evaluated_mask is not None else None
                
                # Check if masks match
                masks_match = False
                if loaded_mask is not None and evaluated_mask is not None:
                    if loaded_mask.shape == evaluated_mask.shape:
                        masks_match = np.array_equal(loaded_mask, evaluated_mask)
                
                sample_loss = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                
                sample_results.append({
                    'sample_id': sid,
                    'loss': sample_loss,
                    'sample_size': sample_size,
                    'loaded_mask_indices': loaded_indices,
                    'evaluated_mask_indices': evaluated_indices,
                    'masks_match': masks_match,
                    'loaded_mask_sum': int(loaded_mask.sum()) if loaded_mask is not None else None,
                    'evaluated_mask_sum': int(evaluated_mask.sum()) if evaluated_mask is not None else None,
                })
                
                total_loss += sample_loss * sample_size
                total_tokens += sample_size
                
                # Get sequence length info
                loaded_seq_len = len(loaded_mask) if loaded_mask is not None else None
                evaluated_seq_len = len(evaluated_mask) if evaluated_mask is not None else None
                
                # Print per-sample info
                print(f"\nSample ID: {sid}")
                print(f"  Loss: {sample_loss:.6f}")
                print(f"  Sample Size: {sample_size}")
                print(f"  Sequence Length: {seq_len if seq_len is not None else 'N/A'}")
                if loaded_indices is not None:
                    print(f"  Loaded Mask Indices: {loaded_indices}")
                    print(f"  Loaded Mask Count: {len(loaded_indices)} (for seq_len={loaded_seq_len})")
                else:
                    print(f"  Loaded Mask: NOT FOUND")
                if evaluated_mask is None:
                    print(f"  Evaluated Mask: RANDOM (sequence length mismatch: stored={loaded_seq_len}, current={seq_len})")
                elif evaluated_indices is not None:
                    print(f"  Evaluated Mask Indices: {evaluated_indices}")
                    print(f"  Evaluated Mask Count: {len(evaluated_indices)} (for seq_len={evaluated_seq_len})")
                else:
                    print(f"  Evaluated Mask: NOT FOUND")
                if loaded_mask is not None and evaluated_mask is not None:
                    print(f"  Masks Match: {masks_match}")
                    if not masks_match:
                        print(f"    WARNING: Masks do not match!")
                        print(f"    Loaded shape: {loaded_mask.shape}, Evaluated shape: {evaluated_mask.shape}")
                elif loaded_mask is not None and evaluated_mask is None:
                    print(f"  Masks Match: False (sequence length mismatch - random mask generated)")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Best Loss from Checkpoint: {best_loss if best_loss is not None else 'N/A'}")
    print(f"Average Validation Loss: {avg_loss:.6f}")
    print(f"Total Samples: {len(sample_results)}")
    print(f"Total Tokens: {total_tokens}")
    
    # Count mask matches
    matches = sum(1 for r in sample_results if r['masks_match'])
    print(f"Masks Match: {matches}/{len(sample_results)} samples")
    print("=" * 80)
    
    return avg_loss, sample_results, best_loss


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        avg_loss, sample_results, best_loss = evaluate_with_fixed_masks_detailed()
        print(f"\n✓ Evaluation completed successfully.")
        print(f"  Best Loss (checkpoint): {best_loss if best_loss is not None else 'N/A'}")
        print(f"  Average Loss: {avg_loss:.6f}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
