#!/usr/bin/env python3
"""
Visualize masked locations on embeddings for the evaluation subset.
Shows which positions were masked during evaluation.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
sys.path.insert(0, '/mnt5/noy/SpectralFM/fairseq')
sys.path.insert(0, '/mnt5/noy/SpectralFM/code')
os.environ['USER_DIR'] = '/mnt5/noy/SpectralFM/fairseq/examples/data2vec'

from fairseq import checkpoint_utils
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


def visualize_masked_embeddings():
    """Visualize masked locations on embeddings."""
    
    # Paths
    checkpoint_path = '/mnt5/noy/SpectralFM/checkpoints/checkpoint_best.pt'
    mask_memory_path = '/mnt5/noy/SpectralFM/checkpoints/mask_memory.pt'
    data_dir = '/mnt5/noy/SpectralFM/fairseq/data/nova_data/base_libri_100'
    subset_manifest = f'{data_dir}/valid_subset_100.tsv'
    output_dir = '/mnt5/noy/SpectralFM/code/eval_results/masked_embeddings_viz'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("VISUALIZING MASKED EMBEDDINGS")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mask Memory: {mask_memory_path}")
    print(f"Dataset: {data_dir}")
    print(f"Subset: {subset_manifest}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Load mask memory
    logger.info(f"Loading mask memory from {mask_memory_path}")
    mask_memory = load_mask_memory(mask_memory_path)
    print(f"[+] Loaded {len(mask_memory)} masks from mask memory")
    
    # Load model
    logger.info("Loading model...")
    import marshal
    pyc_path = '/mnt5/noy/SpectralFM/fairseq/examples/data2vec/models/__pycache__/data2vec_audio.cpython-310.pyc'
    if os.path.exists(pyc_path):
        with open(pyc_path, 'rb') as f:
            f.read(16)
            code = marshal.load(f)
            module = type(sys)('examples.data2vec.models.data2vec_audio')
            sys.modules['examples.data2vec.models.data2vec_audio'] = module
            exec(code, module.__dict__)
    
    from fairseq import utils
    utils.import_user_module({'user_dir': os.environ['USER_DIR']})
    
    overrides = {'task': {'data': data_dir}}
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides=overrides,
    )
    model = models[0]
    model.eval()
    
    # Load mask memory into model first (before moving to GPU)
    if hasattr(model, 'load_mask_memory'):
        # Patch torch.load for mask memory loading
        import torch.serialization
        torch_mod = sys.modules['torch']  # Use different variable name
        original_torch_load = torch_mod.load
        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch_mod.load = patched_torch_load
        try:
            model.load_mask_memory(mask_memory_path)
            model.enable_mask_memory()
            logger.info(f"[+] Enabled mask memory in model")
        finally:
            torch_mod.load = original_torch_load
    
    # Move model to GPU after mask memory is loaded
    if torch.cuda.is_available():
        model.cuda()
    
    # Update config
    with open_dict(saved_cfg):
        saved_cfg.task.data = data_dir
        saved_cfg.dataset.valid_subset = 'valid_subset_100'
        saved_cfg.dataset.batch_size = 1  # Process one at a time for visualization
        saved_cfg.common.fp16 = False
    
    # Load dataset
    task.load_dataset('valid_subset_100', combine=False, epoch=1)
    dataset = task.dataset('valid_subset_100')
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Process first 10 samples for visualization
    num_samples_to_viz = min(10, len(dataset))
    
    for sample_idx in range(num_samples_to_viz):
        logger.info(f"Processing sample {sample_idx + 1}/{num_samples_to_viz}...")
        
        # Get sample
        sample = dataset[sample_idx]
        sample_batch = {
            'id': torch.tensor([sample['id']]),
            'net_input': {
                'source': sample['source'].unsqueeze(0),
                'padding_mask': sample['padding_mask'].unsqueeze(0) if 'padding_mask' in sample else None,
            },
            'target': sample.get('target'),
            'nsentences': 1,
        }
        
        if torch.cuda.is_available():
            sample_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                          for k, v in sample_batch.items()}
            if 'net_input' in sample_batch:
                sample_batch['net_input'] = {
                    k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in sample_batch['net_input'].items()
                }
        
        # Get mask for this sample
        # Try multiple ways to access the mask
        sample_id = int(sample['id'])
        mask = None
        
        # Try sample_id first
        if sample_id in mask_memory:
            mask = mask_memory[sample_id]
        # Try dataset index
        elif sample_idx in mask_memory:
            mask = mask_memory[sample_idx]
        # Try as string
        elif str(sample_id) in mask_memory:
            mask = mask_memory[str(sample_id)]
        # Try sequential access (if mask_memory is list-like)
        elif isinstance(mask_memory, (list, tuple)) and sample_idx < len(mask_memory):
            mask = mask_memory[sample_idx]
        
        if mask is None:
            logger.warning(f"Sample {sample_id} (idx {sample_idx}) not in mask memory, skipping")
            logger.warning(f"Mask memory keys (first 5): {list(mask_memory.keys())[:5] if isinstance(mask_memory, dict) else 'Not a dict'}")
            continue
        
        # Convert mask to numpy array
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        elif isinstance(mask, np.ndarray):
            mask = mask
        else:
            mask = np.array(mask, dtype=bool)
        
        # Forward pass to get embeddings
        with torch.no_grad():
            model.eval()
            # Get embeddings (before masking is applied)
            x = sample_batch['net_input']['source']
            padding_mask = sample_batch['net_input'].get('padding_mask')
            
            # Extract features
            features = model.extract_features(x, padding_mask=padding_mask)
            
            # Get the actual embeddings (from the encoder output)
            if isinstance(features, tuple):
                embeddings = features[0]  # (batch, seq_len, embed_dim)
            else:
                embeddings = features
            
            embeddings = embeddings[0].cpu().numpy()  # (seq_len, embed_dim)
        
        # Create visualization
        seq_len, embed_dim = embeddings.shape
        mask_len = len(mask)
        
        # Adjust mask length if needed
        if mask_len > seq_len:
            mask = mask[:seq_len]
        elif mask_len < seq_len:
            # Pad mask with False
            padded_mask = np.zeros(seq_len, dtype=bool)
            padded_mask[:mask_len] = mask
            mask = padded_mask
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Embedding heatmap with mask overlay
        ax1 = axes[0]
        im = ax1.imshow(embeddings.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_xlabel('Sequence Position', fontsize=12)
        ax1.set_ylabel('Embedding Dimension', fontsize=12)
        ax1.set_title(f'Sample {sample_id}: Embeddings with Mask Overlay\n'
                     f'Sequence Length: {seq_len}, Embedding Dim: {embed_dim}, '
                     f'Masked Positions: {mask.sum()}/{len(mask)} ({100*mask.sum()/len(mask):.1f}%)',
                     fontsize=13, fontweight='bold')
        
        # Overlay mask as red vertical lines
        masked_positions = np.where(mask)[0]
        for pos in masked_positions:
            ax1.axvline(x=pos, color='red', linewidth=2, alpha=0.7)
        
        plt.colorbar(im, ax=ax1, label='Embedding Value')
        
        # Plot 2: Mask pattern
        ax2 = axes[1]
        ax2.bar(range(len(mask)), mask.astype(int), color='red', alpha=0.7, width=1.0)
        ax2.set_xlabel('Sequence Position', fontsize=12)
        ax2.set_ylabel('Masked (1) / Unmasked (0)', fontsize=12)
        ax2.set_title(f'Mask Pattern for Sample {sample_id}', fontsize=13, fontweight='bold')
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Unmasked', 'Masked'])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'sample_{sample_id}_masked_embeddings.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved visualization to {output_path}")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Visualized {num_samples_to_viz} samples")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    visualize_masked_embeddings()
