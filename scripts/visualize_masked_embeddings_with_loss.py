#!/usr/bin/env python3
"""
Visualize masked locations on embeddings with input samples and evaluation loss.
For each sample shows:
- Input audio/spectrogram
- Embeddings with masked locations overlaid
- Evaluation loss
- Mask indices (loaded vs evaluated)
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec

# Setup paths
sys.path.insert(0, '/mnt5/noy/SpectralFM/fairseq')
sys.path.insert(0, '/mnt5/noy/SpectralFM/code')
os.environ['USER_DIR'] = '/mnt5/noy/SpectralFM/fairseq/examples/data2vec'

from fairseq import checkpoint_utils, tasks, utils
from omegaconf import open_dict
from fairseq.trainer import Trainer
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


def visualize_samples_with_masks_and_loss():
    """Visualize input, embeddings, masks, and loss for each sample."""
    
    # Paths
    checkpoint_path = '/mnt5/noy/SpectralFM/checkpoints/checkpoint_best.pt'
    mask_memory_path = '/mnt5/noy/SpectralFM/checkpoints/mask_memory.pt'
    data_dir = '/mnt5/noy/SpectralFM/fairseq/data/nova_data/base_libri_100'
    subset_manifest = f'{data_dir}/valid_subset_100.tsv'
    output_dir = '/mnt5/noy/SpectralFM/code/eval_results/masked_embeddings_viz'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("VISUALIZING MASKED EMBEDDINGS WITH INPUT AND LOSS")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Mask Memory: {mask_memory_path}")
    print(f"Dataset: {data_dir}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Load checkpoint to get best loss
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    best_loss = checkpoint.get('extra_state', {}).get('best', None)
    if best_loss is None:
        best_loss = checkpoint.get('best', None)
    print(f"\n[+] Best Loss from Checkpoint: {best_loss if best_loss is not None else 'N/A'}")
    
    # Load mask memory
    logger.info(f"Loading mask memory from {mask_memory_path}")
    mask_memory = load_mask_memory(mask_memory_path)
    print(f"[+] Loaded {len(mask_memory)} masks from mask memory")
    
    # Import model
    import marshal
    pyc_path = '/mnt5/noy/SpectralFM/fairseq/examples/data2vec/models/__pycache__/data2vec_audio.cpython-310.pyc'
    if os.path.exists(pyc_path):
        with open(pyc_path, 'rb') as f:
            f.read(16)
            code = marshal.load(f)
            module = type(sys)('examples.data2vec.models.data2vec_audio')
            sys.modules['examples.data2vec.models.data2vec_audio'] = module
            exec(code, module.__dict__)
    
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
    
    # Move model to GPU
    use_cuda = torch.cuda.is_available() and not getattr(saved_cfg.common, 'cpu', False)
    trainer_device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.to(trainer_device)
        # Ensure EMA model is also on GPU
        if hasattr(model, 'ema') and model.ema is not None and hasattr(model.ema, 'model') and model.ema.model is not None:
            try:
                model.ema.model = model.ema.model.to(trainer_device)
            except Exception:
                pass
    
    # Update config
    with open_dict(saved_cfg):
        saved_cfg.task.data = data_dir
        saved_cfg.dataset.valid_subset = 'valid_subset_100'
        saved_cfg.dataset.batch_size = 1  # Process one at a time
        saved_cfg.common.fp16 = False
    
    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()
    if use_cuda:
        criterion = criterion.to(trainer_device)
    
    # Create trainer
    trainer = Trainer(saved_cfg, task, model, criterion)
    
    # Load dataset
    task.load_dataset('valid_subset_100', combine=False, epoch=1)
    dataset = task.dataset('valid_subset_100')
    
    # Get iterator
    itr = trainer.get_valid_iterator('valid_subset_100').next_epoch_itr(shuffle=False)
    
    # Process first 10 samples for visualization
    num_samples_to_viz = min(10, len(dataset))
    
    print(f"\n[+] Processing {num_samples_to_viz} samples for visualization...")
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(itr):
            if batch_idx >= num_samples_to_viz:
                break
            
            # Get sample IDs
            sample_ids = sample.get('id', torch.tensor([batch_idx]))
            if isinstance(sample_ids, torch.Tensor):
                sample_ids = sample_ids.cpu().tolist()
            if not isinstance(sample_ids, list):
                sample_ids = [sample_ids]
            
            sample_id = sample_ids[0]
            
            # Forward pass to get loss and embeddings
            log_output = trainer.valid_step(sample)
            if log_output:
                loss = log_output.get('loss', 0)
                sample_size = log_output.get('sample_size', 0)
            else:
                loss = 0
                sample_size = 0
            
            # Get input audio/spectrogram
            source = sample.get('net_input', {}).get('source')
            if source is None:
                logger.warning(f"Sample {sample_id}: No source data found")
                continue
            
            if isinstance(source, torch.Tensor):
                input_data = source[0].cpu().numpy()  # (seq_len, features) or (seq_len,)
                if input_data.ndim == 1:
                    # Raw audio waveform
                    input_data = input_data.reshape(-1, 1)
            else:
                input_data = np.array(source[0]) if hasattr(source, '__len__') else None
            
            # Get embeddings by extracting features
            # We need to get embeddings before masking is applied
            # Actually, let's get the features after feature extraction but before transformer
            try:
                # Extract features to get embeddings
                padding_mask = sample.get('net_input', {}).get('padding_mask')
                if padding_mask is not None and isinstance(padding_mask, torch.Tensor):
                    padding_mask = padding_mask[0:1]  # Keep batch dimension
                    if use_cuda:
                        padding_mask = padding_mask.cuda()
                
                # Ensure source is on correct device
                source_tensor = source[0:1] if isinstance(source, torch.Tensor) else source
                if isinstance(source_tensor, torch.Tensor) and use_cuda:
                    source_tensor = source_tensor.cuda()
                
                # Get embeddings from model
                features = model.extract_features(
                    source_tensor,
                    padding_mask=padding_mask
                )
                
                if isinstance(features, dict):
                    embeddings = features.get('x', features.get('encoder_out'))
                elif isinstance(features, tuple):
                    embeddings = features[0]
                else:
                    embeddings = features
                
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings[0].cpu().numpy()  # (seq_len, embed_dim)
                else:
                    embeddings = np.array(embeddings[0]) if hasattr(embeddings, '__len__') else None
            except Exception as e:
                logger.warning(f"Sample {sample_id}: Could not extract embeddings: {e}")
                embeddings = None
            
            # Get mask for this sample
            loaded_mask = None
            if sample_id in mask_memory:
                loaded_mask = mask_memory[sample_id]
            elif str(sample_id) in mask_memory:
                loaded_mask = mask_memory[str(sample_id)]
            elif batch_idx in mask_memory:
                loaded_mask = mask_memory[batch_idx]
            
            if loaded_mask is not None:
                if isinstance(loaded_mask, torch.Tensor):
                    loaded_mask = loaded_mask.cpu().numpy()
                elif not isinstance(loaded_mask, np.ndarray):
                    loaded_mask = np.array(loaded_mask, dtype=bool)
            
            # Get sequence length
            if embeddings is not None:
                seq_len = embeddings.shape[0]
            elif input_data is not None:
                seq_len = input_data.shape[0]
            else:
                seq_len = None
            
            # Check if mask can be used (sequence length match)
            mask_used = False
            evaluated_mask = None
            if loaded_mask is not None and seq_len is not None:
                if len(loaded_mask) == seq_len:
                    mask_used = True
                    evaluated_mask = loaded_mask
                else:
                    # Sequence length mismatch - random mask was generated
                    mask_used = False
            
            # Create visualization
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Plot 1: Input audio/spectrogram
            ax1 = fig.add_subplot(gs[0, :])
            if input_data is not None:
                if input_data.ndim == 2 and input_data.shape[1] > 1:
                    # Spectrogram-like data
                    im = ax1.imshow(input_data.T, aspect='auto', origin='lower', cmap='viridis')
                    plt.colorbar(im, ax=ax1, label='Amplitude')
                    ax1.set_ylabel('Frequency Bin', fontsize=11)
                else:
                    # Waveform
                    if input_data.ndim == 2:
                        input_data = input_data.flatten()
                    ax1.plot(input_data, linewidth=0.5)
                    ax1.set_ylabel('Amplitude', fontsize=11)
                ax1.set_xlabel('Time Step', fontsize=11)
                ax1.set_title(f'Sample {sample_id}: Input Audio/Spectrogram', fontsize=12, fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'Input data not available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f'Sample {sample_id}: Input (N/A)', fontsize=12)
            
            # Plot 2: Embeddings heatmap with mask overlay
            ax2 = fig.add_subplot(gs[1, :])
            if embeddings is not None:
                im = ax2.imshow(embeddings.T, aspect='auto', cmap='viridis', interpolation='nearest')
                plt.colorbar(im, ax=ax2, label='Embedding Value')
                
                # Overlay mask as red vertical lines
                if evaluated_mask is not None:
                    masked_positions = np.where(evaluated_mask)[0]
                    for pos in masked_positions:
                        ax2.axvline(x=pos, color='red', linewidth=2, alpha=0.7)
                elif loaded_mask is not None:
                    # Show where mask would be if sequence length matched
                    loaded_positions = np.where(loaded_mask)[0]
                    for pos in loaded_positions:
                        if pos < embeddings.shape[0]:
                            ax2.axvline(x=pos, color='orange', linewidth=1.5, alpha=0.5, linestyle='--', label='Loaded mask (if length matched)')
                
                ax2.set_xlabel('Sequence Position', fontsize=11)
                ax2.set_ylabel('Embedding Dimension', fontsize=11)
                mask_status = "USED" if mask_used else "RANDOM (length mismatch)"
                ax2.set_title(f'Sample {sample_id}: Embeddings with Mask Overlay\n'
                             f'Sequence Length: {seq_len}, Mask: {mask_status}, '
                             f'Loss: {loss:.6f}',
                             fontsize=12, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'Embeddings not available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f'Sample {sample_id}: Embeddings (N/A)', fontsize=12)
            
            # Plot 3: Mask pattern (loaded)
            ax3 = fig.add_subplot(gs[2, 0])
            if loaded_mask is not None:
                loaded_indices = np.where(loaded_mask)[0]
                ax3.bar(range(len(loaded_mask)), loaded_mask.astype(int), 
                       color='blue', alpha=0.7, width=1.0)
                ax3.set_xlabel('Position', fontsize=10)
                ax3.set_ylabel('Masked', fontsize=10)
                ax3.set_title(f'Loaded Mask (seq_len={len(loaded_mask)})', fontsize=11, fontweight='bold')
                ax3.set_ylim(-0.1, 1.1)
                ax3.set_yticks([0, 1])
                ax3.set_yticklabels(['Unmasked', 'Masked'])
                ax3.grid(True, alpha=0.3, axis='y')
            else:
                ax3.text(0.5, 0.5, 'No loaded mask', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Loaded Mask (N/A)', fontsize=11)
            
            # Plot 4: Evaluation info and loss
            ax4 = fig.add_subplot(gs[2, 1])
            ax4.axis('off')
            info_text = f"Sample ID: {sample_id}\n"
            info_text += f"Evaluation Loss: {loss:.6f}\n"
            info_text += f"Sample Size: {sample_size}\n"
            if best_loss is not None:
                info_text += f"Best Loss (checkpoint): {best_loss:.6f}\n\n"
            else:
                info_text += f"Best Loss (checkpoint): N/A\n\n"
            
            if loaded_mask is not None:
                loaded_indices = np.where(loaded_mask)[0].tolist()
                info_text += f"Loaded Mask:\n"
                info_text += f"  Indices: {loaded_indices[:20]}{'...' if len(loaded_indices) > 20 else ''}\n"
                info_text += f"  Count: {len(loaded_indices)}/{len(loaded_mask)}\n"
                info_text += f"  Seq Length: {len(loaded_mask)}\n\n"
            
            if embeddings is not None:
                info_text += f"Embeddings:\n"
                info_text += f"  Shape: {embeddings.shape}\n"
                info_text += f"  Seq Length: {seq_len}\n\n"
            
            info_text += f"Mask Status:\n"
            if mask_used:
                info_text += f"  ✓ Mask from memory USED\n"
                info_text += f"  Masks Match: True\n"
            else:
                info_text += f"  ✗ Random mask generated\n"
                info_text += f"  Reason: Sequence length mismatch\n"
                if loaded_mask is not None and seq_len is not None:
                    info_text += f"  (stored={len(loaded_mask)}, current={seq_len})\n"
            
            ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle(f'Sample {sample_id} Visualization: Input, Embeddings, Masks, and Loss', 
                        fontsize=14, fontweight='bold', y=0.98)
            
            # Save figure
            output_path = os.path.join(output_dir, f'sample_{sample_id}_complete_viz.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Saved visualization for sample {sample_id} to {output_path}")
            logger.info(f"    Loss: {loss:.6f}, Mask used: {mask_used}")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Visualized {num_samples_to_viz} samples")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    visualize_samples_with_masks_and_loss()
