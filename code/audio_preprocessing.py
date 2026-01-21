"""
Audio preprocessing utilities for SpectralFM experiments.

This module provides functions for stretching audio samples to 16kHz length,
specifically for experiments that require 16kHz audio.
Note: Samples must come from load_eval_dataset_fairseq() and will be stretched
to exactly 16000 samples using linear interpolation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Any


def stretch_samples_to_16k(samples: List[Dict], 
                            target_length: int = 16000,
                            verbose: bool = True) -> List[Dict]:
    """
    Stretch audio samples to 16kHz length (exactly 16000 samples) using linear interpolation.
    
    Takes fairseq dataset samples (which must come from load_eval_dataset_fairseq())
    and stretches each sample's audio tensor from its original length to exactly
    16000 samples. Preserves all other sample metadata.
    
    This function uses F.interpolate for stretching, not resampling based on sample rates.
    It simply stretches the waveform to the target length.
    
    Args:
        samples: List of fairseq sample dictionaries from load_eval_dataset_fairseq()
                 Each sample must have 'source' key containing audio tensor
        target_length: Target length in samples (default: 16000)
        verbose: If True, print progress information
    
    Returns:
        List of samples with stretched audio in 'source' key, preserving all other metadata
    """
    if verbose:
        print(f"[+] Stretching {len(samples)} samples to {target_length} samples...")
    
    stretched_samples = []
    
    for idx, sample in enumerate(samples):
        try:
            # Create a copy to avoid modifying original
            new_sample = dict(sample)  # Use dict() to ensure a proper copy
            
            # Get source audio tensor
            if "source" not in sample:
                raise KeyError(f"Sample {idx} does not have 'source' key. Samples must come from load_eval_dataset_fairseq()")
            
            source_audio = sample["source"]
            
            # Convert to tensor if needed
            if not isinstance(source_audio, torch.Tensor):
                if isinstance(source_audio, (list, np.ndarray)):
                    source_audio = torch.tensor(source_audio, dtype=torch.float32)
                else:
                    raise TypeError(f"Sample {idx} source is not a tensor, list, or array")
            
            # Get original length
            original_length = source_audio.shape[-1] if source_audio.dim() > 0 else len(source_audio)
            
            # Skip stretching if already at target length
            if original_length == target_length:
                stretched_samples.append(new_sample)
                continue
            
            # Stretch using F.interpolate
            # F.interpolate expects [batch, channels, length] format for 1D interpolation
            # First ensure we have the right shape
            was_1d = source_audio.dim() == 1
            if was_1d:
                # Add batch and channel dimensions: [1, 1, L]
                tensor = source_audio.unsqueeze(0).unsqueeze(0)
            elif source_audio.dim() == 2:
                # Assume [channels, length], add batch dimension: [1, C, L]
                tensor = source_audio.unsqueeze(0)
            else:
                # Already has batch dimension or higher, just ensure it's [B, C, L]
                if source_audio.dim() == 3:
                    tensor = source_audio
                else:
                    # Flatten extra dimensions
                    tensor = source_audio.view(-1, source_audio.shape[-2], source_audio.shape[-1])
            
            # Stretch to target length using linear interpolation
            stretched_tensor = F.interpolate(
                tensor,
                size=target_length,
                mode='linear',
                align_corners=True
            )
            
            # Restore original shape
            if was_1d:
                stretched_audio = stretched_tensor.squeeze(0).squeeze(0)  # Back to [L]
            elif source_audio.dim() == 2:
                stretched_audio = stretched_tensor.squeeze(0)  # Back to [C, L]
            else:
                stretched_audio = stretched_tensor
            
            # Update sample with stretched audio
            new_sample["source"] = stretched_audio
            
            # Optionally store target length in metadata
            if "target_length" not in new_sample:
                new_sample["target_length"] = target_length
            
            stretched_samples.append(new_sample)
            
        except Exception as e:
            if verbose:
                print(f"[!] Error stretching sample {idx}: {e}")
                import traceback
                traceback.print_exc()
            # Keep original sample if stretching fails
            stretched_samples.append(sample)
    
    if verbose:
        print(f"[+] Successfully stretched {len(stretched_samples)} samples to {target_length} samples")
    
    return stretched_samples


def resample_samples_to_16k(samples: List[Dict], 
                             target_sr: int = 16000,
                             source_sr: Optional[int] = None,
                             verbose: bool = True) -> List[Dict]:
    """
    Stretch audio samples to 16kHz length (exactly 16000 samples) using linear interpolation.
    
    This is an alias for stretch_samples_to_16k() for backward compatibility.
    Note: This function now uses F.interpolate for stretching, not sample rate conversion.
    
    Takes fairseq dataset samples (which must come from load_eval_dataset_fairseq())
    and stretches each sample's audio tensor from its original length to exactly
    16000 samples. Preserves all other sample metadata.
    
    Args:
        samples: List of fairseq sample dictionaries from load_eval_dataset_fairseq()
                 Each sample must have 'source' key containing audio tensor
        target_sr: Target length in samples (default: 16000, kept for compatibility)
        source_sr: Ignored (kept for compatibility)
        verbose: If True, print progress information
    
    Returns:
        List of samples with stretched audio in 'source' key
    """
    # Map target_sr to target_length for backward compatibility
    target_length = target_sr
    return stretch_samples_to_16k(samples, target_length=target_length, verbose=verbose)


def resample_tensor_to_16k(audio_tensor: torch.Tensor,
                          source_sr: int,
                          target_sr: int = 16000) -> torch.Tensor:
    """
    Resample a single audio tensor to target sample rate.
    
    Args:
        audio_tensor: Audio tensor [samples] or [channels, samples]
        source_sr: Source sample rate
        target_sr: Target sample rate (default: 16000)
    
    Returns:
        Resampled audio tensor with same shape (except length dimension)
    """
    if source_sr == target_sr:
        return audio_tensor
    
    # Ensure 2D shape [channels, samples]
    was_1d = audio_tensor.dim() == 1
    if was_1d:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Resample
    resampled = torchaudio.functional.resample(
        audio_tensor,
        orig_freq=source_sr,
        new_freq=target_sr
    )
    
    # Restore original shape
    if was_1d:
        resampled = resampled.squeeze(0)
    
    return resampled
