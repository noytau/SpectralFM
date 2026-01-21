"""
Utility functions for sanity testing with fixed masks.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset


def create_subset_dataset(
    original_data_path: str,
    subset_size: int = 100,
    seed: int = 42,
    output_dir: Optional[str] = None
) -> Tuple[str, list]:
    """
    Create a subset of the dataset by sampling a fixed number of samples.
    
    Args:
        original_data_path: Path to the original dataset directory (contains train.tsv)
        subset_size: Number of samples to include in subset (default: 100)
        seed: Random seed for reproducibility
        output_dir: Directory to save subset manifest. If None, uses a temp dir.
    
    Returns:
        Tuple of (subset_data_path, selected_indices)
        - subset_data_path: Path to directory containing the subset manifest
        - selected_indices: List of original indices that were selected
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Check if the data path exists
    if not os.path.exists(original_data_path):
        raise FileNotFoundError(
            f"Dataset directory not found: {original_data_path}\n"
            f"Please provide a valid path to the dataset directory containing train.tsv"
        )
    
    if not os.path.isdir(original_data_path):
        raise ValueError(f"Path is not a directory: {original_data_path}")
    
    # Read original manifest
    original_manifest = os.path.join(original_data_path, "train.tsv")
    if not os.path.exists(original_manifest):
        # Check for alternative manifest files
        alternative_names = ["train.tsv", "train.txt", "manifest.tsv"]
        found_files = [f for f in alternative_names if os.path.exists(os.path.join(original_data_path, f))]
        
        error_msg = (
            f"Manifest file not found: {original_manifest}\n"
            f"Directory contents: {os.listdir(original_data_path) if os.path.exists(original_data_path) else 'N/A'}\n"
        )
        if found_files:
            error_msg += f"Found alternative files: {found_files}\n"
        error_msg += (
            f"Please ensure the dataset directory contains a 'train.tsv' file.\n"
            f"Expected format: first line is root directory, subsequent lines are sample paths."
        )
        raise FileNotFoundError(error_msg)
    
    # Read all lines from manifest
    with open(original_manifest, "r") as f:
        lines = f.readlines()
    
    if len(lines) < 2:  # Need at least header + 1 sample
        raise ValueError(f"Manifest has too few samples: {len(lines) - 1}")
    
    # First line is root directory
    root_dir = lines[0].strip()
    
    # Get all sample lines (skip header)
    sample_lines = lines[1:]
    
    if len(sample_lines) < subset_size:
        raise ValueError(
            f"Requested subset size ({subset_size}) is larger than available samples ({len(sample_lines)})"
        )
    
    # Randomly select indices
    selected_indices = sorted(random.sample(range(len(sample_lines)), subset_size))
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(original_data_path, f"subset_{subset_size}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Write subset manifest
    subset_manifest = os.path.join(output_dir, "train.tsv")
    with open(subset_manifest, "w") as f:
        # Write root directory (use relative path or absolute)
        f.write(f"{root_dir}\n")
        # Write selected samples
        for idx in selected_indices:
            f.write(sample_lines[idx])
    
    print(f"[+] Created subset dataset:")
    print(f"    - Original samples: {len(sample_lines)}")
    print(f"    - Subset size: {subset_size}")
    print(f"    - Subset manifest: {subset_manifest}")
    print(f"    - Selected indices: {selected_indices[:10]}... (showing first 10)")
    
    return output_dir, selected_indices


def create_fixed_mask_indices(
    batch_size: int,
    sequence_length: int,
    mask_start: int = 10,
    mask_end: int = 25
) -> np.ndarray:
    """
    Create a fixed mask that masks only indexes from mask_start to mask_end (inclusive).
    
    Args:
        batch_size: Batch size
        sequence_length: Sequence length at feature level (after feature extraction)
        mask_start: Start index to mask (inclusive, 0-indexed)
        mask_end: End index to mask (inclusive, 0-indexed)
    
    Returns:
        numpy array of shape (batch_size, sequence_length) with boolean mask
        True indicates masked positions
    """
    # Handle edge cases
    if sequence_length <= mask_start:
        # Sequence is too short, return all False
        return np.zeros((batch_size, sequence_length), dtype=bool)
    
    # Adjust mask_end if sequence is shorter
    actual_mask_end = min(mask_end, sequence_length - 1)
    
    if actual_mask_end < mask_start:
        # No valid range to mask
        return np.zeros((batch_size, sequence_length), dtype=bool)
    
    # Create mask: True for positions [mask_start, actual_mask_end] (inclusive)
    mask = np.zeros((batch_size, sequence_length), dtype=bool)
    mask[:, mask_start:actual_mask_end + 1] = True
    
    return mask


def load_subset_dataset(
    subset_data_path: str,
    sample_rate: int = 16000,
    max_sample_size: Optional[int] = None,
    min_sample_size: Optional[int] = None,
    normalize: bool = False
) -> FileAudioDataset:
    """
    Load the subset dataset using FileAudioDataset.
    
    Args:
        subset_data_path: Path to subset directory (contains train.tsv)
        sample_rate: Target sample rate
        max_sample_size: Maximum sample size
        min_sample_size: Minimum sample size
        normalize: Whether to normalize audio
    
    Returns:
        FileAudioDataset instance
    """
    manifest_path = os.path.join(subset_data_path, "train.tsv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Subset manifest not found: {manifest_path}")
    
    dataset = FileAudioDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        max_sample_size=max_sample_size,
        min_sample_size=min_sample_size,
        normalize=normalize,
        shuffle=False,  # Don't shuffle for reproducibility
    )
    
    print(f"[+] Loaded subset dataset: {len(dataset)} samples")
    return dataset
