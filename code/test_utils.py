"""
Utility functions for sanity testing with fixed masks.

This module provides utilities for:
- Creating reproducible masks using seeds
- Storing and loading mask metadata (hash, seed, indices)
- Computing Data2Vec-style losses for validation
"""

import os
import json
import hashlib
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from datetime import datetime
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


# ============================================================================
# Mask Hash and Metadata Functions
# ============================================================================

def generate_mask_hash(
    seed: int,
    sample_ids: List[int],
    mask_prob: float,
    mask_length: int,
    batch_size: int,
    sequence_length: int
) -> str:
    """
    Generate a deterministic hash for a mask configuration.
    
    This hash uniquely identifies a mask configuration so that the exact
    same masks can be reproduced during evaluation.
    
    Args:
        seed: Random seed used for mask generation
        sample_ids: List of sample IDs in the batch
        mask_prob: Mask probability
        mask_length: Mask span length
        batch_size: Batch size
        sequence_length: Sequence length after feature extraction
    
    Returns:
        A hex string hash representing this mask configuration
    """
    # Create a string representation of all parameters
    config_str = f"{seed}_{sorted(sample_ids)}_{mask_prob}_{mask_length}_{batch_size}_{sequence_length}"
    
    # Generate SHA256 hash
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]  # Use first 16 chars for brevity


def save_mask_metadata(
    output_path: str,
    seed: int,
    mask_prob: float,
    mask_length: int,
    sample_ids: List[int],
    mask_indices: Optional[np.ndarray] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save mask metadata to a JSON file for later reproduction.
    
    Args:
        output_path: Path to save the metadata JSON file
        seed: Random seed used for mask generation
        mask_prob: Mask probability
        mask_length: Mask span length
        sample_ids: List of sample IDs that were masked
        mask_indices: Optional numpy array of mask indices (B, T) - will be saved if provided
        additional_info: Optional dict with extra metadata
    
    Returns:
        Path to the saved metadata file
    """
    metadata = {
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "mask_prob": mask_prob,
        "mask_length": mask_length,
        "sample_ids": sample_ids,
        "num_samples": len(sample_ids),
    }
    
    # Generate hash for this configuration
    if mask_indices is not None:
        batch_size, seq_length = mask_indices.shape
        metadata["batch_size"] = batch_size
        metadata["sequence_length"] = seq_length
        metadata["mask_hash"] = generate_mask_hash(
            seed, sample_ids, mask_prob, mask_length, batch_size, seq_length
        )
        # Store mask indices as list for JSON serialization
        metadata["mask_indices"] = mask_indices.tolist()
    
    if additional_info:
        metadata.update(additional_info)
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[+] Saved mask metadata to: {output_path}")
    return str(output_path)


def load_mask_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load mask metadata from a JSON file.
    
    Args:
        metadata_path: Path to the metadata JSON file
    
    Returns:
        Dictionary containing mask metadata
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Convert mask_indices back to numpy array if present
    if "mask_indices" in metadata:
        metadata["mask_indices"] = np.array(metadata["mask_indices"], dtype=bool)
    
    print(f"[+] Loaded mask metadata from: {metadata_path}")
    print(f"    - Seed: {metadata.get('seed')}")
    print(f"    - Mask prob: {metadata.get('mask_prob')}")
    print(f"    - Num samples: {metadata.get('num_samples')}")
    
    return metadata


def load_saved_masks(mask_dir: str) -> Dict[int, np.ndarray]:
    """
    Load all saved mask files from a directory.
    
    Args:
        mask_dir: Directory containing mask_batch_XXXXXX.npy files
    
    Returns:
        Dictionary mapping batch index to mask numpy array
    """
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    
    masks = {}
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith("mask_batch_") and f.endswith(".npy")])
    
    for mask_file in mask_files:
        # Extract batch index from filename: mask_batch_000001.npy -> 1
        batch_idx = int(mask_file.replace("mask_batch_", "").replace(".npy", ""))
        mask_path = os.path.join(mask_dir, mask_file)
        masks[batch_idx] = np.load(mask_path)
    
    print(f"[+] Loaded {len(masks)} saved masks from: {mask_dir}")
    return masks


def get_mask_for_batch(
    saved_masks: Optional[Dict[int, np.ndarray]],
    batch_idx: int,
    batch_size: int,
    sequence_length: int,
    mask_prob: float,
    mask_length: int,
    seed: int,
    sample_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Get mask for a batch - either from saved masks or compute reproducibly.
    
    Args:
        saved_masks: Dictionary of saved masks (batch_idx -> mask array), or None
        batch_idx: Current batch index
        batch_size: Number of samples in batch
        sequence_length: Sequence length after feature extraction
        mask_prob: Mask probability (used if no saved mask)
        mask_length: Mask span length (used if no saved mask)
        seed: Random seed (used if no saved mask)
        sample_indices: Sample indices for additional seeding
    
    Returns:
        Boolean numpy array of shape (batch_size, sequence_length)
    """
    if saved_masks is not None and batch_idx in saved_masks:
        saved_mask = saved_masks[batch_idx]
        # Verify shape matches (or pad/truncate if needed)
        if saved_mask.shape == (batch_size, sequence_length):
            return saved_mask
        else:
            print(f"[!] Warning: Saved mask shape {saved_mask.shape} doesn't match "
                  f"expected ({batch_size}, {sequence_length}). Using saved mask anyway.")
            return saved_mask
    else:
        # Fall back to computing reproducible mask
        return compute_reproducible_mask_indices(
            batch_size=batch_size,
            sequence_length=sequence_length,
            mask_prob=mask_prob,
            mask_length=mask_length,
            seed=seed,
            sample_indices=sample_indices
        )


def compute_reproducible_mask_indices(
    batch_size: int,
    sequence_length: int,
    mask_prob: float,
    mask_length: int,
    seed: int,
    sample_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Compute mask indices in a reproducible way using the seed.
    
    This mimics the fairseq compute_mask_indices behavior but ensures
    reproducibility by properly seeding the random number generator.
    
    Args:
        batch_size: Number of samples in batch
        sequence_length: Sequence length (after feature extraction)
        mask_prob: Probability of masking each position
        mask_length: Length of each mask span
        seed: Random seed for reproducibility
        sample_indices: Optional list of sample indices for additional seeding
    
    Returns:
        Boolean numpy array of shape (batch_size, sequence_length)
        True indicates masked positions
    """
    # Create deterministic seed combining base seed with sample indices
    if sample_indices is not None:
        combined_seed = int(hash((seed, tuple(sample_indices))) % (2**31))
    else:
        combined_seed = seed
    
    # Set random state
    rng = np.random.RandomState(combined_seed)
    
    mask = np.full((batch_size, sequence_length), False)
    
    # Calculate number of mask spans
    num_mask_spans = max(1, int(mask_prob * sequence_length / mask_length))
    
    for batch_idx in range(batch_size):
        # Select random start positions for mask spans
        mask_starts = rng.choice(
            max(1, sequence_length - mask_length + 1),
            size=min(num_mask_spans, max(1, sequence_length - mask_length + 1)),
            replace=False
        )
        
        # Apply masks
        for start in mask_starts:
            end = min(start + mask_length, sequence_length)
            mask[batch_idx, start:end] = True
    
    return mask


def compute_data2vec_loss(
    model,
    source: torch.Tensor,
    mask_indices: torch.Tensor,
    device: torch.device
) -> Dict[str, float]:
    """
    Compute Data2Vec loss for a batch with specified mask indices.
    
    This replicates the loss computation from data2vec_audio.py forward()
    but allows using pre-specified mask indices.
    
    Args:
        model: The Data2VecAudioModel instance
        source: Input audio tensor of shape (B, seq_len)
        mask_indices: Boolean mask tensor of shape (B, T) at feature level
        device: Device to run computation on
    
    Returns:
        Dictionary with loss values and statistics
    """
    model.eval()
    
    with torch.no_grad():
        # Move inputs to device
        source = source.to(device)
        if not isinstance(mask_indices, torch.Tensor):
            mask_indices = torch.from_numpy(mask_indices)
        mask_indices = mask_indices.to(device)
        
        # Forward pass with specified mask indices
        result = model.forward(
            source,
            padding_mask=None,
            mask=True,
            features_only=False,
            mask_indices=mask_indices,
        )
        
        # Extract loss components
        loss_dict = {
            "regression_loss": result["losses"]["regression"].item() if "regression" in result["losses"] else 0.0,
            "sample_size": result.get("sample_size", 0),
            "target_var": result.get("target_var", torch.tensor(0.0)).item() if torch.is_tensor(result.get("target_var")) else result.get("target_var", 0.0),
            "pred_var": result.get("pred_var", torch.tensor(0.0)).item() if torch.is_tensor(result.get("pred_var")) else result.get("pred_var", 0.0),
        }
        
        # Compute normalized loss
        if loss_dict["sample_size"] > 0:
            loss_dict["normalized_loss"] = loss_dict["regression_loss"] / loss_dict["sample_size"]
        else:
            loss_dict["normalized_loss"] = 0.0
        
        return loss_dict


def verify_mask_reproducibility(
    metadata_path: str,
    model,
    dataset,
    device: torch.device,
    max_samples: int = 10
) -> Dict[str, Any]:
    """
    Verify that masks can be reproduced from saved metadata and compute loss.
    
    Args:
        metadata_path: Path to saved mask metadata
        model: The Data2VecAudioModel instance
        dataset: The dataset to get samples from
        device: Device to run computation on
        max_samples: Maximum number of samples to verify
    
    Returns:
        Dictionary with verification results and loss values
    """
    # Load metadata
    metadata = load_mask_metadata(metadata_path)
    
    seed = metadata["seed"]
    mask_prob = metadata["mask_prob"]
    mask_length = metadata["mask_length"]
    sample_ids = metadata["sample_ids"][:max_samples]
    
    # Get samples from dataset
    samples = []
    for sid in sample_ids:
        if sid < len(dataset):
            samples.append(dataset[sid])
    
    if not samples:
        raise ValueError("No valid samples found in dataset for given sample_ids")
    
    # Stack samples into batch
    sources = torch.stack([s["source"] for s in samples])
    batch_size = sources.shape[0]
    
    # Get sequence length after feature extraction (approximate)
    # This requires a forward pass through feature extractor
    model.eval()
    with torch.no_grad():
        features = model.feature_extractor(sources.to(device))
        seq_length = features.shape[2]  # After conv: (B, C, T)
    
    # Compute reproducible masks
    reproduced_masks = compute_reproducible_mask_indices(
        batch_size=batch_size,
        sequence_length=seq_length,
        mask_prob=mask_prob,
        mask_length=mask_length,
        seed=seed,
        sample_indices=sample_ids
    )
    
    # Check if we have stored masks to compare
    verification_result = {
        "samples_verified": len(samples),
        "seed": seed,
        "mask_prob": mask_prob,
        "mask_length": mask_length,
    }
    
    if "mask_indices" in metadata:
        stored_masks = metadata["mask_indices"]
        # Compare masks (shape might differ, so compare what we can)
        min_samples = min(stored_masks.shape[0], reproduced_masks.shape[0])
        min_seq = min(stored_masks.shape[1], reproduced_masks.shape[1])
        
        stored_subset = stored_masks[:min_samples, :min_seq]
        reproduced_subset = reproduced_masks[:min_samples, :min_seq]
        
        masks_match = np.array_equal(stored_subset, reproduced_subset)
        verification_result["masks_match"] = masks_match
        verification_result["mask_comparison_shape"] = f"({min_samples}, {min_seq})"
    
    # Compute loss with reproduced masks
    loss_result = compute_data2vec_loss(
        model=model,
        source=sources,
        mask_indices=reproduced_masks,
        device=device
    )
    verification_result["loss"] = loss_result
    
    print(f"[+] Verification results:")
    print(f"    - Samples verified: {verification_result['samples_verified']}")
    print(f"    - Masks match stored: {verification_result.get('masks_match', 'N/A')}")
    print(f"    - Normalized loss: {loss_result['normalized_loss']:.6f}")
    
    return verification_result
