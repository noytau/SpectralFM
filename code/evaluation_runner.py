"""
Evaluation Runner for SpectralFM

This module provides functionality to:
1. Discover and load fairseq checkpoints from output directories
2. Run evaluations on trained models
3. Generate comparison reports across runs

Usage:
    python evaluation_runner.py --checkpoint_dir /path/to/outputs --output_dir /path/to/eval_results
    python evaluation_runner.py --checkpoint /path/to/checkpoint_best.pt --output_dir /path/to/eval_results
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import pandas as pd
import torch
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

# Suppress torchaudio/torio FFmpeg extension loading warnings
# These are non-critical - torchaudio will fall back to available FFmpeg versions
logging.getLogger("torio._extension.utils").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("fairseq.trainer").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*FFmpeg.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*libavutil.*", category=UserWarning)

# Add fairseq to path using relative path from this file's location
_FAIRSEQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fairseq")
if _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)

try:
    from model_loader import load_fairseq_checkpoint
except ImportError:
    load_fairseq_checkpoint = None  # Optional import

# Fallback function if model_loader is not available
def _load_fairseq_checkpoint_fallback(checkpoint_path: str):
    """
    Fallback checkpoint loader using fairseq's checkpoint_utils.
    Returns (model, model_cfg, checkpoint_info_dict) to match model_loader interface.
    """
    from fairseq import checkpoint_utils, tasks, utils
    from omegaconf import open_dict
    import os
    
    # Import user module if USER_DIR is set (for custom models like data2vec_audio)
    if 'USER_DIR' in os.environ:
        user_dir = os.environ['USER_DIR']
        if user_dir:
            utils.import_user_module({'user_dir': user_dir})
    
    # Load checkpoint
    overrides = {}
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [checkpoint_path],
        arg_overrides=overrides,
    )
    model = models[0]
    
    # Return in the same format as model_loader
    checkpoint_info = {
        "cfg": saved_cfg,
        "task": task,
    }
    
    return model, saved_cfg.model, checkpoint_info

from omegaconf import OmegaConf


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors, handling shape properly."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_cosine_similarity_matrix(data: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix for rows of data."""
    from sklearn.metrics.pairwise import cosine_similarity
    # Ensure 2D array
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return cosine_similarity(data)


@dataclass
class CheckpointInfo:
    """Information about a discovered checkpoint."""
    path: str
    run_dir: str
    date: str
    time: str
    checkpoint_type: str  # 'best', 'last', or update number
    epoch: int = 0
    num_updates: int = 0
    config: Dict = field(default_factory=dict)
    
    @property
    def run_name(self) -> str:
        return f"{self.date}_{self.time}"


@dataclass 
class EvalResult:
    """Results from a single evaluation run."""
    checkpoint_path: str
    run_name: str
    timestamp: str
    metrics: Dict[str, float] = field(default_factory=dict)
    config_summary: Dict[str, Any] = field(default_factory=dict)
    

class CheckpointDiscovery:
    """Discovers and catalogs fairseq checkpoints in the outputs directory."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
    
    def find_all_checkpoints(self) -> List[CheckpointInfo]:
        """Find all checkpoint files in the outputs directory structure."""
        checkpoints = []
        
        # First try the standard fairseq outputs structure: date/time/checkpoints/
        for date_dir in sorted(self.base_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            
            # Check if this is the standard date directory structure
            if self._is_date_dir(date_dir.name):
                # Walk through time directories
                for time_dir in sorted(date_dir.iterdir()):
                    if not time_dir.is_dir():
                        continue
                        
                    checkpoint_dir = time_dir / "checkpoints"
                    if not checkpoint_dir.exists():
                        continue
                    
                    # Find checkpoint files
                    for ckpt_file in checkpoint_dir.glob("*.pt"):
                        ckpt_info = self._parse_checkpoint(
                            ckpt_file, 
                            date_dir.name, 
                            time_dir.name,
                            time_dir
                        )
                        if ckpt_info:
                            checkpoints.append(ckpt_info)
            
            # Also check for flat structure: date_time/ (runai copied checkpoints)
            elif "_" in date_dir.name:
                # Check if checkpoint_best.pt exists directly in this dir
                ckpt_file = date_dir / "checkpoint_best.pt"
                if ckpt_file.exists():
                    # Parse date_time format: 2025-12-27_09-59-50
                    parts = date_dir.name.split("_")
                    if len(parts) >= 2:
                        date = parts[0]
                        time = "_".join(parts[1:])
                        ckpt_info = self._parse_checkpoint(
                            ckpt_file,
                            date,
                            time,
                            date_dir
                        )
                        if ckpt_info:
                            checkpoints.append(ckpt_info)
        
        # Flat layout: all checkpoints are *.pt files directly under base_dir (no date/ subdirs).
        # Typical for RunAI exports or manual copies with long stems like
        # 2026-04-14_07-42-33_recon-fe1.0_recon-tr0.0_frozen-encFalse_3k.pt
        for ckpt_file in sorted(self.base_dir.glob("*.pt")):
            ckpt_info = self._parse_flat_named_checkpoint(ckpt_file)
            if ckpt_info is not None:
                checkpoints.append(ckpt_info)
        
        return checkpoints
    
    def find_best_checkpoints(self) -> List[CheckpointInfo]:
        """Find only checkpoint_best.pt files."""
        return [c for c in self.find_all_checkpoints() if c.checkpoint_type == 'best']
    
    def find_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Find the most recent best checkpoint."""
        best_checkpoints = self.find_best_checkpoints()
        if not best_checkpoints:
            return None
        return sorted(best_checkpoints, key=lambda c: (c.date, c.time))[-1]
    
    def _is_date_dir(self, name: str) -> bool:
        """Check if directory name looks like a date (YYYY-MM-DD)."""
        parts = name.split('-')
        return len(parts) == 3 and all(p.isdigit() for p in parts)
    
    def _parse_flat_named_checkpoint(self, path: Path) -> Optional[CheckpointInfo]:
        """
        Parse a .pt file sitting directly under base_dir.

        Accepts two naming conventions:
          1. Date-prefixed: ``YYYY-MM-DD_<rest>.pt``  →  date extracted from prefix
          2. Any other ``.pt``:  →  date/time set to "unknown", stem used as run name
        """
        stem = path.stem
        parts = stem.split("_")
        if len(parts) >= 2 and self._is_date_dir(parts[0]):
            date = parts[0]
            time = "_".join(parts[1:])
        else:
            date = "unknown"
            time = stem
        config = self._load_run_config(self.base_dir)
        return CheckpointInfo(
            path=str(path.resolve()),
            run_dir=str(self.base_dir.resolve()),
            date=date,
            time=time,
            checkpoint_type="best",
            config=config,
        )
    
    def _parse_checkpoint(self, path: Path, date: str, time: str, run_dir: Path) -> Optional[CheckpointInfo]:
        """Parse checkpoint file and extract metadata."""
        filename = path.name
        
        # Determine checkpoint type
        if filename == "checkpoint_best.pt":
            ckpt_type = "best"
        elif filename == "checkpoint_last.pt":
            ckpt_type = "last"
        elif filename.startswith("checkpoint_"):
            # Extract update number: checkpoint_1_500.pt -> 1_500
            ckpt_type = filename.replace("checkpoint_", "").replace(".pt", "")
        else:
            return None
        
        # Try to load config from hydra log
        config = self._load_run_config(run_dir)
        
        return CheckpointInfo(
            path=str(path),
            run_dir=str(run_dir),
            date=date,
            time=time,
            checkpoint_type=ckpt_type,
            config=config
        )
    
    def _load_run_config(self, run_dir: Path) -> Dict:
        """Load configuration from hydra_train.log or .hydra directory."""
        config = {}
        
        # Try to load from .hydra/config.yaml
        hydra_config = run_dir / ".hydra" / "config.yaml"
        if hydra_config.exists():
            try:
                cfg = OmegaConf.load(hydra_config)
                config = OmegaConf.to_container(cfg, resolve=True)
            except Exception as e:
                print(f"[!] Error loading hydra config: {e}")
        
        # Parse key params from hydra_train.log as fallback
        log_file = run_dir / "hydra_train.log"
        if log_file.exists() and not config:
            config = self._parse_log_config(log_file)
        
        return config
    
    def _parse_log_config(self, log_file: Path) -> Dict:
        """Extract configuration from hydra_train.log."""
        config = {}
        try:
            with open(log_file, 'r') as f:
                first_line = f.readline()
                # The first log line often contains the full config as JSON
                if '{' in first_line:
                    # Extract JSON portion
                    start = first_line.find('{')
                    json_str = first_line[start:]
                    config = json.loads(json_str.replace("'", '"'))
        except Exception as e:
            pass
        return config


class EvaluationRunner:
    """
    Runs evaluations on checkpoints and collects results.
    
    Data path priority for evaluation:
    1. eval_data_dir parameter (if provided in evaluate_all/evaluate_checkpoint)
    2. self.data_dir (if not default value)
    3. cfg.task.data (from checkpoint config)
    """
    
    def __init__(self, output_dir: str, data_dir: str = "/mnt5/noy/fairseq/data/single_channel_1m/"):
        """
        Initialize EvaluationRunner.
        
        Args:
            output_dir: Directory to save evaluation results
            data_dir: Default evaluation data directory (used if eval_data_dir not provided).
                     Note: This is stored but only used if not the default value.
                     Use eval_data_dir parameter in evaluate_all() for explicit control.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.results: List[EvalResult] = []
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir_out = self.output_dir / "data"
        self.data_dir_out.mkdir(parents=True, exist_ok=True)
        
        # Store intermediate data for comprehensive reporting
        self.eval_data: Dict[str, Any] = {}
        
        # Store pre-loaded samples for reuse across checkpoints
        self._preloaded_samples_valid: Optional[List[Dict]] = None
        self._preloaded_samples_custom: Optional[List[Dict]] = None
        
        # Store selected indices for pre-loaded samples
        self._preloaded_indices_valid: Optional[List[int]] = None
        self._preloaded_indices_custom: Optional[List[int]] = None
        
        # Directory for spectrogram plots
        self.spectrograms_dir = self.output_dir / "spectrograms"
        self.spectrograms_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug directory for data plots
        self.debug_plots_dir = self.output_dir / "debug_plots"
        self.debug_plots_dir.mkdir(parents=True, exist_ok=True)

        # Structured similarity (100-sample nova panel); set in evaluate_all()
        self._nova_data_dir: Optional[str] = None
        self._structured_similarity_seed: int = 42
        self._structured_similarity_entries_json: Optional[str] = None
        self._structured_similarity_prefer_manifest: str = "train"
        self._allow_structured_single_channel_fallback: bool = False
    
    def _plot_spectrogram_with_mask(self, data: np.ndarray, mask_indices: np.ndarray,
                                    run_name: str, sample_id: int, 
                                    mask_prob: float, mask_length: int,
                                    save: bool = True) -> None:
        """
        Plot 245-length spectrogram data with masking overlay.
        
        Args:
            data: 245-length spectrogram data array
            mask_indices: Boolean mask array [245] indicating masked positions
            run_name: Name of the run for file naming
            sample_id: Sample ID for file naming
            mask_prob: Mask probability used (from training config)
            mask_length: Mask length used (from training config)
            save: Whether to save the plot
        """
        # Ensure data is 245 length
        if len(data) != 245:
            if len(data) > 245:
                data = data[:245]
            else:
                # Pad if shorter
                padded = np.zeros(245)
                padded[:len(data)] = data
                data = padded
        
        # Ensure mask_indices matches data length
        if len(mask_indices) != 245:
            if len(mask_indices) > 245:
                mask_indices = mask_indices[:245]
            else:
                # Pad if shorter
                padded_mask = np.zeros(245, dtype=bool)
                padded_mask[:len(mask_indices)] = mask_indices
                mask_indices = padded_mask
        
        # Create masked data
        masked_data = data.copy()
        masked_data[mask_indices] = 0.0
        
        # Plot single figure with overlay
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original data
        x = np.arange(245)
        ax.plot(x, data, 'b-', linewidth=1.5, label='Original', alpha=0.7)
        
        # Plot masked data (overlay)
        ax.plot(x, masked_data, 'r--', linewidth=1.5, label='Masked', alpha=0.7)
        
        # Highlight masked regions with red shading
        masked_positions = np.where(mask_indices)[0]
        if len(masked_positions) > 0:
            # Group consecutive masked positions
            masked_ranges = []
            start = masked_positions[0]
            for i in range(1, len(masked_positions)):
                if masked_positions[i] != masked_positions[i-1] + 1:
                    masked_ranges.append((start, masked_positions[i-1]))
                    start = masked_positions[i]
            masked_ranges.append((start, masked_positions[-1]))
            
            # Shade masked regions
            for start_idx, end_idx in masked_ranges:
                ax.axvspan(start_idx, end_idx + 1, alpha=0.2, color='red', label='Masked Region' if start_idx == masked_ranges[0][0] else '')
        
        ax.set_xlabel('Time Step (245 length)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title(f'Spectrogram with Mask Overlay\n'
                    f'Run: {run_name} | Sample ID: {sample_id}\n'
                    f'Mask Prob: {mask_prob:.2f} | Mask Length: {mask_length} | '
                    f'Masked Positions: {mask_indices.sum()}/245',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            run_spectrograms_dir = self.spectrograms_dir / run_name
            run_spectrograms_dir.mkdir(parents=True, exist_ok=True)
            fname = f"spectrogram_sample{sample_id}_maskprob{mask_prob}_masklen{mask_length}.png"
            plt.savefig(run_spectrograms_dir / fname, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
    def _compute_sample_statistics(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Compute mean and std statistics from a list of samples.
        
        Args:
            samples: List of sample dictionaries with 'source' key containing audio data
            
        Returns:
            Dict with 'mean' and 'std' keys
        """
        if not samples:
            return {"mean": 0.0, "std": 0.0}
        
        all_values = []
        for sample in samples:
            source = sample.get("source", None)
            if source is not None:
                if isinstance(source, torch.Tensor):
                    source = source.cpu().numpy()
                source = np.asarray(source).flatten()
                all_values.extend(source.tolist())
        
        if not all_values:
            return {"mean": 0.0, "std": 0.0}
        
        all_values = np.array(all_values)
        return {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values))
        }
    
    def _compute_dataset_statistics(self, dataset) -> Dict[str, float]:
        """
        Compute mean and std statistics from a fairseq dataset.
        
        Args:
            dataset: Fairseq dataset object
            
        Returns:
            Dict with 'mean' and 'std' keys
        """
        if len(dataset) == 0:
            return {"mean": 0.0, "std": 0.0}
        
        # Sample up to 1000 samples for efficiency
        sample_size = min(1000, len(dataset))
        random.seed(42)
        sample_indices = random.sample(range(len(dataset)), sample_size)
        
        all_values = []
        for idx in sample_indices:
            try:
                sample = dataset[idx]
                source = sample.get("source", None)
                if source is not None:
                    if isinstance(source, torch.Tensor):
                        source = source.cpu().numpy()
                    source = np.asarray(source).flatten()
                    all_values.extend(source.tolist())
            except Exception:
                continue
        
        if not all_values:
            return {"mean": 0.0, "std": 0.0}
        
        all_values = np.array(all_values)
        return {
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values))
        }
    
    def _log_dataset_info(self, dataset_name: str, samples: List[Dict], 
                         selected_indices: Optional[List[int]], 
                         full_dataset=None, run_name: str = None):
        """
        Log detailed information about the evaluated dataset subset.
        
        Args:
            dataset_name: Name of the dataset
            samples: List of samples in the subset
            selected_indices: List of indices that were selected
            full_dataset: Optional full dataset object for computing full dataset stats
            run_name: Optional run name for logging
        """
        print(f"\n{'='*60}")
        print(f"DATASET INFORMATION - {dataset_name}")
        print(f"{'='*60}")
        print(f"Dataset name: {dataset_name}")
        
        if selected_indices is not None:
            print(f"Selected indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''} "
                  f"(total: {len(selected_indices)} samples)")
            if len(selected_indices) > 10:
                print(f"  First 5: {selected_indices[:5]}")
                print(f"  Last 5: {selected_indices[-5:]}")
        else:
            print(f"Selected indices: All samples (total: {len(samples)} samples)")
        
        # Compute subset statistics
        subset_stats = self._compute_sample_statistics(samples)
        print(f"Subset statistics:")
        print(f"  Mean: {subset_stats['mean']:.6f}")
        print(f"  Std:  {subset_stats['std']:.6f}")
        
        # Compute full dataset statistics if available
        if full_dataset is not None:
            full_stats = self._compute_dataset_statistics(full_dataset)
            print(f"Full dataset statistics:")
            print(f"  Mean: {full_stats['mean']:.6f}")
            print(f"  Std:  {full_stats['std']:.6f}")
            print(f"  Total samples: {len(full_dataset)}")
        else:
            print(f"Full dataset statistics: Not available")
        
        print(f"{'='*60}\n")
    
    def _plot_evaluated_data(self, samples: List[Dict], dataset_name: str, 
                             run_name: str, selected_indices: Optional[List[int]] = None,
                             max_plots: int = 10):
        """
        Plot evaluated data samples for debugging.
        
        Args:
            samples: List of samples to plot
            dataset_name: Name of the dataset
            run_name: Run name for directory structure
            selected_indices: Optional list of selected indices
            max_plots: Maximum number of samples to plot
        """
        plot_dir = self.debug_plots_dir / run_name / dataset_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        num_plots = min(max_plots, len(samples))
        for idx in range(num_plots):
            try:
                sample = samples[idx]
                source = sample.get("source", None)
                if source is None:
                    continue
                
                if isinstance(source, torch.Tensor):
                    source = source.cpu().numpy()
                source = np.asarray(source).flatten()
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(source, linewidth=1.5)
                ax.set_xlabel('Time Step', fontsize=12)
                ax.set_ylabel('Amplitude', fontsize=12)
                
                sample_idx = selected_indices[idx] if selected_indices else idx
                ax.set_title(f'Sample {idx} (Dataset Index: {sample_idx})\n'
                            f'Dataset: {dataset_name} | Run: {run_name}\n'
                            f'Mean: {np.mean(source):.6f} | Std: {np.std(source):.6f}',
                            fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                fname = f"sample_{idx}_index_{sample_idx}.png"
                plt.savefig(plot_dir / fname, dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"[!] Warning: Could not plot sample {idx}: {e}")
                continue
        
        print(f"[+] Saved {num_plots} debug plots to: {plot_dir}")
    
    def analyze_similarity_outliers(self, run_name: str, dataset_name: str,
                                    similarity_type: str = "both",
                                    k_outliers: int = 5,
                                    k_neighbors: int = 5,
                                    save_plots: bool = True,
                                    analyze_inliers: bool = True):
        """
        Analyze samples with lowest and highest average cosine similarity and visualize outliers/inliers.
        
        Finds the k_outliers samples with lowest average similarity (outliers) and optionally
        k_outliers samples with highest average similarity (inliers) to the subset,
        then creates visualizations showing each outlier/inlier with its k_neighbors most
        similar and k_neighbors most different samples.
        
        Args:
            run_name: Checkpoint run name
            dataset_name: Dataset name (e.g., "valid", "single_channel_10k")
            similarity_type: "embedding", "input", or "both"
            k_outliers: Number of outlier/inlier samples to analyze
            k_neighbors: Number of similar/different neighbors to show per outlier/inlier
            save_plots: Whether to save the plots
            analyze_inliers: If True, also analyze inliers (highest avg similarity)
        """
        print(f"\n[+] Analyzing similarity outliers for {run_name} on {dataset_name}...")
        print(f"    Similarity type: {similarity_type}, Outliers: {k_outliers}, Neighbors: {k_neighbors}")
        if analyze_inliers:
            print(f"    Also analyzing inliers (highest avg similarity)")
        
        # Step 1: Load embeddings and inputs
        embeddings = None
        inputs = None
        
        if similarity_type in ["embedding", "both"]:
            embeddings_key = f'embeddings_{run_name}_{dataset_name}'
            if embeddings_key in self.eval_data:
                embeddings = self.eval_data[embeddings_key]
            else:
                embeddings_path = self.data_dir_out / f"embeddings_{run_name}_{dataset_name}.npy"
                if embeddings_path.exists():
                    try:
                        embeddings = np.load(embeddings_path)
                    except Exception as e:
                        print(f"[!] Could not load embeddings: {e}")
            
            if embeddings is None or len(embeddings) == 0:
                print(f"[!] No embeddings found for {run_name} on {dataset_name}")
                return
        
        if similarity_type in ["input", "both"]:
            inputs_key = f'inputs_{run_name}_{dataset_name}'
            if inputs_key in self.eval_data:
                inputs = self.eval_data[inputs_key]
            else:
                inputs_path = self.data_dir_out / f"inputs_{run_name}_{dataset_name}.npy"
                if inputs_path.exists():
                    try:
                        inputs = np.load(inputs_path)
                    except Exception as e:
                        print(f"[!] Could not load inputs: {e}")
            
            if inputs is None or len(inputs) == 0:
                print(f"[!] No inputs found for {run_name} on {dataset_name}")
                if similarity_type == "input":
                    return
                # If both, continue with embedding-only analysis
                if similarity_type == "both":
                    similarity_type = "embedding"
        
        # Step 2: Load samples and selected indices
        samples = None
        selected_indices = None
        
        # Determine which preloaded samples to use
        if dataset_name == "valid":
            samples = self._preloaded_samples_valid
            selected_indices = self._preloaded_indices_valid
        else:
            # Assume it's a custom dataset
            samples = self._preloaded_samples_custom
            selected_indices = self._preloaded_indices_custom
        
        if samples is None or len(samples) == 0:
            print(f"[!] No samples found for {dataset_name}")
            return
        
        n_samples = len(samples)
        if selected_indices is None:
            selected_indices = list(range(n_samples))
        
        print(f"[+] Loaded {n_samples} samples with indices: {selected_indices[:5]}...{selected_indices[-5:]}")
        
        # Step 3: Recompute similarity matrices
        sim_matrices = {}
        
        if similarity_type in ["embedding", "both"] and embeddings is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            emb_sim_matrix = cosine_similarity(embeddings)
            sim_matrices["embedding"] = emb_sim_matrix
            print(f"[+] Computed embedding similarity matrix: {emb_sim_matrix.shape}")
        
        if similarity_type in ["input", "both"] and inputs is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            input_sim_matrix = cosine_similarity(inputs)
            sim_matrices["input"] = input_sim_matrix
            print(f"[+] Computed input similarity matrix: {input_sim_matrix.shape}")
        
        # Step 4: Find outliers for each similarity type
        similarity_types_to_analyze = []
        if similarity_type == "both":
            similarity_types_to_analyze = ["embedding", "input"]
        else:
            similarity_types_to_analyze = [similarity_type]
        
        for sim_type in similarity_types_to_analyze:
            if sim_type not in sim_matrices:
                continue
            
            sim_matrix = sim_matrices[sim_type]
            
            # Compute average similarity for each sample (excluding diagonal)
            avg_similarities = []
            for i in range(n_samples):
                # Get all similarities for sample i, excluding self-similarity (diagonal = 1.0)
                similarities = sim_matrix[i, :].copy()
                similarities[i] = np.nan  # Exclude diagonal
                avg_sim = np.nanmean(similarities)
                avg_similarities.append(avg_sim)
            
            avg_similarities = np.array(avg_similarities)
            
            # Find k_outliers samples with lowest average similarity (outliers)
            outlier_indices = np.argsort(avg_similarities)[:k_outliers]
            print(f"\n[+] {sim_type.capitalize()} similarity outliers (lowest avg similarity):")
            for idx in outlier_indices:
                print(f"    Matrix Index {idx} (Dataset Index {selected_indices[idx]}): avg_sim = {avg_similarities[idx]:.4f}")
            
            # Find k_outliers samples with highest average similarity (inliers)
            inlier_indices = None
            if analyze_inliers:
                inlier_indices = np.argsort(avg_similarities)[-k_outliers:][::-1]  # Reverse to get highest first
                print(f"\n[+] {sim_type.capitalize()} similarity inliers (highest avg similarity):")
                for idx in inlier_indices:
                    print(f"    Matrix Index {idx} (Dataset Index {selected_indices[idx]}): avg_sim = {avg_similarities[idx]:.4f}")
            
            # Step 5: Create visualizations for each outlier
            self._plot_outlier_analysis(
                outlier_indices, sim_matrix, samples, selected_indices,
                sim_type, run_name, dataset_name, k_neighbors, save_plots,
                is_inlier=False
            )
            
            # Step 6: Create visualizations for each inlier (if requested)
            if analyze_inliers and inlier_indices is not None:
                self._plot_outlier_analysis(
                    inlier_indices, sim_matrix, samples, selected_indices,
                    sim_type, run_name, dataset_name, k_neighbors, save_plots,
                    is_inlier=True
                )
            
            # Step 7: Create similarity heatmap with markers
            self._plot_similarity_heatmap_with_markers(
                sim_matrix, outlier_indices, inlier_indices if analyze_inliers else None,
                sim_type, run_name, dataset_name, selected_indices, save_plots
            )
    
    def _plot_outlier_analysis(self, outlier_indices: np.ndarray, sim_matrix: np.ndarray,
                               samples: List[Dict], selected_indices: List[int],
                               similarity_type: str, run_name: str, dataset_name: str,
                               k_neighbors: int, save_plots: bool, is_inlier: bool = False):
        """
        Plot outlier/inlier analysis for each sample.
        
        Args:
            outlier_indices: Array of indices of outlier/inlier samples in similarity matrix
            sim_matrix: Similarity matrix [n_samples, n_samples]
            samples: List of sample dictionaries
            selected_indices: List of dataset indices corresponding to samples
            similarity_type: Type of similarity ("embedding" or "input")
            run_name: Checkpoint run name
            dataset_name: Dataset name
            k_neighbors: Number of similar/different neighbors to show
            save_plots: Whether to save plots
            is_inlier: If True, these are inliers (highest avg similarity), else outliers (lowest avg similarity)
        """
        n_samples = len(samples)
        analysis_type = "inliers" if is_inlier else "outliers"
        plot_dir = self.plots_dir / run_name / f"similarity_{analysis_type}_{dataset_name}_{similarity_type}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        for outlier_idx in outlier_indices:
            try:
                # Get similarity scores for this outlier
                outlier_similarities = sim_matrix[outlier_idx, :].copy()
                outlier_similarities[outlier_idx] = -np.inf  # Exclude self
                
                # Find most similar (highest similarity, excluding self)
                most_similar_indices = np.argsort(outlier_similarities)[::-1][:k_neighbors]
                most_similar_scores = outlier_similarities[most_similar_indices]
                
                # Find most different (lowest similarity)
                outlier_similarities[outlier_idx] = np.inf  # Re-exclude for different search
                most_different_indices = np.argsort(outlier_similarities)[:k_neighbors]
                most_different_scores = outlier_similarities[most_different_indices]
                
                # Get outlier sample
                outlier_sample = samples[outlier_idx]
                outlier_source = outlier_sample.get("source", None)
                if outlier_source is None:
                    if "net_input" in outlier_sample and "source" in outlier_sample["net_input"]:
                        outlier_source = outlier_sample["net_input"]["source"]
                    else:
                        print(f"[!] Could not find source for outlier {outlier_idx}")
                        continue
                
                if isinstance(outlier_source, torch.Tensor):
                    outlier_source = outlier_source.cpu().numpy()
                outlier_source = np.asarray(outlier_source).flatten()
                
                # Create plot with 3 rows: similar samples, outlier, different samples
                fig, axes = plt.subplots(3, k_neighbors, figsize=(3 * k_neighbors, 9))
                if k_neighbors == 1:
                    axes = axes.reshape(-1, 1)
                
                # Row 1: Most similar samples
                for i, (sim_idx, sim_score) in enumerate(zip(most_similar_indices, most_similar_scores)):
                    ax = axes[0, i]
                    sim_sample = samples[sim_idx]
                    sim_source = sim_sample.get("source", None)
                    if sim_source is None and "net_input" in sim_sample:
                        sim_source = sim_sample["net_input"].get("source", None)
                    
                    if sim_source is not None:
                        if isinstance(sim_source, torch.Tensor):
                            sim_source = sim_source.cpu().numpy()
                        sim_source = np.asarray(sim_source).flatten()
                        
                        ax.plot(sim_source, linewidth=1.5, color='green')
                        ax.set_title(f'Dataset: {selected_indices[sim_idx]}\n'
                                   f'Matrix: {sim_idx} | Sim: {sim_score:.3f}',
                                   fontsize=9, color='green')
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_xlabel('Time Step', fontsize=8)
                    if i == 0:
                        ax.set_ylabel('Amplitude\n(Most Similar)', fontsize=9)
                    ax.grid(True, alpha=0.3)
                
                # Row 2: Outlier/Inlier sample (centered, highlighted)
                outlier_dataset_idx = selected_indices[outlier_idx]
                # Center the outlier/inlier in the middle row
                center_col = k_neighbors // 2 if k_neighbors > 1 else 0
                outlier_ax = axes[1, center_col]
                label = "INLIER" if is_inlier else "OUTLIER"
                color = 'green' if is_inlier else 'red'
                outlier_ax.plot(outlier_source, linewidth=2, color=color)
                outlier_ax.set_title(f'{label}\n'
                                  f'Dataset: {outlier_dataset_idx} | Matrix: {outlier_idx}\n'
                                  f'Mean: {np.mean(outlier_source):.6f} | Std: {np.std(outlier_source):.6f}',
                                  fontsize=10, fontweight='bold', color=color)
                outlier_ax.set_xlabel('Time Step', fontsize=9)
                outlier_ax.set_ylabel('Amplitude', fontsize=9)
                outlier_ax.grid(True, alpha=0.3)
                # Add bold border
                for spine in outlier_ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                
                # Hide other subplots in middle row
                for i in range(k_neighbors):
                    if i != center_col:
                        ax = axes[1, i]
                        ax.axis('off')
                
                # Row 3: Most different samples
                for i, (diff_idx, diff_score) in enumerate(zip(most_different_indices, most_different_scores)):
                    ax = axes[2, i]
                    diff_sample = samples[diff_idx]
                    diff_source = diff_sample.get("source", None)
                    if diff_source is None and "net_input" in diff_sample:
                        diff_source = diff_sample["net_input"].get("source", None)
                    
                    if diff_source is not None:
                        if isinstance(diff_source, torch.Tensor):
                            diff_source = diff_source.cpu().numpy()
                        diff_source = np.asarray(diff_source).flatten()
                        
                        ax.plot(diff_source, linewidth=1.5, color='blue')
                        ax.set_title(f'Dataset: {selected_indices[diff_idx]}\n'
                                   f'Matrix: {diff_idx} | Sim: {diff_score:.3f}',
                                   fontsize=9, color='blue')
                    else:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                    ax.set_xlabel('Time Step', fontsize=8)
                    if i == 0:
                        ax.set_ylabel('Amplitude\n(Most Different)', fontsize=9)
                    ax.grid(True, alpha=0.3)
                
                analysis_label = "Inlier" if is_inlier else "Outlier"
                plt.suptitle(f'Similarity {analysis_label} Analysis - {similarity_type.capitalize()} Similarity\n'
                           f'Run: {run_name} | Dataset: {dataset_name}\n'
                           f'{analysis_label}: Dataset Index {outlier_dataset_idx} | Matrix Index {outlier_idx}',
                           fontsize=12, fontweight='bold')
                plt.tight_layout()
                
                if save_plots:
                    prefix = "inlier" if is_inlier else "outlier"
                    filename = f"{prefix}_{outlier_idx}_dataset_{outlier_dataset_idx}.png"
                    plt.savefig(plot_dir / filename, dpi=150, bbox_inches='tight')
                    print(f"[+] Saved {analysis_label.lower()} plot: {plot_dir / filename}")
                
                plt.close()
                
            except Exception as e:
                print(f"[!] Error plotting outlier {outlier_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        analysis_type = "inlier" if is_inlier else "outlier"
        print(f"[+] Saved {len(outlier_indices)} {analysis_type} analysis plots to: {plot_dir}")
    
    def _plot_similarity_heatmap_with_markers(self, sim_matrix: np.ndarray,
                                               outlier_indices: np.ndarray,
                                               inlier_indices: Optional[np.ndarray],
                                               similarity_type: str, run_name: str,
                                               dataset_name: str, selected_indices: List[int],
                                               save_plots: bool = True):
        """
        Plot similarity heatmap with star markers indicating outlier and inlier locations.
        
        Creates a heatmap of the similarity matrix and overlays markers:
        - Red stars (*): Outlier samples (lowest avg similarity)
        - Green circles: Inlier samples (highest avg similarity)
        
        Args:
            sim_matrix: Similarity matrix [n_samples, n_samples]
            outlier_indices: Array of indices of outlier samples (lowest avg similarity)
            inlier_indices: Optional array of indices of inlier samples (highest avg similarity)
            similarity_type: Type of similarity ("embedding" or "input")
            run_name: Checkpoint run name
            dataset_name: Dataset name
            selected_indices: List of dataset indices corresponding to samples
            save_plots: Whether to save the plot
        """
        n_samples = sim_matrix.shape[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create heatmap
        sns.heatmap(sim_matrix, ax=ax, cmap="viridis",
                   xticklabels=False, yticklabels=False,
                   vmin=0, vmax=1, cbar=True)
        
        # Add markers for outliers (red stars)
        if outlier_indices is not None and len(outlier_indices) > 0:
            # Mark both row and column positions (symmetric matrix)
            # For each outlier, mark all positions in its row and column
            for outlier_idx in outlier_indices:
                # Mark row positions (all columns for this row)
                ax.scatter(range(n_samples), [outlier_idx] * n_samples,
                          marker='*', s=50, c='red', alpha=0.6, edgecolors='darkred', linewidths=0.5,
                          label='Outlier (lowest avg sim)' if outlier_idx == outlier_indices[0] else '')
                # Mark column positions (all rows for this column)
                ax.scatter([outlier_idx] * n_samples, range(n_samples),
                          marker='*', s=50, c='red', alpha=0.6, edgecolors='darkred', linewidths=0.5)
        
        # Add markers for inliers (green circles)
        if inlier_indices is not None and len(inlier_indices) > 0:
            # Mark both row and column positions (symmetric matrix)
            for inlier_idx in inlier_indices:
                # Mark row positions (all columns for this row)
                ax.scatter(range(n_samples), [inlier_idx] * n_samples,
                          marker='o', s=40, c='lime', alpha=0.7, edgecolors='darkgreen', linewidths=1,
                          label='Inlier (highest avg sim)' if inlier_idx == inlier_indices[0] else '')
                # Mark column positions (all rows for this column)
                ax.scatter([inlier_idx] * n_samples, range(n_samples),
                          marker='o', s=40, c='lime', alpha=0.7, edgecolors='darkgreen', linewidths=1)
        
        # Calculate statistics (excluding diagonal)
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        sim_values = sim_matrix[triu_indices]
        sim_mean = float(np.mean(sim_values))
        sim_std = float(np.std(sim_values))
        
        # Set labels and title
        ax.set_xlabel(f'Sample Index (N={n_samples})', fontsize=12)
        ax.set_ylabel(f'Sample Index (N={n_samples})', fontsize=12)
        
        # Create title
        title = f'Similarity Heatmap with Markers - {similarity_type.capitalize()} Similarity\n'
        title += f'Run: {run_name} | Dataset: {dataset_name}\n'
        title += f'Mean={sim_mean:.3f}, Std={sim_std:.3f}'
        if outlier_indices is not None and len(outlier_indices) > 0:
            outlier_dataset_indices = [selected_indices[idx] for idx in outlier_indices]
            title += f'\nOutliers (red *): Matrix {list(outlier_indices)}, Dataset {outlier_dataset_indices}'
        if inlier_indices is not None and len(inlier_indices) > 0:
            inlier_dataset_indices = [selected_indices[idx] for idx in inlier_indices]
            title += f'\nInliers (green o): Matrix {list(inlier_indices)}, Dataset {inlier_dataset_indices}'
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Add legend
        if (outlier_indices is not None and len(outlier_indices) > 0) or \
           (inlier_indices is not None and len(inlier_indices) > 0):
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        plt.tight_layout()
        
        if save_plots:
            plot_dir = self.plots_dir / run_name
            plot_dir.mkdir(parents=True, exist_ok=True)
            filename = f"similarity_heatmap_with_markers_{dataset_name}_{similarity_type}.png"
            plt.savefig(plot_dir / filename, dpi=150, bbox_inches='tight')
            print(f"[+] Saved similarity heatmap with markers to: {plot_dir / filename}")
        
        plt.close()
    
    def _prepare_model_for_eval(self, model, cfg):
        """
        Prepare model for evaluation: move to correct device.
        Note: model.eval() is NOT called here - fairseq's trainer.valid_step() handles it.
        This function only ensures the model is on the correct device.
        
        Returns:
            model: The prepared model (on correct device)
            device: The device the model is on
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # Note: model.eval() is called by fairseq's trainer.valid_step() internally
        
        # Ensure EMA model is also on correct device
        if hasattr(model, 'ema') and model.ema is not None and hasattr(model.ema, 'model'):
            try:
                ema_device = next(model.ema.model.parameters()).device
                if ema_device != device:
                    model.ema.model = model.ema.model.to(device)
            except Exception:
                pass
        
        return model, device
    
    def evaluate_checkpoint(self, checkpoint_info: CheckpointInfo, 
                          eval_methods: List[str] = None,
                          custom_dataset_path: Optional[str] = None,
                          preloaded_samples_valid: Optional[List[Dict]] = None,
                          preloaded_samples_custom: Optional[List[Dict]] = None,
                          eval_data_dir: Optional[str] = None,
                          debug: bool = False,
                          preloaded_indices_valid: Optional[List[int]] = None,
                          preloaded_indices_custom: Optional[List[int]] = None,
                          include_random_weights: bool = False,
                          mask_memory_path: Optional[str] = None) -> EvalResult:
        """
        Run evaluation on a single checkpoint.
        
        Args:
            checkpoint_info: Information about the checkpoint to evaluate
            eval_methods: List of evaluation methods to run
            custom_dataset_path: Optional custom dataset path for additional evaluation
            preloaded_samples_valid: Optional pre-loaded valid samples (for consistent evaluation across checkpoints)
            preloaded_samples_custom: Optional pre-loaded custom samples
            eval_data_dir: Optional evaluation data directory override. Priority: eval_data_dir > self.data_dir > cfg.task.data
            
        Returns:
            EvalResult with metrics
        """
        if eval_methods is None:
            eval_methods = ["embedding_similarity"]
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint_info.run_name}")
        print(f"Checkpoint: {checkpoint_info.checkpoint_type}")
        print(f"{'='*60}")
        
        # Store current run name for use in sub-methods
        self._current_run_name = checkpoint_info.run_name
        
        # Load model and extract best_loss from checkpoint
        try:
            if load_fairseq_checkpoint is None:
                # Use fallback if model_loader is not available
                model, model_cfg, checkpoint_info_loaded = _load_fairseq_checkpoint_fallback(checkpoint_info.path)
            else:
                model, model_cfg, checkpoint_info_loaded = load_fairseq_checkpoint(checkpoint_info.path)
            cfg = checkpoint_info_loaded["cfg"]  # Full config from checkpoint
            
            # Extract best_loss directly from checkpoint
            best_loss = self._extract_best_loss(checkpoint_info.path)
        except Exception as e:
            print(f"[!] Failed to load checkpoint: {e}")
            return EvalResult(
                checkpoint_path=checkpoint_info.path,
                run_name=checkpoint_info.run_name,
                timestamp=datetime.now().isoformat(),
                metrics={"error": str(e)}
            )
        
        metrics = {"best_loss": best_loss} if best_loss is not None else {}
        
        # Store cfg for use in evaluation methods
        self._current_cfg = cfg
        
        # Prepare model once: device alignment
        # Note: model.eval() is handled by fairseq's trainer.valid_step() during validation
        model, device = self._prepare_model_for_eval(model, cfg)
        
        # Run validation_loss only if it's in eval_methods
        if "validation_loss" in eval_methods:
            print("[+] Running validation loss evaluation on sanity dataset (training data)...")
            sanity_metrics = self._eval_validation_loss(model, cfg, split="sanity", eval_data_dir=None, debug=False, checkpoint_path=checkpoint_info.path, mask_memory_path=mask_memory_path)
            metrics.update({f"sanity_{k}": v for k, v in sanity_metrics.items()})
            
            # print("[+] Running validation loss evaluation on eval dataset...")
            # eval_metrics = self._eval_validation_loss(model, cfg, split="valid", eval_data_dir=eval_data_dir, debug=False, checkpoint_path=checkpoint_info.path, mask_memory_path=mask_memory_path)
            # metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
        else:
            print("[+] Skipping validation loss evaluation (not in eval_methods)")
        
        # Load samples for embedding extraction
        # Use pre-loaded samples if provided (for consistent evaluation across checkpoints)
        print("[+] Loading samples for embedding extraction...")
        selected_indices_valid = None
        if preloaded_samples_valid is not None:
            print(f"[+] Using pre-loaded valid samples: {len(preloaded_samples_valid)} samples")
            samples_for_embeddings = preloaded_samples_valid
            selected_indices_valid = preloaded_indices_valid if preloaded_indices_valid is not None else self._preloaded_indices_valid
            # Still need to get task and dataset for compatibility, but use pre-loaded samples
            task, _, dataset = self.load_eval_dataset_fairseq(
                cfg, split="valid", max_samples=None, verbose=False, eval_data_dir=eval_data_dir
            )
        else:
            # Load fresh samples (single checkpoint evaluation)
            task, samples_for_embeddings, dataset = self.load_eval_dataset_fairseq(
                cfg, split="valid", max_samples=100, verbose=False, eval_data_dir=eval_data_dir
            )
            # Extract indices from samples if available
            if samples_for_embeddings:
                selected_indices_valid = [int(s.get("id", idx)) for idx, s in enumerate(samples_for_embeddings)]
        
        if samples_for_embeddings is None or len(samples_for_embeddings) == 0:
            raise RuntimeError("Failed to load samples for embedding extraction!")
        
        print(f"[+] Loaded {len(samples_for_embeddings)} samples for embedding extraction")
        
        # Log dataset information
        # Determine dataset name
        if eval_data_dir:
            dataset_name_valid = Path(eval_data_dir).name
        elif hasattr(cfg.task, 'data') and cfg.task.data:
            dataset_name_valid = Path(cfg.task.data).name
        else:
            dataset_name_valid = "valid"
        
        self._log_dataset_info(
            dataset_name=dataset_name_valid,
            samples=samples_for_embeddings,
            selected_indices=selected_indices_valid,
            full_dataset=dataset,
            run_name=checkpoint_info.run_name
        )
        
        # Plot evaluated data if debug mode is enabled
        if debug:
            self._plot_evaluated_data(
                samples=samples_for_embeddings,
                dataset_name=dataset_name_valid,
                run_name=checkpoint_info.run_name,
                selected_indices=selected_indices_valid
            )
        
        # Store valid samples from first checkpoint for reuse in comparison plots
        # This ensures the same samples are used in evaluation and comparison
        if self._preloaded_samples_valid is None:
            self._preloaded_samples_valid = samples_for_embeddings
            print(f"[+] Stored valid samples for reuse in comparison plots: {len(samples_for_embeddings)} samples")
        
        # Extract embeddings separately (same pattern as frozen/random)
        embedding_data = self._extract_trained_model_embeddings(
            model, samples_for_embeddings, device
        )
        
        if embedding_data is None:
            raise RuntimeError("Failed to extract embeddings from trained model - this should not happen!")
        
        inputs, embeddings, embedding_samples = embedding_data
        sample_ids = [int(s.get("id", idx)) for idx, s in enumerate(embedding_samples)]
        print(f"[+] Using extracted embeddings: {len(inputs)} samples")
        
        # All evaluation methods now receive the same data from validation_loss
        # Run each evaluation method (skip validation_loss since it already ran)
        for method in eval_methods:
            if method == "validation_loss":
                continue  # Already ran above
            
            try:
                if method == "embedding_similarity":
                    # Run embedding similarity on "valid" dataset
                    print("\n[+] Running embedding similarity on 'valid' dataset...")
                    method_metrics = self._eval_embedding_similarity(
                        checkpoint_info.path,
                        embedding_samples,
                        device,
                        checkpoint_info=checkpoint_info,
                        best_loss=best_loss,
                        dataset_name="valid",
                        include_random_weights=include_random_weights
                    )
                    if method_metrics:
                        # Prefix metrics with dataset name
                        valid_metrics = {f"valid_{k}": v for k, v in method_metrics.items()}
                        metrics.update(valid_metrics)
                        valid_metrics_raw = method_metrics  # Keep original for summary
                    else:
                        valid_metrics_raw = None
                    
                    # Run embedding similarity on custom dataset if provided
                    custom_metrics_raw = None
                    if custom_dataset_path is not None:
                        print(f"\n[+] Running embedding similarity on custom dataset: {custom_dataset_path}...")
                        selected_indices_custom = None
                        dataset_custom = None
                        # Use pre-loaded samples if provided, otherwise load samples from custom dataset
                        if preloaded_samples_custom is not None:
                            print(f"[+] Using pre-loaded custom samples: {len(preloaded_samples_custom)} samples")
                            samples_custom = preloaded_samples_custom
                            selected_indices_custom = preloaded_indices_custom if preloaded_indices_custom is not None else self._preloaded_indices_custom
                            # Load dataset for stats
                            task_custom, _, dataset_custom = self.load_eval_dataset_fairseq(
                                cfg, split="valid", max_samples=None, verbose=False,
                                custom_dataset_path=custom_dataset_path, eval_data_dir=eval_data_dir
                            )
                        else:
                            # Load samples from custom dataset
                            task_custom, samples_custom, dataset_custom = self.load_eval_dataset_fairseq(
                                cfg, split="valid", max_samples=100, verbose=True, 
                                custom_dataset_path=custom_dataset_path, eval_data_dir=eval_data_dir
                            )
                            # Extract indices from samples if available
                            if samples_custom:
                                selected_indices_custom = [int(s.get("id", idx)) for idx, s in enumerate(samples_custom)]
                        
                        if samples_custom is not None and len(samples_custom) > 0:
                            # Log dataset information for custom dataset
                            custom_dataset_name = Path(custom_dataset_path).name
                            self._log_dataset_info(
                                dataset_name=custom_dataset_name,
                                samples=samples_custom,
                                selected_indices=selected_indices_custom,
                                full_dataset=dataset_custom,
                                run_name=checkpoint_info.run_name
                            )
                            
                            # Plot evaluated data if debug mode is enabled
                            if debug:
                                self._plot_evaluated_data(
                                    samples=samples_custom,
                                    dataset_name=custom_dataset_name,
                                    run_name=checkpoint_info.run_name,
                                    selected_indices=selected_indices_custom
                                )
                            custom_metrics = self._eval_embedding_similarity(
                                checkpoint_info.path,
                                samples_custom,
                                device,
                                checkpoint_info=checkpoint_info,
                                best_loss=best_loss,
                                dataset_name=Path(custom_dataset_path).name,
                                include_random_weights=include_random_weights
                            )
                            if custom_metrics:
                                # Prefix metrics with dataset name
                                custom_metrics_prefixed = {f"custom_{k}": v for k, v in custom_metrics.items()}
                                metrics.update(custom_metrics_prefixed)
                                custom_metrics_raw = custom_metrics
                        else:
                            print(f"[!] Warning: Failed to load samples from custom dataset")
                    
                    # Print embedding quality summary comparison
                    # Call summary if we have at least one set of metrics (always call after both evaluations)
                    if valid_metrics_raw is not None or custom_metrics_raw is not None:
                        # Extract config info for dataset name
                        cfg_for_summary = self._current_cfg
                        config_info_summary = self._extract_config_info_for_title(cfg_for_summary)
                        
                        # Ensure we have at least valid_metrics_raw for the summary
                        # (custom_metrics_raw can be None if no custom dataset was provided)
                        if valid_metrics_raw is None and custom_metrics_raw is not None:
                            # Edge case: only custom metrics exist, use them as "valid" for display
                            valid_metrics_raw = custom_metrics_raw
                            custom_metrics_raw = None
                        
                        self._print_embedding_quality_summary(
                            valid_metrics_raw, custom_metrics_raw,
                            dataset_trained=config_info_summary.get('dataset_trained', 'N/A'),
                            custom_dataset_name=Path(custom_dataset_path).name if custom_dataset_path else None
                        )
                elif method == "signal_completion":
                    # Pass model, device, and samples from validation
                    # Check if model has fixed mask set (for sanity testing)
                    fixed_mask_start = getattr(model, '_fixed_mask_start', None)
                    fixed_mask_end = getattr(model, '_fixed_mask_end', None)
                    method_metrics = self._eval_signal_completion(
                        model, device, embedding_samples,
                        fixed_mask_start=fixed_mask_start,
                        fixed_mask_end=fixed_mask_end
                    )
                elif method == "noise_robustness":
                    # Pass model, device, and samples from validation
                    method_metrics = self._eval_noise_robustness(model, device, embedding_samples)
                elif method == "stack_similarity":
                    method_metrics = self._eval_stack_similarity(inputs, embeddings)
                elif method == "structured_similarity":
                    method_metrics = self._eval_structured_similarity(
                        model, device, checkpoint_info, checkpoint_info.path
                    )
                else:
                    print(f"[!] Unknown eval method: {method}")
                    continue
                    
                if method_metrics:
                    metrics.update(method_metrics)
                else:
                    print(f"[!] Warning: {method} returned empty metrics")
            except Exception as e:
                print(f"[!] Error in {method}: {e}")
                import traceback
                traceback.print_exc()
                metrics[f"{method}_error"] = str(e)
        
        # Generate basic 2-panel similarity matrix plot (embeddings are always available from validation)
        # The 4-way comparison is generated inside _eval_embedding_similarity
        embedding_methods = {"embedding_similarity", "stack_similarity"}
        needs_embeddings = bool(embedding_methods & set(eval_methods))
        if needs_embeddings:
            try:
                # Basic 2-panel similarity matrices (separate from 4-way comparison)
                self.plot_similarity_matrices(
                    checkpoint_info, inputs, embeddings,
                    best_loss=best_loss, save_plots=True
                )
            except Exception as e:
                print(f"[!] Error generating similarity plots: {e}")
        
        # Extract key config parameters for comparison
        config_summary = self._extract_config_summary(checkpoint_info.config)
        
        result = EvalResult(
            checkpoint_path=checkpoint_info.path,
            run_name=checkpoint_info.run_name,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            config_summary=config_summary
        )
        
        self.results.append(result)
        return result
    
    def evaluate_all(self, checkpoints: List[CheckpointInfo], 
                    eval_methods: List[str] = None,
                    custom_dataset_path: Optional[str] = None,
                    eval_data_dir: Optional[str] = None,
                    debug: bool = False,
                    include_random_weights: bool = False,
                    mask_memory_path: Optional[str] = None,
                    nova_data_dir: Optional[str] = None,
                    structured_similarity_seed: int = 42,
                    structured_similarity_entries_json: Optional[str] = None,
                    structured_similarity_prefer_manifest: str = "train",
                    structured_similarity_allow_single_channel_fallback: bool = False) -> List[EvalResult]:
        """
        Evaluate multiple checkpoints.
        
        For custom_dataset_path and valid datasets: samples are loaded once (randomly select 100 with fixed seed) 
        and reused across all checkpoints for fair comparison.
        
        Args:
            checkpoints: List of checkpoints to evaluate
            eval_methods: List of evaluation methods to run
            custom_dataset_path: Optional custom dataset path for additional evaluation
            eval_data_dir: Optional evaluation data directory override. Priority: eval_data_dir > self.data_dir > cfg.task.data
            nova_data_dir: Parent directory of nova datasets; used by ``structured_similarity`` (default in CLI: env or /mnt5/noy/fairseq/data).
            structured_similarity_seed: RNG seed for :func:`eval_utils.build_structured_similarity_subset` (default 42).
            structured_similarity_entries_json: Path to ``structured_similarity_full.json`` (100 exact entries); default in CLI is repo file if present.
            structured_similarity_prefer_manifest: ``train`` or ``valid`` for manifest line indices (default ``train``, matches JSON / epoch cosim).
            structured_similarity_allow_single_channel_fallback: If True and full layout is missing, use 10×10 stacks from single_channel_all only.
        """
        if not checkpoints:
            return []

        self._nova_data_dir = nova_data_dir
        self._structured_similarity_seed = structured_similarity_seed
        self._structured_similarity_entries_json = structured_similarity_entries_json
        self._structured_similarity_prefer_manifest = structured_similarity_prefer_manifest
        self._allow_structured_single_channel_fallback = structured_similarity_allow_single_channel_fallback
        
        results = []
        
        # Pre-load samples for custom dataset (randomly select 100 samples for fair comparison)
        samples_custom = None
        if custom_dataset_path:
            print("\n[+] Pre-loading custom dataset samples (will be reused across all checkpoints)...")
            # Load first checkpoint to get config for dataset loading
            try:
                if load_fairseq_checkpoint is None:
                    model_first, model_cfg_first, checkpoint_info_first = _load_fairseq_checkpoint_fallback(checkpoints[0].path)
                else:
                    model_first, model_cfg_first, checkpoint_info_first = load_fairseq_checkpoint(checkpoints[0].path)
                cfg_first = checkpoint_info_first["cfg"]
            except Exception as e:
                print(f"[!] Failed to load first checkpoint for custom dataset sample loading: {e}")
                print("[!] Will load custom dataset samples per checkpoint...")
                samples_custom = None
            else:
                print(f"[+] Loading custom dataset: {custom_dataset_path}...")
                # First, load the full dataset to get its size
                task_custom, _, dataset_custom = self.load_eval_dataset_fairseq(
                    cfg_first, split="valid", max_samples=None, verbose=True,
                    custom_dataset_path=custom_dataset_path, eval_data_dir=eval_data_dir
                )
                
                # Randomly select 100 samples from the full dataset with fixed seed
                dataset_size = len(dataset_custom)
                num_samples = min(100, dataset_size)
                
                if dataset_size > 0:
                    print(f"[+] Randomly selecting {num_samples} samples from {dataset_size} total samples...")
                    random.seed(42)  # Fixed seed for reproducibility
                    
                    # Randomly select indices
                    selected_indices = random.sample(range(dataset_size), num_samples)
                    selected_indices.sort()  # Sort for easier debugging/tracking
                    
                    # Load only the selected samples
                    samples_custom = []
                    for idx in tqdm(selected_indices, desc=f"Loading {num_samples} selected samples"):
                        sample = dataset_custom[idx]
                        samples_custom.append(sample)
                    
                    print(f"[+] Loaded {len(samples_custom)} samples (indices: {selected_indices[:5]}...{selected_indices[-5:]})")
                    # Store indices for logging
                    self._preloaded_indices_custom = selected_indices
                else:
                    print(f"[!] Custom dataset is empty!")
                    samples_custom = None
                    self._preloaded_indices_custom = None
        
        # Pre-load valid samples when evaluating multiple checkpoints (for fair comparison)
        samples_valid = None
        if len(checkpoints) > 1:
            print("\n[+] Pre-loading valid samples (will be reused across all checkpoints)...")
            # Load first checkpoint to get config for dataset loading
            try:
                if load_fairseq_checkpoint is None:
                    model_first, model_cfg_first, checkpoint_info_first = _load_fairseq_checkpoint_fallback(checkpoints[0].path)
                else:
                    model_first, model_cfg_first, checkpoint_info_first = load_fairseq_checkpoint(checkpoints[0].path)
                cfg_first = checkpoint_info_first["cfg"]
            except Exception as e:
                print(f"[!] Failed to load first checkpoint for valid sample loading: {e}")
                print("[!] Will load valid samples per checkpoint...")
                samples_valid = None
            else:
                print(f"[+] Loading valid dataset...")
                # Load the full dataset to get its size
                task_valid, _, dataset_valid = self.load_eval_dataset_fairseq(
                    cfg_first, split="valid", max_samples=None, verbose=True,
                    eval_data_dir=eval_data_dir
                )
                
                # Randomly select 100 samples with fixed seed
                dataset_size = len(dataset_valid)
                num_samples = min(100, dataset_size)
                
                if dataset_size > 0:
                    print(f"[+] Randomly selecting {num_samples} samples from {dataset_size} total samples...")
                    random.seed(42)  # Fixed seed for reproducibility
                    
                    # Randomly select indices
                    selected_indices = random.sample(range(dataset_size), num_samples)
                    selected_indices.sort()  # Sort for easier debugging/tracking
                    
                    # Load only the selected samples
                    samples_valid = []
                    for idx in tqdm(selected_indices, desc=f"Loading {num_samples} selected samples"):
                        sample = dataset_valid[idx]
                        samples_valid.append(sample)
                    
                    print(f"[+] Loaded {len(samples_valid)} valid samples (indices: {selected_indices[:5]}...{selected_indices[-5:]})")
                    # Store indices for logging
                    self._preloaded_indices_valid = selected_indices
                else:
                    print(f"[!] Valid dataset is empty!")
                    samples_valid = None
                    self._preloaded_indices_valid = None
        
        # Store samples in runner instance for reuse in other functions
        self._preloaded_samples_custom = samples_custom
        self._preloaded_samples_valid = samples_valid
        
        # Evaluate all checkpoints
        print(f"\n[+] Evaluating {len(checkpoints)} checkpoints...")
        for ckpt in tqdm(checkpoints, desc="Evaluating checkpoints"):
            result = self.evaluate_checkpoint(
                ckpt, 
                eval_methods, 
                custom_dataset_path=custom_dataset_path,
                preloaded_samples_valid=samples_valid,  # Use pre-loaded valid samples
                preloaded_samples_custom=samples_custom,  # Use pre-loaded custom samples
                eval_data_dir=eval_data_dir,  # Pass eval_data_dir to evaluate_checkpoint
                debug=debug,  # Pass debug flag
                preloaded_indices_valid=self._preloaded_indices_valid,  # Pass indices
                preloaded_indices_custom=self._preloaded_indices_custom,  # Pass indices
                include_random_weights=include_random_weights,  # Pass include_random_weights flag
                mask_memory_path=mask_memory_path  # Pass mask_memory_path
            )
            results.append(result)
        
        if "structured_similarity" in (eval_methods or []):
            self._write_structured_similarity_multi_plot(eval_methods, results, checkpoints)
        
        return results
    
    def _extract_embeddings(self, model, device, max_samples: int = 100) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Extract embeddings using fairseq's data loading infrastructure.
        Assumes model is already in eval mode and on correct device.
        
        Args:
            model: The loaded model (already in eval mode)
            device: The device to use
            max_samples: Maximum number of samples to process
            
        Returns:
            Tuple of (inputs, embeddings, sample_ids)
        """
        # Load dataset using fairseq infrastructure
        _, samples, _ = self.load_eval_dataset_fairseq(
            self._current_cfg, split="valid", max_samples=max_samples, verbose=False
        )
        
        inputs = []
        embeddings = []
        sample_ids = []
        
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(samples, desc="Extracting embeddings")):
                try:
                    source = sample["source"]
                    sample_id = sample.get("id", idx)
                    
                    # Prepare input: [batch, seq_len]
                    data = source.to(device)
                    if data.dim() == 1:
                        data = data.unsqueeze(0)
                    
                    # Store input for input-space similarity
                    inputs.append(source.cpu().numpy())
                    
                    # Get features (no masking for embedding extraction)
                    result = model.extract_features(data, padding_mask=None, mask=False)
                    emb = result["x"].mean(dim=1).cpu().numpy().squeeze()
                    embeddings.append(emb)
                    sample_ids.append(int(sample_id))
                    
                except Exception as e:
                    print(f"[!] Error processing sample {idx}: {e}")
                    continue
        
        inputs_arr = np.stack(inputs) if inputs else np.array([])
        embeddings_arr = np.stack(embeddings) if embeddings else np.array([])
        
        return inputs_arr, embeddings_arr, sample_ids
    
    def _extract_embeddings_from_samples(self, model, device, samples: List[Dict], 
                                         handle_batches: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from pre-loaded samples (ensures exact same samples for all models).
        
        Args:
            model: The model (already in eval mode)
            device: The device to use
            samples: Pre-loaded samples from fairseq dataset (can be individual samples or batched)
            handle_batches: If True, handles batched samples (splits them into individual samples)
        
        Returns:
            Tuple of (inputs, embeddings)
        """
        inputs = []
        embeddings = []
        failed_samples = 0
        
        # Get model dtype for dtype matching
        model_dtype = next(model.parameters()).dtype
        
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(samples, desc="Extracting embeddings")):
                try:
                    # Handle different sample structures
                    if "source" not in sample:
                        # Try alternative keys
                        if "net_input" in sample and "source" in sample["net_input"]:
                            source = sample["net_input"]["source"]
                        elif "src_tokens" in sample:
                            source = sample["src_tokens"]
                        else:
                            raise KeyError(f"Sample {idx} does not have 'source' key. Available keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
                    else:
                        source = sample["source"]
                    
                    # Convert to tensor if it's not already
                    if not isinstance(source, torch.Tensor):
                        source = torch.tensor(source) if isinstance(source, (list, np.ndarray)) else source
                    
                    # Handle batched samples
                    if handle_batches and isinstance(source, torch.Tensor) and source.dim() > 1 and source.shape[0] > 1:
                        # Process each sample in the batch
                        batch_size = source.shape[0]
                        for b in range(batch_size):
                            source_single = source[b]
                            data = source_single.to(device)
                            if data.dim() == 1:
                                data = data.unsqueeze(0)
                            
                            # Ensure dtype matches model
                            if data.dtype != model_dtype:
                                data = data.to(dtype=model_dtype)
                            
                            # Store input
                            inputs.append(source_single.cpu().numpy())
                            
                            # Extract embeddings
                            result = model.extract_features(data, padding_mask=None, mask=False)
                            emb = result["x"].mean(dim=1).cpu().numpy().squeeze()
                            embeddings.append(emb)
                    else:
                        # Single sample (handle_batches=False means samples are already individual)
                        # Convert to tensor if needed and move to device
                        if isinstance(source, torch.Tensor):
                            # Ensure tensor is contiguous (in case it's a view/slice)
                            source_contig = source.contiguous() if not source.is_contiguous() else source
                            data = source_contig.to(device)
                        elif isinstance(source, np.ndarray):
                            data = torch.from_numpy(source).to(device)
                        else:
                            data = torch.tensor(source).to(device)
                        
                        if data.dim() == 1:
                            data = data.unsqueeze(0)
                        
                        # Ensure dtype matches model
                        if data.dtype != model_dtype:
                            data = data.to(dtype=model_dtype)
                        
                        # Store input (convert to numpy, ensure it's on CPU and contiguous)
                        if isinstance(source, torch.Tensor):
                            # Detach and clone to ensure we have a CPU copy
                            source_np = source.detach().cpu().contiguous().numpy()
                        else:
                            source_np = np.array(source)
                        inputs.append(source_np)
                        
                        # Extract embeddings
                        result = model.extract_features(data, padding_mask=None, mask=False)
                        emb = result["x"].mean(dim=1).cpu().numpy().squeeze()
                        embeddings.append(emb)
                    
                except Exception as e:
                    failed_samples += 1
                    if failed_samples <= 3:  # Only print first 3 errors to avoid spam
                        print(f"[!] Error processing sample {idx}: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
        
        if failed_samples > 0:
            print(f"[!] Warning: Failed to extract embeddings from {failed_samples}/{len(samples)} samples")
        
        if len(inputs) == 0:
            print(f"[!] Error: No embeddings extracted! All {len(samples)} samples failed.")
        
        inputs_arr = np.stack(inputs) if inputs else np.array([])
        embeddings_arr = np.stack(embeddings) if embeddings else np.array([])
        
        return inputs_arr, embeddings_arr
    
    def _extract_inputs_from_samples(self, samples: List[Dict]) -> np.ndarray:
        """
        Extract input data from samples (without needing a model).
        Ensures consistent input extraction across all embedding sources.
        
        Args:
            samples: Pre-loaded samples from fairseq dataset
            
        Returns:
            Inputs array [N, seq_len]
        """
        inputs = []
        
        for idx, sample in enumerate(samples):
            try:
                # Handle different sample structures
                if "source" not in sample:
                    if "net_input" in sample and "source" in sample["net_input"]:
                        source = sample["net_input"]["source"]
                    elif "src_tokens" in sample:
                        source = sample["src_tokens"]
                    else:
                        raise KeyError(f"Sample {idx} does not have 'source' key. Available keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
                else:
                    source = sample["source"]
                
                # Convert to numpy
                if isinstance(source, torch.Tensor):
                    # Handle batched samples
                    if source.dim() > 1 and source.shape[0] > 1:
                        batch_size = source.shape[0]
                        for b in range(batch_size):
                            source_single = source[b]
                            inputs.append(source_single.detach().cpu().contiguous().numpy())
                    else:
                        inputs.append(source.detach().cpu().contiguous().numpy())
                elif isinstance(source, np.ndarray):
                    if source.ndim > 1 and source.shape[0] > 1:
                        # Batched
                        for b in range(source.shape[0]):
                            inputs.append(source[b])
                    else:
                        inputs.append(source)
                else:
                    inputs.append(np.array(source))
                    
            except Exception as e:
                print(f"[!] Error extracting input from sample {idx}: {e}")
                continue
        
        if len(inputs) == 0:
            print(f"[!] Error: No inputs extracted from {len(samples)} samples")
            return np.array([])
        
        return np.stack(inputs)
    
    def _load_embeddings_from_checkpoint(self, checkpoint_path: str, samples: List[Dict], device, 
                                         checkpoint_name: str = "checkpoint") -> Optional[np.ndarray]:
        """
        Generic function to load embeddings from a checkpoint path.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            samples: Pre-loaded samples from fairseq dataset
            device: The device to use
            checkpoint_name: Name for logging purposes (e.g., "frozen encoder", "random init")
            
        Returns:
            Embeddings array or None if failed
        """
        if samples is None:
            print(f"[!] Warning: No samples provided for {checkpoint_name} embeddings")
            return None
        
        try:
            if not os.path.exists(checkpoint_path):
                print(f"[!] {checkpoint_name.capitalize()} checkpoint not found at {checkpoint_path}")
                return None
            
            print(f"[+] Loading {checkpoint_name} checkpoint from {checkpoint_path}...")
            if load_fairseq_checkpoint is None:
                model, model_cfg, checkpoint_info = _load_fairseq_checkpoint_fallback(checkpoint_path)
            else:
                model, model_cfg, checkpoint_info = load_fairseq_checkpoint(checkpoint_path)
            cfg = checkpoint_info["cfg"]  # Full config from checkpoint
            self._current_cfg = cfg
            model, model_device = self._prepare_model_for_eval(model, cfg)
            
            # Use the device from _prepare_model_for_eval (model might have been moved)
            # Extract embeddings using the EXACT same samples
            _, embeddings = self._extract_embeddings_from_samples(
                model, model_device, samples
            )
            del model  # Free memory
            return embeddings
            
        except Exception as e:
            print(f"[!] Failed to load {checkpoint_name} embeddings: {e}")
            return None
    
    def _load_frozen_encoder_embeddings(self, samples: List[Dict], device) -> Optional[np.ndarray]:
        """
        Load embeddings from frozen encoder checkpoint (train_only_fe=True).
        
        Args:
            samples: Pre-loaded samples from fairseq dataset
            device: The device to use
            
        Returns:
            Embeddings array or None if failed
        """
        FROZEN_ENCODER_PATH = "/mnt5/noy/SpectralFM/checkpoints/runai/2025-12-27_00-06-01/checkpoint_best.pt"
        
        return self._load_embeddings_from_checkpoint(
            FROZEN_ENCODER_PATH, samples, device, checkpoint_name="frozen encoder"
        )
    
    def _load_random_init_embeddings(self, samples: List[Dict], device) -> Optional[np.ndarray]:
        """
        Load embeddings from a random init checkpoint (no training).
        
        Args:
            samples: Pre-loaded samples from fairseq dataset
            device: The device to use
            
        Returns:
            Embeddings array or None if failed
        """
        # TODO: Update this path to point to a random init checkpoint
        RANDOM_INIT_PATH = "/mnt5/noy/SpectralFM/checkpoints/runai/2026-01-12_11-44-25/checkpoint_best.pt"  # FIXME: Replace with actual random init checkpoint
        
        return self._load_embeddings_from_checkpoint(
            RANDOM_INIT_PATH, samples, device, checkpoint_name="random init"
        )
    
    def _eval_random_weight_embeddings(self, samples: List[Dict], device, 
                                       resample_to_16k: bool = True,
                                       cfg=None) -> Optional[np.ndarray]:
        """
        Extract embeddings from pretrained Data2VecAudio model from transformers.
        
        This function:
        1. Optionally stretches samples to 16kHz length (exactly 16000 samples)
        2. Loads pretrained Data2VecAudio model from transformers (facebook/data2vec-audio-base)
        3. Extracts embeddings using the same samples
        
        Args:
            samples: Pre-loaded samples from load_eval_dataset_fairseq() (must be from fairseq dataset)
            device: Device to use
            resample_to_16k: If True, stretch samples to 16000 samples before processing
            cfg: Model config (ignored, kept for compatibility)
        
        Returns:
            Embeddings array [N, embed_dim] or None if extraction failed
        """
        if samples is None or len(samples) == 0:
            print("[!] Warning: No samples provided for pretrained model embeddings")
            return None
        
        try:
            print(f"[+] Extracting embeddings from transformers pretrained Data2VecAudio model ({len(samples)} samples)...")
            
            # Step 1: Stretch samples if requested
            if resample_to_16k:
                print(f"[+] Stretching samples to 16000 samples...")
                from audio_preprocessing import stretch_samples_to_16k
                samples = stretch_samples_to_16k(samples, target_length=16000, verbose=True)
            
            # Step 2: Load pretrained model from transformers
            print(f"[+] Loading pretrained Data2VecAudio model from transformers...")
            from transformers import Data2VecAudioModel
            model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
            model = model.to(device)
            model.eval()
            print(f"[+] Model loaded successfully on {device}")
            
            # Step 3: Extract embeddings using existing function
            # Note: Transformers model's extract_features returns BaseModelOutput
            # which has 'last_hidden_state' instead of 'x'. We need to adapt.
            print(f"[+] Extracting embeddings from pretrained model...")
            _, embeddings = self._extract_embeddings_from_samples_transformers(
                model, device, samples, handle_batches=True
            )
            
            if embeddings is None or len(embeddings) == 0:
                print(f"[!] Warning: Failed to extract embeddings from pretrained model")
                return None
            
            print(f"[+] Successfully extracted {len(embeddings)} embeddings from pretrained model")
            print(f"    - Embedding shape: {embeddings.shape}")
            
            # Clean up model to free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return embeddings
            
        except Exception as e:
            print(f"[!] Failed to extract pretrained model embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_embeddings_from_samples_transformers(self, model, device, samples: List[Dict], 
                                                       handle_batches: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from transformers Data2VecAudio model.
        
        This is similar to _extract_embeddings_from_samples but adapted for transformers models,
        which return BaseModelOutput with 'last_hidden_state' instead of dict with 'x'.
        
        Args:
            model: The transformers model (already in eval mode)
            device: The device to use
            samples: Pre-loaded samples from fairseq dataset
            handle_batches: If True, handles batched samples (splits them into individual samples)
        
        Returns:
            Tuple of (inputs, embeddings)
        """
        inputs = []
        embeddings = []
        failed_samples = 0
        
        # Get model dtype for dtype matching
        model_dtype = next(model.parameters()).dtype
        
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(samples, desc="Extracting embeddings (transformers)")):
                try:
                    # Handle different sample structures
                    if "source" not in sample:
                        # Try alternative keys
                        if "net_input" in sample and "source" in sample["net_input"]:
                            source = sample["net_input"]["source"]
                        elif "src_tokens" in sample:
                            source = sample["src_tokens"]
                        else:
                            raise KeyError(f"Sample {idx} does not have 'source' key. Available keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
                    else:
                        source = sample["source"]
                    
                    # Convert to tensor if it's not already
                    if not isinstance(source, torch.Tensor):
                        source = torch.tensor(source) if isinstance(source, (list, np.ndarray)) else source
                    
                    # Handle batched samples
                    if handle_batches and isinstance(source, torch.Tensor) and source.dim() > 1 and source.shape[0] > 1:
                        # Process each sample in the batch
                        batch_size = source.shape[0]
                        for b in range(batch_size):
                            source_single = source[b]
                            data = source_single.to(device)
                            if data.dim() == 1:
                                data = data.unsqueeze(0)  # [1, L]
                            
                            # Ensure dtype matches model
                            if data.dtype != model_dtype:
                                data = data.to(dtype=model_dtype)
                            
                            # Store input
                            inputs.append(source_single.cpu().numpy())
                            
                            # Extract embeddings using transformers model
                            # Transformers models expect input_values, not source
                            output = model(input_values=data)
                            # Transformers returns BaseModelOutput with last_hidden_state
                            # Shape: [batch, seq_len, hidden_dim]
                            emb = output.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()  # [hidden_dim]
                            embeddings.append(emb)
                    else:
                        # Single sample
                        if isinstance(source, torch.Tensor):
                            source_contig = source.contiguous() if not source.is_contiguous() else source
                            data = source_contig.to(device)
                        elif isinstance(source, np.ndarray):
                            data = torch.from_numpy(source).to(device)
                        else:
                            data = torch.tensor(source).to(device)
                        
                        if data.dim() == 1:
                            data = data.unsqueeze(0)  # [1, L]
                        
                        # Ensure dtype matches model
                        if data.dtype != model_dtype:
                            data = data.to(dtype=model_dtype)
                        
                        # Store input
                        if isinstance(source, torch.Tensor):
                            source_np = source.detach().cpu().contiguous().numpy()
                        else:
                            source_np = np.array(source)
                        inputs.append(source_np)
                        
                        # Extract embeddings using transformers model
                        # Transformers models expect input_values, not source
                        output = model(input_values=data)
                        # Transformers returns BaseModelOutput with last_hidden_state
                        # Shape: [batch, seq_len, hidden_dim]
                        emb = output.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()  # [hidden_dim]
                        embeddings.append(emb)
                    
                except Exception as e:
                    failed_samples += 1
                    if failed_samples <= 3:  # Only print first 3 errors to avoid spam
                        print(f"[!] Error processing sample {idx}: {e}")
                        import traceback
                        traceback.print_exc()
                    continue
        
        if failed_samples > 0:
            print(f"[!] Warning: Failed to extract embeddings from {failed_samples}/{len(samples)} samples")
        
        if len(inputs) == 0:
            print(f"[!] Error: No embeddings extracted! All {len(samples)} samples failed.")
        
        inputs_arr = np.stack(inputs) if inputs else np.array([])
        embeddings_arr = np.stack(embeddings) if embeddings else np.array([])
        
        return inputs_arr, embeddings_arr
    
    def _eval_embedding_similarity(self, checkpoint_path: str, samples: List[Dict], device,
                                    checkpoint_info: CheckpointInfo = None,
                                    best_loss: Optional[float] = None,
                                    dataset_name: str = "valid",
                                    include_random_weights: bool = False) -> Dict[str, float]:
        """
        Evaluate embedding quality by comparing input-space vs embedding-space similarity.
        Also generates a 3-way (or 4-way if include_random_weights) similarity matrix comparison plot.
        
        All embeddings (evaluated model, frozen encoder, optionally random weights with 16k) 
        are loaded using the EXACT same samples to ensure fair comparison.
        
        Args:
            checkpoint_path: Path to the evaluated model checkpoint
            samples: Pre-loaded samples (used for ALL embedding extractions)
            device: Device to use (for all models)
            checkpoint_info: Checkpoint info (for saving plots)
            best_loss: Training loss (for plot title)
            dataset_name: Name of the dataset (for plot filename)
            include_random_weights: If True, include random weight model (16k resampled) in comparison
            
        A good model should:
        - Preserve relative similarities (similar inputs -> similar embeddings)
        - Have higher variance in embedding similarity than a collapsed model
        - Show correlation between input and embedding similarities
        """
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.stats import pearsonr, spearmanr
        
        print(f"[+] Running embedding similarity analysis...")
        
        if samples is None or len(samples) == 0:
            return {"error": "No samples provided for embedding similarity analysis"}
        
        if device is None:
            return {"error": "No device provided for embedding similarity analysis"}
        
        # Store original cfg
        original_cfg = self._current_cfg
        
        # Extract inputs from samples (same for all comparisons)
        print(f"[+] Extracting inputs from {len(samples)} samples...")
        inputs = self._extract_inputs_from_samples(samples)
        
        if len(inputs) == 0:
            return {"error": "Failed to extract inputs from samples"}
        
        n_samples = len(inputs)
        print(f"[+] Extracted inputs from {n_samples} samples")
        
        # Load embeddings from evaluated model checkpoint
        print(f"[+] Loading embeddings from evaluated model checkpoint...")
        embeddings = self._load_embeddings_from_checkpoint(
            checkpoint_path, samples, device, checkpoint_name="evaluated model"
        )
        
        if embeddings is None or len(embeddings) == 0:
            return {"error": "Failed to load embeddings from evaluated model checkpoint"}
        
        if len(embeddings) != n_samples:
            error_msg = f"Mismatch: evaluated model embeddings has {len(embeddings)} samples, expected {n_samples}"
            print(f"[!] Error: {error_msg}")
            return {"error": error_msg}
        
        # Load baseline embeddings (using the EXACT same samples)
        embeddings_frozen = None
        best_loss_frozen = None
        
        embeddings_frozen = self._load_frozen_encoder_embeddings(samples, device)
        
        # Extract best_loss from frozen encoder checkpoint
        if embeddings_frozen is not None:
            FROZEN_ENCODER_PATH = "/mnt5/noy/SpectralFM/checkpoints/runai/2025-12-27_00-06-01/checkpoint_best.pt"
            best_loss_frozen = self._extract_best_loss(FROZEN_ENCODER_PATH)
            if best_loss_frozen is not None:
                print(f"[+] Frozen encoder best_loss: {best_loss_frozen:.4f}")
        
        # Load random weight embeddings (16k resampled) if requested
        embeddings_random_weights = None
        if include_random_weights:
            print(f"[+] Loading embeddings from random weight model (16k resampled)...")
            embeddings_random_weights = self._eval_random_weight_embeddings(
                samples, device, resample_to_16k=True, cfg=original_cfg
            )
            
            if embeddings_random_weights is None:
                print(f"[!] Error: Failed to load random weight embeddings even though include_random_weights=True!")
            else:
                print(f"[+] Successfully loaded {len(embeddings_random_weights)} random weight embeddings")
        
        # Validate baseline embeddings have the same count
        if embeddings_frozen is not None:
            n_frozen = len(embeddings_frozen)
            if n_frozen != n_samples:
                print(f"[!] Warning: Frozen encoder embeddings has {n_frozen} samples, expected {n_samples}. Skipping frozen encoder comparison.")
                embeddings_frozen = None
                best_loss_frozen = None
        
        if embeddings_random_weights is not None:
            n_random_weights = len(embeddings_random_weights)
            if n_random_weights != n_samples:
                print(f"[!] Warning: Random weight embeddings has {n_random_weights} samples, expected {n_samples}. Skipping random weight comparison.")
                print(f"[!] This may indicate a problem with the random weight embedding extraction.")
                embeddings_random_weights = None
            else:
                print(f"[+] Random weight embeddings validated: {n_random_weights} samples (matches expected {n_samples})")
        
        # Restore original cfg
        self._current_cfg = original_cfg
        
        # Final validation: all embeddings should have the same number of samples
        if len(embeddings) < 2:
            return {"error": "Not enough valid samples for similarity computation"}
        
        # Compute similarity matrices (all should have the same number of samples now)
        input_sim_matrix = cosine_similarity(inputs)
        emb_sim_matrix = cosine_similarity(embeddings)
        emb_sim_frozen = cosine_similarity(embeddings_frozen) if embeddings_frozen is not None else None
        
        # Compute random weights similarity matrix if available
        emb_sim_random_weights = None
        if embeddings_random_weights is not None:
            try:
                emb_sim_random_weights = cosine_similarity(embeddings_random_weights)
                print(f"[+] Computed random weights similarity matrix: shape {emb_sim_random_weights.shape}")
            except Exception as e:
                print(f"[!] Error computing random weights similarity matrix: {e}")
                emb_sim_random_weights = None
        
        # Final validation: ensure all similarity matrices have the same shape
        expected_shape = (n_samples, n_samples)
        assert input_sim_matrix.shape == expected_shape, f"Input sim matrix shape {input_sim_matrix.shape} != expected {expected_shape}"
        assert emb_sim_matrix.shape == expected_shape, f"Embed sim matrix shape {emb_sim_matrix.shape} != expected {expected_shape}"
        if emb_sim_frozen is not None:
            assert emb_sim_frozen.shape == expected_shape, f"Frozen sim matrix shape {emb_sim_frozen.shape} != expected {expected_shape}"
        if emb_sim_random_weights is not None:
            assert emb_sim_random_weights.shape == expected_shape, f"Random weights sim matrix shape {emb_sim_random_weights.shape} != expected {expected_shape}"
        
        # Get upper triangle indices (excluding diagonal)
        triu_idx = np.triu_indices_from(input_sim_matrix, k=1)
        
        input_sims = input_sim_matrix[triu_idx]
        emb_sims = emb_sim_matrix[triu_idx]
        
        # Store embedding similarity scores and embeddings for later histogram/comparison plots
        if checkpoint_info is not None:
            run_name = checkpoint_info.run_name
            if run_name not in self.eval_data:
                self.eval_data[run_name] = {}
            # Store scores with dataset_name as key to support multiple datasets per run
            scores_key = f'embedding_similarity_scores_{dataset_name}'
            self.eval_data[run_name][scores_key] = emb_sims
            
            # Store input similarity scores for later histogram comparison
            input_scores_key = f'input_similarity_scores_{dataset_name}'
            self.eval_data[run_name][input_scores_key] = input_sims
            
            # Store embeddings with dataset_name in key for side-by-side comparison plots
            embeddings_key = f'embeddings_{run_name}_{dataset_name}'
            self.eval_data[embeddings_key] = embeddings
            
            # Store inputs with dataset_name in key for outlier analysis
            inputs_key = f'inputs_{run_name}_{dataset_name}'
            self.eval_data[inputs_key] = inputs
            
            # Save to numpy file for persistence (include dataset_name in filename)
            scores_path = self.data_dir_out / f"embedding_similarity_scores_{run_name}_{dataset_name}.npy"
            np.save(scores_path, emb_sims)
            
            # Save input similarity scores to numpy file
            input_scores_path = self.data_dir_out / f"input_similarity_scores_{run_name}_{dataset_name}.npy"
            np.save(input_scores_path, input_sims)
            
            # Also save embeddings for later use
            embeddings_path = self.data_dir_out / f"embeddings_{run_name}_{dataset_name}.npy"
            np.save(embeddings_path, embeddings)
            
            # Save inputs for later use
            inputs_path = self.data_dir_out / f"inputs_{run_name}_{dataset_name}.npy"
            np.save(inputs_path, inputs)
        
        # Compute correlation between input and embedding similarities
        pearson_corr, pearson_p = pearsonr(input_sims, emb_sims)
        spearman_corr, spearman_p = spearmanr(input_sims, emb_sims)
        
        # Metrics
        metrics = {
            "input_mean_sim": float(np.mean(input_sims)),
            "input_std_sim": float(np.std(input_sims)),
            "emb_mean_sim": float(np.mean(emb_sims)),
            "emb_std_sim": float(np.std(emb_sims)),
            "pearson_corr": float(pearson_corr),
            "pearson_p_value": float(pearson_p),
            "spearman_corr": float(spearman_corr),
            "spearman_p_value": float(spearman_p),
            "sim_variance_ratio": float(np.std(emb_sims) / (np.std(input_sims) + 1e-8)),
            "emb_dim": embeddings.shape[1],
            "num_samples": len(embeddings),
            "num_pairs": len(input_sims),
        }
        
        print(f"[+] Embedding Similarity Analysis:")
        print(f"    Input space:  mean={metrics['input_mean_sim']:.4f}, std={metrics['input_std_sim']:.4f}")
        print(f"    Embed space:  mean={metrics['emb_mean_sim']:.4f}, std={metrics['emb_std_sim']:.4f}")
        print(f"    Pearson corr: {metrics['pearson_corr']:.4f} (p={metrics['pearson_p_value']:.2e})")
        print(f"    Variance ratio: {metrics['sim_variance_ratio']:.4f}")
        
        # Generate 4-way similarity matrix comparison plot
        if checkpoint_info is not None:
            # Extract config info for plot title
            cfg = self._current_cfg
            config_info = self._extract_config_info_for_title(cfg)
            
            # Ensure random weights matrix is passed if include_random_weights was requested
            if include_random_weights and emb_sim_random_weights is None:
                print(f"[!] Warning: include_random_weights=True but emb_sim_random_weights is None. Plot will not include random weights model.")
            
            self._plot_embedding_similarity_comparison(
                checkpoint_info, input_sim_matrix, emb_sim_matrix,
                emb_sim_frozen, None, best_loss, best_loss_frozen,
                dataset_name=dataset_name, config_info=config_info,
                emb_sim_random_weights=emb_sim_random_weights
            )
        
        return metrics
    
    def _eval_structured_similarity(
        self,
        model,
        device,
        checkpoint_info: CheckpointInfo,
        checkpoint_path: str,
    ) -> Optional[Dict[str, float]]:
        """
        Fixed 100-sample panel: prefer ``structured_similarity_full.json`` (exact indices / groups),
        else :func:`eval_utils.build_structured_similarity_subset` with seed + prefer_manifest.

        Saves ``inputs_*_structured_similarity.npy``, ``embeddings_*_structured_similarity.npy``,
        ``fe_outputs_*_structured_similarity.npy`` using the **same** sample list for embeddings
        and CNN FE outputs (required for comparable heatmaps).
        """
        from sklearn.metrics.pairwise import cosine_similarity
        from eval_utils import (
            check_nova_layout_for_structured_entries,
            check_nova_layout_for_structured_similarity,
            check_single_channel_structured_similarity_viable,
            extract_fe_outputs_from_fairseq_checkpoint,
            load_structured_similarity_entries_from_json,
            load_structured_similarity_spectrograms,
            load_structured_similarity_spectrograms_from_entries,
            remap_structured_entries_to_nova_dir,
            structured_numpy_rows_to_samples,
        )

        nova = self._nova_data_dir
        if not nova or not os.path.isdir(nova):
            print(
                "[!] structured_similarity: provide --nova_data_dir "
                "(parent of single_channel_all, multi_channel, …), e.g. /mnt5/noy/fairseq/data"
            )
            return None

        prefer = self._structured_similarity_prefer_manifest  # "train" matches JSON / label tooling
        seed = self._structured_similarity_seed
        entries_json = self._structured_similarity_entries_json
        allow_fallback = self._allow_structured_single_channel_fallback

        single_channel_only = False
        inputs_arr = None
        panel_entries = None

        if entries_json and os.path.isfile(entries_json):
            raw_entries, json_prefer, json_seed = load_structured_similarity_entries_from_json(entries_json)
            pm = json_prefer if json_prefer is not None else prefer
            panel_entries = remap_structured_entries_to_nova_dir(raw_entries, nova)
            check_nova_layout_for_structured_entries(panel_entries, nova)
            print(
                f"[+] Structured similarity: exact 100-sample panel from {entries_json} "
                f"(prefer_manifest={pm!r}, json seed={json_seed})."
            )
            inputs_arr, _ = load_structured_similarity_spectrograms_from_entries(
                panel_entries, prefer_manifest=pm
            )
        else:
            if entries_json:
                print(f"[!] structured_similarity: entries JSON not found: {entries_json}")
                return None
            try:
                check_nova_layout_for_structured_similarity(nova)
            except FileNotFoundError as exc_full:
                if not allow_fallback:
                    raise FileNotFoundError(
                        f"{exc_full}\n\n"
                        "Use --structured_similarity_allow_single_channel_fallback for a reduced panel, "
                        "or provide --structured_similarity_entries_json pointing to structured_similarity_full.json "
                        "and a --nova_data_dir tree that contains those datasets."
                    ) from exc_full
                print("[!] Full nova_data layout not available.")
                print("[+] Optional fallback: 100 samples from single_channel_all only (10 stacks × 10).")
                try:
                    check_single_channel_structured_similarity_viable(nova)
                except FileNotFoundError as exc_sc:
                    raise FileNotFoundError(
                        f"{exc_full}\n\nFallback (single_channel_all only) also failed:\n{exc_sc}"
                    ) from exc_sc
                single_channel_only = True

            print(
                f"[+] Structured similarity: RNG panel seed={seed}, prefer_manifest={prefer!r}"
                + (" (single_channel_all only)" if single_channel_only else "")
            )
            inputs_arr, panel_entries = load_structured_similarity_spectrograms(
                nova,
                seed=seed,
                prefer_manifest=prefer,
                single_channel_only=single_channel_only,
            )
        samples = structured_numpy_rows_to_samples(inputs_arr)

        _, embeddings = self._extract_embeddings_from_samples(model, device, samples)
        if embeddings is None or len(embeddings) != len(inputs_arr):
            print("[!] structured_similarity: embedding extraction failed or length mismatch")
            return None

        fe_outputs = extract_fe_outputs_from_fairseq_checkpoint(
            checkpoint_path, samples, device, checkpoint_name=checkpoint_info.run_name
        )

        run_name = checkpoint_info.run_name
        np.save(self.data_dir_out / f"inputs_{run_name}_structured_similarity.npy", inputs_arr)
        np.save(self.data_dir_out / f"embeddings_{run_name}_structured_similarity.npy", embeddings)
        if fe_outputs is not None:
            np.save(self.data_dir_out / f"fe_outputs_{run_name}_structured_similarity.npy", fe_outputs)

        emb_sim = cosine_similarity(embeddings)
        triu = np.triu_indices_from(emb_sim, k=1)
        metrics: Dict[str, float] = {
            "strsim_num_samples": float(len(embeddings)),
            "strsim_emb_mean_sim": float(np.mean(emb_sim[triu])),
            "strsim_emb_std_sim": float(np.std(emb_sim[triu])),
        }
        if fe_outputs is not None and len(fe_outputs) >= 2:
            fe_sim = cosine_similarity(fe_outputs)
            ft = np.triu_indices_from(fe_sim, k=1)
            metrics["strsim_fe_mean_sim"] = float(np.mean(fe_sim[ft]))
            metrics["strsim_fe_std_sim"] = float(np.std(fe_sim[ft]))

        print(f"[+] Structured similarity: saved panel arrays for {run_name}")
        return metrics

    def _write_structured_similarity_multi_plot(
        self,
        eval_methods: Optional[List[str]],
        results: List[EvalResult],
        checkpoints: Optional[List[CheckpointInfo]] = None,
    ) -> None:
        """Cross-run figure: Input | models (embed row, FE row). Requires saved structured .npy files."""
        if not eval_methods or "structured_similarity" not in eval_methods:
            return
        from eval_plots import plot_structured_similarity_all_models

        order: List[str] = (
            [c.run_name for c in checkpoints] if checkpoints else [r.run_name for r in results]
        )

        COLORS = ["#3498DB", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12"]
        run_data: List[Dict[str, Any]] = []
        skipped: List[str] = []

        for name in order:
            r = next((x for x in results if x.run_name == name), None)
            if r is None:
                skipped.append(f"{name}: no EvalResult (check run_name / discovery order)")
                continue
            if r.metrics.get("error"):
                skipped.append(f"{name}: checkpoint error — {r.metrics.get('error')}")
                continue
            inp_p = self.data_dir_out / f"inputs_{name}_structured_similarity.npy"
            emb_p = self.data_dir_out / f"embeddings_{name}_structured_similarity.npy"
            fe_p = self.data_dir_out / f"fe_outputs_{name}_structured_similarity.npy"
            if not inp_p.exists() or not emb_p.exists():
                err = r.metrics.get("structured_similarity_error")
                extra = f" ({err[:200]}…)" if err else ""
                skipped.append(f"{name}: missing saved arrays{extra}")
                continue
            run_data.append(
                {
                    "run_name": name,
                    "inputs": np.load(inp_p),
                    "embeddings": np.load(emb_p),
                    "fe_outputs": np.load(fe_p) if fe_p.exists() else None,
                    "color": COLORS[len(run_data) % len(COLORS)],
                }
            )

        if skipped:
            print("[!] structured_similarity multi-plot: skipped run(s):")
            for line in skipped:
                print(f"    - {line}")

        if not run_data:
            print("[!] structured_similarity: no saved panel data for multi-run plot — skipping")
            return

        inputs_ref = run_data[0]["inputs"]
        for rd in run_data[1:]:
            if not np.array_equal(inputs_ref, rd["inputs"]):
                print(
                    f"[!] Warning: inputs differ between {run_data[0]['run_name']} and {rd['run_name']}; "
                    "cosine maps are not directly comparable."
                )

        out_path = self.plots_dir / "all_models_structured_similarity_with_fe.png"
        plot_structured_similarity_all_models(run_data, inputs_ref, str(out_path))
        print(
            f"[+] Saved {out_path} ({len(run_data)} model column(s) + Input; "
            f"{len(order)} checkpoint(s) in evaluation order)"
        )

    def _print_embedding_quality_summary(self, valid_metrics: Dict[str, float],
                                         custom_metrics: Optional[Dict[str, float]] = None,
                                         dataset_trained: str = "N/A",
                                         custom_dataset_name: Optional[str] = None):
        """
        Print a summary comparison of embedding quality metrics across different datasets.
        
        Args:
            valid_metrics: Metrics from evaluating on valid dataset
            custom_metrics: Optional metrics from evaluating on custom dataset
            dataset_trained: Name of the dataset the model was trained on
            custom_dataset_name: Name of the custom evaluation dataset
        """
        print("\n" + "="*80)
        print("EMBEDDING QUALITY SUMMARY")
        print("="*80)
        
        # Determine dataset names for display
        valid_dataset_name = f"{dataset_trained} (valid)" if dataset_trained != 'N/A' else "valid"
        custom_dataset_name_display = custom_dataset_name if custom_dataset_name else None
        
        # Key metrics to compare
        key_metrics = [
            ("Pearson Correlation", "pearson_corr", "Higher is better"),
            ("Spearman Correlation", "spearman_corr", "Higher is better"),
            ("Variance Ratio", "sim_variance_ratio", "Higher is better"),
            ("Embedding Mean Similarity", "emb_mean_sim", "Context dependent"),
            ("Embedding Std Similarity", "emb_std_sim", "Context dependent"),
            ("Input Mean Similarity", "input_mean_sim", "Baseline"),
            ("Input Std Similarity", "input_std_sim", "Baseline"),
            ("Number of Samples", "num_samples", "Info"),
        ]
        
        # Print header
        if custom_metrics is not None:
            print(f"\n{'Metric':<30} {'Valid Dataset':<25} {'Custom Dataset':<25} {'Note':<20}")
            print("-" * 100)
        else:
            print(f"\n{'Metric':<30} {'Value':<25} {'Note':<20}")
            print("-" * 75)
        
        # Print each metric
        for metric_name, metric_key, note in key_metrics:
            valid_value = valid_metrics.get(metric_key, None)
            
            if valid_value is not None:
                if isinstance(valid_value, float):
                    valid_str = f"{valid_value:.4f}"
                else:
                    valid_str = str(valid_value)
            else:
                valid_str = "N/A"
            
            if custom_metrics is not None:
                custom_value = custom_metrics.get(metric_key, None)
                if custom_value is not None:
                    if isinstance(custom_value, float):
                        custom_str = f"{custom_value:.4f}"
                    else:
                        custom_str = str(custom_value)
                else:
                    custom_str = "N/A"
                
                # Add comparison indicator
                if isinstance(valid_value, (int, float)) and isinstance(custom_value, (int, float)):
                    if metric_key in ["pearson_corr", "spearman_corr", "sim_variance_ratio"]:
                        # Higher is better
                        if custom_value > valid_value:
                            indicator = "↑"
                        elif custom_value < valid_value:
                            indicator = "↓"
                        else:
                            indicator = "="
                        custom_str += f" {indicator}"
                    elif metric_key in ["emb_std_sim"]:
                        # Higher variance might be better (less collapsed)
                        if custom_value > valid_value:
                            indicator = "↑"
                        elif custom_value < valid_value:
                            indicator = "↓"
                        else:
                            indicator = "="
                        custom_str += f" {indicator}"
                    else:
                        indicator = ""
                
                print(f"{metric_name:<30} {valid_str:<25} {custom_str:<25} {note:<20}")
            else:
                print(f"{metric_name:<30} {valid_str:<25} {note:<20}")
        
        # Print dataset info
        print("\n" + "-" * 80)
        print(f"Valid Dataset: {valid_dataset_name}")
        if custom_dataset_name_display:
            print(f"Custom Dataset: {custom_dataset_name_display}")
        print("="*80 + "\n")
    
    def _extract_config_info_for_title(self, cfg) -> Dict[str, Any]:
        """
        Extract relevant config information for plot title.
        
        Args:
            cfg: The full config from checkpoint
            
        Returns:
            Dictionary with config info
        """
        from omegaconf import OmegaConf
        
        # Convert to dict if needed
        if hasattr(cfg, '_content'):
            config_dict = OmegaConf.to_container(cfg, resolve=True)
        else:
            config_dict = OmegaConf.to_container(cfg, resolve=True) if hasattr(cfg, 'keys') else {}
        
        info = {}
        
        # Batch size
        info['batch_size'] = config_dict.get('dataset', {}).get('batch_size', 'N/A')
        
        # Learning rate
        lr = config_dict.get('optimization', {}).get('lr', [])
        if isinstance(lr, list) and len(lr) > 0:
            info['lr'] = lr[0] if len(lr) == 1 else lr
        else:
            info['lr'] = lr if lr else 'N/A'
        
        # LR scheduler
        lr_scheduler = config_dict.get('optimization', {}).get('lr_scheduler', {})
        if isinstance(lr_scheduler, dict):
            info['lr_scheduler'] = lr_scheduler.get('_name', 'N/A')
        else:
            info['lr_scheduler'] = str(lr_scheduler) if lr_scheduler else 'N/A'
        
        # Max epochs
        info['max_epoch'] = config_dict.get('optimization', {}).get('max_epoch', 'N/A')
        
        # Dataset trained on
        data_path = config_dict.get('task', {}).get('data', 'N/A')
        if isinstance(data_path, str):
            # Extract dataset name from path
            info['dataset_trained'] = Path(data_path).name if data_path != 'N/A' else 'N/A'
        else:
            info['dataset_trained'] = 'N/A'
        
        return info
    
    def _plot_embedding_similarity_comparison(self, checkpoint_info: CheckpointInfo,
                                               input_sim: np.ndarray,
                                               emb_sim_current: np.ndarray,
                                               emb_sim_frozen: Optional[np.ndarray],
                                               emb_sim_random: Optional[np.ndarray],
                                               best_loss: Optional[float] = None,
                                               best_loss_frozen: Optional[float] = None,
                                               dataset_name: str = "valid",
                                               config_info: Optional[Dict[str, Any]] = None,
                                               emb_sim_random_weights: Optional[np.ndarray] = None):
        """
        Plot 3-way (or 4-way if random weights included) similarity matrix comparison side by side.
        
        Args:
            checkpoint_info: Checkpoint info
            input_sim: Input space similarity matrix
            emb_sim_current: Current model embedding similarity matrix
            emb_sim_frozen: Frozen encoder embedding similarity matrix (optional)
            emb_sim_random: Random init embedding similarity matrix (ignored, kept for compatibility)
            best_loss: Training loss for current model title
            best_loss_frozen: Training loss for frozen encoder model title
            dataset_name: Name of the dataset evaluated on
            config_info: Dictionary with model config info (batch_size, lr, lr_scheduler, max_epoch, dataset_trained)
            emb_sim_random_weights: Random weight model (16k resampled) similarity matrix (optional)
        """
        # Build matrices list
        matrices = [
            ("Input Space", input_sim),
            (f"Trained Model\n(loss: {best_loss:.3f})" if best_loss else "Trained Model", emb_sim_current),
        ]
        if emb_sim_frozen is not None:
            frozen_title = "Frozen Transformer - train only FE"
            if best_loss_frozen is not None:
                frozen_title += f"\n(loss: {best_loss_frozen:.3f})"
            matrices.append((frozen_title, emb_sim_frozen))
        # Note: emb_sim_random is ignored - we removed random_init from comparisons
        if emb_sim_random_weights is not None:
            matrices.append(("Random Weights\n(16k resampled)", emb_sim_random_weights))
        
        n_matrices = len(matrices)
        
        run_plots_dir = self.plots_dir / checkpoint_info.run_name
        run_plots_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, n_matrices, figsize=(5 * n_matrices, 5))
        if n_matrices == 1:
            axes = [axes]
        
        # Build title with config info
        title_parts = ["Embedding Similarity Comparison"]
        
        if config_info:
            # Add model config info
            config_lines = []
            if config_info.get('batch_size') != 'N/A':
                config_lines.append(f"BS={config_info['batch_size']}")
            
            lr = config_info.get('lr', 'N/A')
            if lr != 'N/A':
                if isinstance(lr, list):
                    lr_str = f"[{','.join(map(str, lr))}]"
                else:
                    lr_str = f"{lr:.4f}" if isinstance(lr, (int, float)) else str(lr)
                config_lines.append(f"LR={lr_str}")
            
            lr_sched = config_info.get('lr_scheduler', 'N/A')
            if lr_sched != 'N/A':
                config_lines.append(f"Sched={lr_sched}")
            
            max_epoch = config_info.get('max_epoch', 'N/A')
            if max_epoch != 'N/A' and max_epoch != 0:
                config_lines.append(f"Epochs={max_epoch}")
            
            if config_lines:
                title_parts.append(" | ".join(config_lines))
            
            # Add dataset trained on
            dataset_trained = config_info.get('dataset_trained', 'N/A')
            if dataset_trained != 'N/A':
                title_parts.append(f"Trained: {dataset_trained}")
        
        # Add dataset evaluated on
        if dataset_name == "valid":
            # Use the dataset name from config with "(valid)" suffix
            eval_dataset_name = config_info.get('dataset_trained', 'N/A') if config_info else 'N/A'
            if eval_dataset_name != 'N/A':
                title_parts.append(f"Eval: {eval_dataset_name} (valid)")
            else:
                title_parts.append(f"Eval: {dataset_name}")
        else:
            # Use the provided dataset name (custom dataset)
            title_parts.append(f"Eval: {dataset_name}")
        
        title = "\n".join(title_parts)
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        # Determine number of samples for axis labels
        n_samples = input_sim.shape[0]
        
        # Create sample index labels (show every Nth label to avoid clutter)
        step = max(1, n_samples // 10)  # Show ~10 labels max
        tick_positions = list(range(0, n_samples, step))
        if n_samples - 1 not in tick_positions:
            tick_positions.append(n_samples - 1)
        tick_labels = [str(i) for i in tick_positions]
        
        for i, (title, sim_matrix) in enumerate(matrices):
            # Calculate mean and std for this similarity matrix
            # Exclude diagonal (self-similarity = 1.0) for more meaningful stats
            triu_indices = np.triu_indices_from(sim_matrix, k=1)
            sim_values = sim_matrix[triu_indices]
            sim_mean = np.mean(sim_values)
            sim_std = np.std(sim_values)
            
            # Add mean and std to the title
            title_with_stats = f"{title}\n(mean={sim_mean:.3f}, std={sim_std:.3f})"
            
            sns.heatmap(sim_matrix, ax=axes[i], cmap="viridis",
                       xticklabels=False, yticklabels=False,
                       vmin=0, vmax=1)
            axes[i].set_title(title_with_stats, fontsize=11)
            axes[i].set_xlabel(f"Sample Index (N={n_samples})", fontsize=10)
            
            # Add x-axis tick labels
            axes[i].set_xticks(tick_positions)
            axes[i].set_xticklabels(tick_labels, rotation=0, fontsize=8)
            
            if i == 0:
                axes[i].set_ylabel(f"Sample Index (N={n_samples})", fontsize=10)
                # Add y-axis tick labels
                axes[i].set_yticks(tick_positions)
                axes[i].set_yticklabels(tick_labels, rotation=0, fontsize=8)
        
        plt.tight_layout()
        # Include dataset name in filename
        filename = f"embedding_similarity_comparison_{dataset_name}.png"
        plt.savefig(run_plots_dir / filename, dpi=150)
        plt.close()
        
        n_way = len(matrices)
        print(f"[+] Saved {n_way}-way embedding similarity comparison ({dataset_name}) to {run_plots_dir}")
    
    def load_eval_dataset_fairseq(self, cfg, split="valid", max_samples=None, verbose=True, 
                                   custom_dataset_path=None, eval_data_dir=None, 
                                   sample_indices=None):
        """
        Load evaluation data using fairseq's task.load_dataset and __getitem__.
        This ensures evaluation uses the exact same data loading as training.
        
        Args:
            cfg: The saved config from the checkpoint
            split: Dataset split to load. Supports:
                - "sanity": Load the training dataset (same dataset the model trained on)
                - "valid": Load the validation split from the dataset the model trained on
                          If "valid" doesn't exist, falls back to "train" (same as trained)
                - "train": Alias for "sanity" (for backward compatibility)
            max_samples: Maximum samples to load for efficiency (None = all)
            verbose: Whether to print loading messages
            custom_dataset_path: Optional custom dataset path to override cfg.task.data
            eval_data_dir: Optional evaluation data directory override. Priority: eval_data_dir > self.data_dir > cfg.task.data
            sample_indices: Optional list of specific sample indices to load. If None and max_samples is set, uses random selection with fixed seed.
            
        Returns:
            task: The fairseq task
            samples: List of samples (if max_samples specified), else None
            dataset: The fairseq dataset
        """
        from fairseq import tasks
        
        # Map "sanity" to "train" (the dataset the model was trained on)
        if split == "sanity":
            fairseq_split = "train"
            split_display = "sanity (training dataset)"
        elif split == "valid":
            fairseq_split = "valid"
            split_display = "valid"
        elif split == "train":
            fairseq_split = "train"
            split_display = "train"
        else:
            # Default to valid for unknown splits
            fairseq_split = "valid"
            split_display = f"{split} (mapped to valid)"
            if verbose:
                print(f"[!] Warning: Unknown split '{split}', using 'valid' instead")
        
        # Build task from config
        task = tasks.setup_task(cfg.task)
        
        # Determine data path with priority: eval_data_dir > self.data_dir > cfg.task.data
        # Only override if eval_data_dir is provided or self.data_dir is not the default
        task_cfg = cfg.task
        data_path_override = None
        
        if eval_data_dir is not None:
            data_path_override = eval_data_dir
        elif self.data_dir and self.data_dir != "/mnt5/noy/fairseq/data/single_channel_1m/":
            data_path_override = self.data_dir
        
        # Override data path if custom_dataset_path is provided (highest priority for custom dataset)
        if custom_dataset_path is not None:
            # Create a copy of the task config with the custom data path
            from omegaconf import OmegaConf
            task_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task))
            task_cfg.data = custom_dataset_path
            if verbose:
                print(f"[+] Using custom dataset path: {custom_dataset_path}")
        elif data_path_override is not None:
            # Create a copy of the task config with the eval data path
            from omegaconf import OmegaConf
            task_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task))
            task_cfg.data = data_path_override
            if verbose:
                print(f"[+] Using evaluation data path: {data_path_override}")
        
        # Try to load dataset, fallback to "train" if "valid" doesn't exist
        try:
            task.load_dataset(fairseq_split, task_cfg=task_cfg)
            dataset = task.dataset(fairseq_split)
            
            # Check if dataset is empty or doesn't exist
            if len(dataset) == 0 and fairseq_split == "valid":
                if verbose:
                    print(f"[!] Warning: 'valid' split is empty, falling back to 'train' (same as trained dataset)")
                fairseq_split = "train"
                split_display = "train (valid not available)"
                task.load_dataset(fairseq_split, task_cfg=task_cfg)
                dataset = task.dataset(fairseq_split)
        except (KeyError, AttributeError) as e:
            # If "valid" split doesn't exist, fallback to "train"
            if fairseq_split == "valid":
                if verbose:
                    print(f"[!] Warning: 'valid' split not found ({e}), falling back to 'train' (same as trained dataset)")
                fairseq_split = "train"
                split_display = "train (valid not available)"
                task.load_dataset(fairseq_split, task_cfg=task_cfg)
                dataset = task.dataset(fairseq_split)
            else:
                raise
        
        if verbose:
            dataset_source = f"custom dataset ({custom_dataset_path})" if custom_dataset_path else split_display
            if data_path_override and not custom_dataset_path:
                dataset_source = f"{split_display} (from {data_path_override})"
            print(f"[+] Loaded {len(dataset)} samples from {dataset_source} dataset")
        
        # If max_samples specified, load individual samples
        samples = None
        if max_samples is not None or sample_indices is not None:
            if sample_indices is not None:
                # Use provided sample indices
                indices = sample_indices
            else:
                # Use random selection with fixed seed for reproducibility
                random.seed(42)
                dataset_size = len(dataset)
                sample_size = min(max_samples, dataset_size)
                indices = random.sample(range(dataset_size), sample_size)
                indices.sort()  # Sort for easier debugging
            
            samples = []
            for idx in tqdm(indices, desc=f"Loading {split_display} data"):
                # This calls FileAudioDataset.__getitem__()
                sample = dataset[idx]
                samples.append(sample)
        
        return task, samples, dataset
    
    def _eval_validation_loss(self, model, cfg, split: str = "valid", 
                             eval_data_dir: Optional[str] = None, 
                             debug: bool = False,
                             checkpoint_path: Optional[str] = None,
                             mask_memory_path: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate the model using trainer.valid_step (same as train.py's validate function).
        
        This mimics the exact validation done during training:
        - Uses trainer.get_valid_iterator() for batching
        - Uses trainer.valid_step(sample) which handles GPU, metrics, etc.
        
        Args:
            model: The model
            cfg: The config
            split: "sanity" (train split) or "valid" (valid split or eval_data_dir)
            eval_data_dir: Optional override for eval dataset path (only used when split="valid")
            debug: Debug flag for verbose output (default: False)
            checkpoint_path: Optional path to checkpoint file (used to resolve mask_memory_save_path)
            mask_memory_path: Optional path to mask memory file. If provided, loads masks from this file.
                            If None, tries to load from cfg.model.mask_memory_save_path (for backward compatibility).
        
        Returns:
            Dict with validation loss metrics
        """
        from fairseq.trainer import Trainer
        from fairseq.logging import metrics as fairseq_metrics
        
        # Map split to fairseq split
        if split == "sanity":
            fairseq_split = "train"
            split_display = "sanity (training dataset)"
        else:  # "valid"
            fairseq_split = "valid"
            split_display = "valid"
        
        print(f"[+] Evaluating validation loss on {split_display}...")
        
        # Load mask memory if path is provided (for fixed masking experiments)
        if mask_memory_path is not None and hasattr(model, 'load_mask_memory'):
            mask_memory_path_str = str(mask_memory_path)
            if os.path.exists(mask_memory_path_str):
                # Temporarily patch torch.load to use weights_only=False for numpy arrays (PyTorch 2.6+)
                import torch.serialization
                torch_module = sys.modules['torch']
                original_torch_load = torch_module.load
                def patched_torch_load(*args, **kwargs):
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return original_torch_load(*args, **kwargs)
                torch_module.load = patched_torch_load
                try:
                    success = model.load_mask_memory(mask_memory_path_str)
                finally:
                    torch_module.load = original_torch_load
                
                if success:
                    model.enable_mask_memory()
                    num_masks = len(model._mask_memory) if hasattr(model, '_mask_memory') else 0
                    print(f"[+] Loaded mask memory from {mask_memory_path_str} ({num_masks} masks)")
                else:
                    print(f"[!] Failed to load mask memory from {mask_memory_path_str}")
            else:
                print(f"[!] Mask memory file not found: {mask_memory_path_str}")
                print(f"[!] Will use seed-based masking instead")
        # Backward compatibility: try loading from cfg if mask_memory_path not provided
        elif split == "sanity" and hasattr(model, 'load_mask_memory') and hasattr(cfg.model, 'mask_memory_save_path'):
            mask_memory_path = cfg.model.mask_memory_save_path
            if mask_memory_path:
                mask_memory_path_str = str(mask_memory_path)
                # Try to find mask memory file relative to checkpoint directory
                if checkpoint_path:
                    checkpoint_dir = Path(checkpoint_path).parent
                    # Try absolute path first, then relative to checkpoint dir
                    if not os.path.exists(mask_memory_path_str):
                        mask_memory_path_str = str(checkpoint_dir.parent / Path(mask_memory_path_str).name)
                
                if os.path.exists(mask_memory_path_str):
                    # Temporarily patch torch.load to use weights_only=False for numpy arrays (PyTorch 2.6+)
                    import torch.serialization
                    torch_module = sys.modules['torch']
                    original_torch_load = torch_module.load
                    def patched_torch_load(*args, **kwargs):
                        if 'weights_only' not in kwargs:
                            kwargs['weights_only'] = False
                        return original_torch_load(*args, **kwargs)
                    torch_module.load = patched_torch_load
                    try:
                        success = model.load_mask_memory(mask_memory_path_str)
                    finally:
                        torch_module.load = original_torch_load
                    
                    if success:
                        model.enable_mask_memory()
                        num_masks = len(model._mask_memory) if hasattr(model, '_mask_memory') else 0
                        print(f"[+] Loaded mask memory for sanity validation from {mask_memory_path_str} ({num_masks} masks)")
                    else:
                        print(f"[!] Failed to load mask memory from {mask_memory_path_str}")
                else:
                    print(f"[!] Mask memory file not found: {mask_memory_path_str}")
                    print(f"[!] Will use seed-based masking instead")
        
        try:
            # Load dataset - use eval_data_dir only for "valid" split
            task, _, dataset = self.load_eval_dataset_fairseq(
                cfg, split=fairseq_split, max_samples=None, verbose=False,
                eval_data_dir=eval_data_dir if split == "valid" else None
            )
            
            # Use the fairseq_split that was requested (already loaded by load_eval_dataset_fairseq)
            # load_eval_dataset_fairseq handles fallback internally, so we use fairseq_split directly
            validation_split = fairseq_split
            
            criterion = task.build_criterion(cfg.criterion) # fixme do I need this if calling already to criterion? 

            # Ensure model is on the correct device
            use_cuda = torch.cuda.is_available() and not getattr(cfg.common, 'cpu', False)
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
            
            # Create trainer and get iterator
            # Use the split that was loaded
            trainer = Trainer(cfg, task, model, criterion)
            # Try the requested split, fallback to train if it fails (only for "valid" split)
            try:
                itr = trainer.get_valid_iterator(validation_split).next_epoch_itr(shuffle=False)
            except (KeyError, AttributeError) as e:
                if validation_split == "valid":
                    # Fallback to train if valid fails
                    print(f"[!] Warning: Could not get 'valid' iterator, using 'train': {e}")
                    validation_split = "train"
                    itr = trainer.get_valid_iterator(validation_split).next_epoch_itr(shuffle=False)
                else:
                    raise
            
            all_losses = []
            num_batches = 0
            
            # Validation loop - follows fairseq's pattern
            with fairseq_metrics.aggregate(new_root=True) as agg:
                for sample in tqdm(itr, desc="Computing validation loss"):
                    # Call trainer.valid_step() (matches fairseq's validate.py flow)
                    # This sets model.eval() internally and processes the sample via _prepare_sample()
                    log_output = trainer.valid_step(sample)
                    
                    if log_output:
                        loss_val = log_output.get("loss", 0)
                        all_losses.append(float(loss_val))
                        num_batches += 1
                
                # Get aggregated stats
                stats = agg.get_smoothed_values()
            
            # Build result metrics
            result_metrics = {
                "eval_loss": float(stats.get("loss", np.mean(all_losses) if all_losses else 0)),
                "eval_loss_std": float(np.std(all_losses)) if all_losses else 0.0,
                "eval_loss_min": float(min(all_losses)) if all_losses else 0.0,
                "eval_loss_max": float(max(all_losses)) if all_losses else 0.0,
                "eval_num_batches": num_batches,
            }
            
            # Add other metrics from aggregation
            for key, val in stats.items():
                if key not in ["loss"]:
                    result_metrics[f"eval_{key}"] = float(val) if isinstance(val, (int, float)) else val
            
            print(f"\n[+] Validation Loss Results:")
            print(f"    Loss:     {result_metrics['eval_loss']:.6f}")
            print(f"    Std:      {result_metrics['eval_loss_std']:.6f}")
            print(f"    Batches:  {result_metrics['eval_num_batches']}")
            
            return result_metrics
            
        except Exception as e:
            print(f"\n[!] ========== ERROR in validation loss evaluation ==========")
            print(f"[!] Error type: {type(e).__name__}")
            print(f"[!] Error message: {str(e)}")
            import traceback
            error_traceback = traceback.format_exc()
            print(f"[!] Full traceback:\n{error_traceback}")
            print(f"[!] ============================================================\n")
            return {"validation_loss_error": str(e), "validation_loss_error_traceback": error_traceback}
    
    def _extract_trained_model_embeddings(self, model, samples: List[Dict], device) -> Optional[Tuple[np.ndarray, np.ndarray, List[Dict]]]:
        """
        Extract embeddings from the trained model.
        
        Follows the exact same pattern as _load_frozen_encoder_embeddings and _load_random_init_embeddings:
        - Receives pre-loaded samples from fairseq dataset
        - Extract embeddings using _extract_embeddings_from_samples
        
        Args:
            model: The trained model
            samples: Pre-loaded samples from fairseq dataset
            device: The device to use
            
        Returns:
            Tuple of (inputs, embeddings, samples) or None if extraction failed
        """
        if samples is None:
            print("[!] Warning: No samples provided for trained model embeddings")
            return None
        
        try:
            print(f"[+] Extracting embeddings from trained model ({len(samples)} samples)...")
            
            # Ensure model is in eval mode
            model.eval()
            
            # Extract embeddings using the shared function (same as frozen/random)
            inputs_arr, embeddings_arr = self._extract_embeddings_from_samples(
                model, device, samples, handle_batches=True
            )
            
            if len(inputs_arr) == 0 or len(embeddings_arr) == 0:
                print(f"[!] Warning: Extracted empty arrays! inputs={len(inputs_arr)}, embeddings={len(embeddings_arr)}")
                return None
            
            print(f"[+] Extracted embeddings from {len(inputs_arr)} samples")
            
            return (inputs_arr, embeddings_arr, samples)
            
        except Exception as e:
            print(f"[!] Failed to extract trained model embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _eval_signal_completion(self, model, device, samples: List[Dict], 
                                 fixed_mask_start: Optional[int] = None,
                                 fixed_mask_end: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate signal completion/reconstruction capability.
        
        Tests how well the model can predict masked portions of signals by:
        1. Getting embeddings for full (unmasked) signals as ground truth
        2. Masking portions of signals (second half for causal, random spans, or fixed range)
        3. Comparing model's contextualized representations at masked positions
        
        Args:
            model: The model (already in eval mode)
            device: The device to use
            samples: Pre-loaded samples from fairseq dataset
            fixed_mask_start: If provided, use fixed mask starting at this index (for sanity testing)
            fixed_mask_end: If provided, use fixed mask ending at this index (for sanity testing)
        
        Key metrics:
        - Cosine similarity between predicted and target embeddings at masked positions
        - MSE loss at masked positions
        - Comparison across different masking strategies (causal vs random vs fixed)
        """
        from scipy.stats import pearsonr, spearmanr
        
        sample_size = len(samples)
        print(f"[+] Evaluating signal completion on {sample_size} samples...")
        
        # Results storage
        completion_results = []
        
        # Check if using fixed mask
        use_fixed_mask = fixed_mask_start is not None and fixed_mask_end is not None
        if use_fixed_mask:
            print(f"[+] Using fixed mask: indexes {fixed_mask_start}-{fixed_mask_end}")
            # Import fixed mask function
            from test_utils import create_fixed_mask_indices
        else:
            # Get masking parameters from model config (same as training)
            mask_prob = getattr(model.cfg, 'mask_prob', 0.65)
            mask_length = getattr(model.cfg, 'mask_length', 10)
            mask_selection = getattr(model.cfg, 'mask_selection', 'static')
            mask_other = getattr(model.cfg, 'mask_other', 0.0)
            no_mask_overlap = getattr(model.cfg, 'no_mask_overlap', False)
            mask_min_space = getattr(model.cfg, 'mask_min_space', 1)
            
            print(f"[+] Using training masking config: mask_prob={mask_prob}, mask_length={mask_length}, "
                  f"mask_selection={mask_selection}")
            
            # Import fairseq's compute_mask_indices to use same masking as training
            from fairseq.data.data_utils import compute_mask_indices
        
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(samples, desc="Signal completion eval")):
                try:
                    # Get source from fairseq-loaded sample (already preprocessed)
                    source = sample["source"]
                    sample_id = sample.get("id", idx)
                    
                    # Prepare input: [batch, seq_len]
                    data = source.to(device)
                    if data.dim() == 1:
                        data = data.unsqueeze(0)  # [1, seq_len]
                    
                    # Get ground truth embeddings (unmasked)
                    gt_result = model.extract_features(data, padding_mask=None, mask=False)
                    gt_embeddings = gt_result["x"]  # [1, T, embed_dim]
                    
                    T = gt_embeddings.shape[1]  # Sequence length after feature extraction
                    
                    # Create mask indices
                    if use_fixed_mask:
                        # Use fixed mask for sanity testing
                        mask_indices_np = create_fixed_mask_indices(
                            1, T, fixed_mask_start, fixed_mask_end
                        )
                    else:
                        # Use same masking as training (fairseq's compute_mask_indices)
                        mask_indices_np = compute_mask_indices(
                            (1, T),  # (batch_size, sequence_length)
                            None,  # padding_mask
                            mask_prob,
                            mask_length,
                            mask_selection,
                            mask_other,
                            min_masks=1,
                            no_overlap=no_mask_overlap,
                            min_space=mask_min_space,
                        )
                    mask_indices = torch.from_numpy(mask_indices_np).to(device)  # [1, T]
                    
                    # Get masked embeddings using the same masking as training
                    masked_result = model.forward(
                        data, 
                        padding_mask=None, 
                        mask=True,
                        features_only=True,
                        mask_indices=mask_indices_np
                    )
                    masked_embeddings = masked_result["x"]  # [1, T, embed_dim]
                    
                    # Compute metrics at masked positions
                    gt_masked = gt_embeddings[mask_indices]  # [num_masked, embed_dim]
                    pred_masked = masked_embeddings[mask_indices]  # [num_masked, embed_dim]
                    
                    sample_result = {
                        "index": idx,
                        "sample_id": int(sample_id),
                        "seq_len": T,
                        "num_masked": mask_indices.sum().item(),
                    }
                    
                    if gt_masked.numel() > 0:
                        # Cosine similarity at masked positions
                        cos_sim = torch.nn.functional.cosine_similarity(
                            gt_masked, pred_masked, dim=-1
                        ).mean().item()
                        
                        # MSE at masked positions
                        mse = torch.nn.functional.mse_loss(
                            pred_masked, gt_masked
                        ).item()
                        
                        # L1 distance
                        l1 = torch.nn.functional.l1_loss(
                            pred_masked, gt_masked
                        ).item()
                        
                        sample_result["cos_sim"] = cos_sim
                        sample_result["mse"] = mse
                        sample_result["l1"] = l1
                    
                    # Plot spectrograms for first few samples
                    if idx < 5:
                        try:
                            # Get 245-length data (source is the raw audio/spectrogram)
                            source_np = source.cpu().numpy() if isinstance(source, torch.Tensor) else source
                            
                            # Map mask from feature level (T) to input level (245)
                            # The mask is at feature level, but we want to visualize at input level
                            # For simplicity, we'll create a mask at input level using the same pattern
                            # by computing mask for input length
                            input_len = len(source_np) if source_np.ndim == 1 else source_np.shape[-1]
                            if input_len == 245:
                                # Compute mask for 245-length input
                                mask_indices_input_np = compute_mask_indices(
                                    (1, input_len),
                                    None,
                                    mask_prob,
                                    mask_length,
                                    mask_selection,
                                    mask_other,
                                    min_masks=1,
                                    no_overlap=no_mask_overlap,
                                    min_space=mask_min_space,
                                )
                                mask_indices_input = mask_indices_input_np[0]  # [245]
                                
                                self._plot_spectrogram_with_mask(
                                    source_np if source_np.ndim == 1 else source_np.squeeze(),
                                    mask_indices_input,
                                    self._current_run_name, int(sample_id),
                                    mask_prob, mask_length,
                                    save=True
                                )
                        except Exception as e:
                            print(f"[!] Warning: Could not plot spectrogram for sample {idx}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    completion_results.append(sample_result)
                    
                except Exception as e:
                    print(f"[!] Error processing sample {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if not completion_results:
            return {"error": "No samples processed successfully"}
        
        # Create DataFrame and save
        completion_df = pd.DataFrame(completion_results)
        completion_df_path = self.data_dir_out / f"signal_completion_{self._current_run_name}.csv"
        completion_df.to_csv(completion_df_path, index=False)
        print(f"[+] Saved signal completion data to: {completion_df_path}")
        
        # Store for visualization
        self.eval_data[f"completion_df_{self._current_run_name}"] = completion_df
        
        # Aggregate metrics (using training masking config)
        metrics = {}
        if "cos_sim" in completion_df.columns:
            cos_sims = completion_df["cos_sim"].dropna().values
            mse_vals = completion_df["mse"].dropna().values
            l1_vals = completion_df["l1"].dropna().values
            
            if len(cos_sims) > 0:
                metrics["completion_cos_sim_mean"] = float(np.mean(cos_sims))
                metrics["completion_cos_sim_std"] = float(np.std(cos_sims))
                metrics["completion_mse_mean"] = float(np.mean(mse_vals))
                metrics["completion_mse_std"] = float(np.std(mse_vals))
                metrics["completion_l1_mean"] = float(np.mean(l1_vals))
                metrics["completion_l1_std"] = float(np.std(l1_vals))
                
                # Overall completion score (cos sim ranges from -1 to 1)
                metrics["completion_score"] = float((np.mean(cos_sims) + 1) * 50)
        
        print(f"\n[+] Signal Completion Analysis (mask_prob={mask_prob}, mask_length={mask_length}):")
        if "completion_cos_sim_mean" in metrics:
            print(f"    Cosine Similarity: {metrics['completion_cos_sim_mean']:.4f} ± {metrics['completion_cos_sim_std']:.4f}")
            print(f"    MSE: {metrics['completion_mse_mean']:.4f} ± {metrics['completion_mse_std']:.4f}")
            print(f"    L1: {metrics['completion_l1_mean']:.4f} ± {metrics['completion_l1_std']:.4f}")
        if "completion_score" in metrics:
            print(f"    Overall completion score: {metrics['completion_score']:.1f}/100")
        
        return metrics
    
    def _eval_noise_robustness(self, model, device, samples: List[Dict]) -> Dict[str, float]:
        """
        Evaluate embedding robustness to various noise types.
        Compares clean vs noisy embeddings using cosine similarity.
        
        Args:
            model: The model (already in eval mode)
            device: The device to use
            samples: Pre-loaded samples from fairseq dataset
        """
        print(f"[+] Evaluating noise robustness on {len(samples)} samples...")
        
        noise_types = {
            "gaussian_std": lambda x: x + np.random.normal(0, 0.01, size=x.shape),
            "gaussian_mean": lambda x: x + np.random.normal(0.02, 0.001, size=x.shape),
            "gain_low": lambda x: x * np.random.normal(1, 0.05),
            "gain_high": lambda x: x * np.random.normal(1, 0.1),
        }
        
        noise_results = []
        
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(samples, desc="Noise robustness eval")):
                try:
                    source = sample["source"]
                    sample_id = sample.get("id", idx)
                    
                    clean_data = source.cpu().numpy()
                    
                    # Prepare clean input
                    data_tensor = source.unsqueeze(0).to(device) if source.dim() == 1 else source.to(device)
                    
                    # Get clean embedding
                    clean_result = model.extract_features(data_tensor, padding_mask=None, mask=False)
                    clean_emb = clean_result["x"].mean(dim=1).cpu().numpy().squeeze()
                    
                    sample_result = {
                        "index": idx,
                        "sample_id": int(sample_id),
                    }
                    
                    # Evaluate each noise type
                    for noise_type, noise_fn in noise_types.items():
                        noisy_data = noise_fn(clean_data).astype(np.float32)
                        noisy_tensor = torch.tensor(noisy_data).unsqueeze(0).to(device)
                        
                        noisy_result = model.extract_features(noisy_tensor, padding_mask=None, mask=False)
                        noisy_emb = noisy_result["x"].mean(dim=1).cpu().numpy().squeeze()
                        
                        data_sim = compute_cosine_similarity(clean_data, noisy_data)
                        emb_sim = compute_cosine_similarity(clean_emb, noisy_emb)
                        
                        sample_result[f"{noise_type}_data_sim"] = data_sim
                        sample_result[f"{noise_type}_emb_sim"] = emb_sim
                        
                        # Store data for visualization (first few samples only)
                        if idx < 10:
                            self.eval_data[f"clean_data_{self._current_run_name}_{idx}"] = clean_data
                            self.eval_data[f"noisy_data_{self._current_run_name}_{idx}_{noise_type}"] = noisy_data
                    
                    noise_results.append(sample_result)
                        
                except Exception as e:
                    print(f"[!] Error processing sample {idx}: {e}")
                    continue
        
        # Create and save results dataframe
        noise_df = pd.DataFrame(noise_results)
        noise_df_path = self.data_dir_out / f"noise_robustness_{self._current_run_name}.csv"
        noise_df.to_csv(noise_df_path, index=False)
        
        self.eval_data[f"noise_df_{self._current_run_name}"] = noise_df
        
        # Aggregate metrics
        metrics = {}
        for noise_type in noise_types:
            data_sims = noise_df[f"{noise_type}_data_sim"].values
            emb_sims = noise_df[f"{noise_type}_emb_sim"].values
            
            if len(data_sims) > 0:
                metrics[f"noise_{noise_type}_data_sim_mean"] = float(np.mean(data_sims))
                metrics[f"noise_{noise_type}_data_sim_std"] = float(np.std(data_sims))
                metrics[f"noise_{noise_type}_emb_sim_mean"] = float(np.mean(emb_sims))
                metrics[f"noise_{noise_type}_emb_sim_std"] = float(np.std(emb_sims))
                metrics[f"noise_{noise_type}_robustness_ratio"] = float(
                    np.mean(emb_sims) / (np.mean(data_sims) + 1e-8)
                )
        
        print(f"[+] Noise Robustness Analysis:")
        for noise_type in noise_types:
            if f"noise_{noise_type}_emb_sim_mean" in metrics:
                print(f"    {noise_type}: data_sim={metrics[f'noise_{noise_type}_data_sim_mean']:.4f}, "
                      f"emb_sim={metrics[f'noise_{noise_type}_emb_sim_mean']:.4f}, "
                      f"ratio={metrics[f'noise_{noise_type}_robustness_ratio']:.4f}")
        
        return metrics

    def _eval_stack_similarity(self, inputs: np.ndarray, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Evaluate similarity comparison by stack membership.
        Compares how well embedding-space neighbors match input-space neighbors,
        particularly for samples from the same "stack" (group).
        
        Args:
            inputs: Pre-extracted input data [N, seq_len]
            embeddings: Pre-extracted embeddings [N, embed_dim]
        """
        print(f"[+] Evaluating stack similarity...")
        
        if len(embeddings) < 10:
            return {"error": "Not enough valid samples for stack similarity"}
        
        # Assign stack indices (every 10 samples = 1 stack)
        stack_indices = np.array([idx // 10 for idx in range(len(embeddings))])
        
        # Compare embedding vs input space similarity
        return self._compare_similarity_by_stack_membership(
            inputs, embeddings, stack_indices, k=5, run_name=self._current_run_name
        )
    
    def _compare_similarity_by_stack_membership(
        self, inputs: np.ndarray, embeddings: np.ndarray, 
        stack_indices: np.ndarray, k: int = 5, run_name: str = ""
    ) -> Dict[str, float]:
        """
        For each sample, compare top-k neighbors in input vs embedding space.
        Measures how well the model preserves stack membership in embedding space.
        Saves detailed match_df to data directory.
        """
        n_samples = len(inputs)
        input_sim_matrix = compute_cosine_similarity_matrix(inputs)
        emb_sim_matrix = compute_cosine_similarity_matrix(embeddings)
        
        # Detailed results for match_df
        match_results = []
        
        for idx in range(n_samples):
            query_stack = stack_indices[idx]
            
            # Get top-k neighbors (excluding self)
            input_sims = input_sim_matrix[idx].copy()
            input_sims[idx] = -np.inf
            topk_input_idx = input_sims.argsort()[-k:][::-1]
            
            emb_sims = emb_sim_matrix[idx].copy()
            emb_sims[idx] = -np.inf
            topk_emb_idx = emb_sims.argsort()[-k:][::-1]
            
            # Get same-stack matches
            input_stack_matches = [int(i) for i in topk_input_idx if stack_indices[i] == query_stack]
            emb_stack_matches = [int(i) for i in topk_emb_idx if stack_indices[i] == query_stack]
            
            match_diff = len(emb_stack_matches) - len(input_stack_matches)
            match_score = ((match_diff + k) / (2 * k)) * 100
            
            match_results.append({
                "index": idx,
                "stack_idx": int(query_stack),
                "embedding_neighbors": topk_emb_idx.tolist(),
                "embedding_similarities": emb_sims[topk_emb_idx].tolist(),
                "embedding_stack_matches": emb_stack_matches,
                "input_neighbors": topk_input_idx.tolist(),
                "input_similarities": input_sims[topk_input_idx].tolist(),
                "input_stack_matches": input_stack_matches,
                "match_diff": match_diff,
                "match_score": match_score,
            })
        
        # Create match_df and save
        match_df = pd.DataFrame(match_results)
        
        # Add summary row
        avg_score = match_df["match_score"].mean()
        
        # Save match_df to data directory
        match_df_path = self.data_dir_out / f"match_df_{run_name}.csv"
        match_df.to_csv(match_df_path, index=False)
        print(f"[+] Saved match_df to: {match_df_path}")
        
        # Store for visualization
        self.eval_data[f"match_df_{run_name}"] = match_df
        self.eval_data[f"inputs_{run_name}"] = inputs
        self.eval_data[f"embeddings_{run_name}"] = embeddings
        self.eval_data[f"stack_indices_{run_name}"] = stack_indices
        
        # Compute aggregate metrics
        input_match_counts = [len(r["input_stack_matches"]) for r in match_results]
        emb_match_counts = [len(r["embedding_stack_matches"]) for r in match_results]
        match_diffs = [r["match_diff"] for r in match_results]
        
        metrics = {
            "stack_input_match_mean": float(np.mean(input_match_counts)),
            "stack_input_match_std": float(np.std(input_match_counts)),
            "stack_emb_match_mean": float(np.mean(emb_match_counts)),
            "stack_emb_match_std": float(np.std(emb_match_counts)),
            "stack_match_diff_mean": float(np.mean(match_diffs)),
            "stack_match_improvement_pct": float(
                100 * sum(1 for d in match_diffs if d > 0) / len(match_diffs)
            ),
            "stack_match_score_mean": float(avg_score),
            "stack_num_samples": n_samples,
            "stack_num_stacks": len(np.unique(stack_indices)),
        }
        
        print(f"\n[+] Stack Similarity Analysis:")
        print(f"    Input-space same-stack matches: {metrics['stack_input_match_mean']:.2f} ± {metrics['stack_input_match_std']:.2f}")
        print(f"    Embed-space same-stack matches: {metrics['stack_emb_match_mean']:.2f} ± {metrics['stack_emb_match_std']:.2f}")
        print(f"    Match improvement: {metrics['stack_match_improvement_pct']:.1f}% of samples improved")
        print(f"    Match score (0-100): {metrics['stack_match_score_mean']:.1f}")
        
        return metrics
    
    def _add_noisy_data(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Add various types of noise to input data.
        Returns dict of noisy versions.
        """
        noise_data = {
            "gaussian_std": data + np.random.normal(0, 0.01, size=len(data)),
            "gaussian_mean": data + np.random.normal(0.02, 0.001, size=len(data)),
            "shot_low": np.random.poisson((data - data.min() + 1e-4) * 0.1) / 0.1 - 1e-4,
            "shot_high": np.random.poisson((data - data.min() + 1e-4) * 0.05) / 0.05 - 1e-4,
            "gain_low": data * np.random.normal(1, 0.05),
            "gain_high": data * np.random.normal(1, 0.1),
        }
        return noise_data
    
    def plot_similarity_matrices(self, checkpoint_info: CheckpointInfo,
                                  inputs: np.ndarray = None,
                                  embeddings: np.ndarray = None,
                                  best_loss: Optional[float] = None,
                                  save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate and optionally save similarity matrix visualizations.
        Uses the same cosine similarity calculation as _eval_embedding_similarity.
        
        Args:
            checkpoint_info: Checkpoint info
            inputs: Pre-extracted inputs (optional - will extract if not provided)
            embeddings: Pre-extracted embeddings (optional - will extract if not provided)
            best_loss: Training loss for title
            save_plots: Whether to save plots
            
        Note:
            Number of samples in cosine matrix is determined by max_samples parameter
            in _eval_validation_loss() (default: 100). The matrix will be [N x N] where
            N is the number of samples extracted during validation.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # If embeddings not provided, extract them
        if inputs is None or embeddings is None:
            if load_fairseq_checkpoint is None:
                model, model_cfg, checkpoint_info_loaded = _load_fairseq_checkpoint_fallback(checkpoint_info.path)
            else:
                model, model_cfg, checkpoint_info_loaded = load_fairseq_checkpoint(checkpoint_info.path)
            cfg = checkpoint_info_loaded["cfg"]  # Full config from checkpoint
            self._current_cfg = cfg
            model, device = self._prepare_model_for_eval(model, cfg)
            inputs, embeddings, _ = self._extract_embeddings(model, device, max_samples=100)
        
        if len(embeddings) < 2:
            return {}
        
        # Use the same calculation as _eval_embedding_similarity
        # This ensures consistency between the metrics and the plots
        input_sim_matrix = cosine_similarity(inputs)
        emb_sim_matrix = cosine_similarity(embeddings)
        
        # Print statistics (matching _eval_embedding_similarity format)
        triu_idx = np.triu_indices_from(input_sim_matrix, k=1)
        input_sims = input_sim_matrix[triu_idx]
        emb_sims = emb_sim_matrix[triu_idx]
        
        print(f"[+] Similarity Matrices Plot:")
        print(f"    Input similarity: shape={input_sim_matrix.shape}, mean={np.mean(input_sims):.4f}, std={np.std(input_sims):.4f}")
        print(f"    Embedding similarity: shape={emb_sim_matrix.shape}, mean={np.mean(emb_sims):.4f}, std={np.std(emb_sims):.4f}")
        print(f"    Number of samples: {len(embeddings)} (matrix size: {emb_sim_matrix.shape[0]}x{emb_sim_matrix.shape[1]})")
        
        loss_str = f" | Training Loss: {best_loss:.4f}" if best_loss is not None else ""
        
        if save_plots:
            run_plots_dir = self.plots_dir / checkpoint_info.run_name
            run_plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot input similarity matrix
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"Run: {checkpoint_info.run_name}{loss_str}", fontsize=12, fontweight='bold')
            
            sns.heatmap(input_sim_matrix, ax=axes[0], cmap="viridis", 
                       xticklabels=False, yticklabels=False, vmin=0, vmax=1)
            axes[0].set_title(f"Input Space Cosine Similarity (N={len(embeddings)})")
            axes[0].set_xlabel("Sample Index")
            axes[0].set_ylabel("Sample Index")
            
            sns.heatmap(emb_sim_matrix, ax=axes[1], cmap="viridis",
                       xticklabels=False, yticklabels=False, vmin=0, vmax=1)
            axes[1].set_title(f"Embedding Space Cosine Similarity (N={len(embeddings)})")
            axes[1].set_xlabel("Sample Index")
            axes[1].set_ylabel("Sample Index")
            
            plt.tight_layout()
            plt.savefig(run_plots_dir / "similarity_matrices.png", dpi=150)
            plt.close()
            
            print(f"[+] Saved similarity matrix plot to {run_plots_dir}")
        
        return {
            "input_sim_matrix": input_sim_matrix,
            "emb_sim_matrix": emb_sim_matrix,
            "num_samples": len(embeddings),
        }
    
    def plot_noise_robustness_comparison(self, results: List[EvalResult], 
                                         save_plots: bool = True):
        """
        Plot noise robustness comparison across runs.
        """
        if not results:
            return
        
        noise_types = ["gaussian_std", "gaussian_mean", "gain_low", "gain_high"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Create x-axis labels with loss info
        run_labels = []
        for r in results:
            best_loss = r.metrics.get("best_loss")
            if best_loss is not None:
                run_labels.append(f"{r.run_name}\n(loss: {best_loss:.3f})")
            else:
                run_labels.append(r.run_name)
        
        # Plot embedding similarity under noise
        emb_sims = {nt: [] for nt in noise_types}
        for r in results:
            for nt in noise_types:
                key = f"noise_{nt}_emb_sim_mean"
                emb_sims[nt].append(r.metrics.get(key, 0))
        
        x = np.arange(len(run_labels))
        width = 0.2
        for i, nt in enumerate(noise_types):
            axes[0].bar(x + i * width, emb_sims[nt], width, label=nt)
        
        axes[0].set_xlabel("Run")
        axes[0].set_ylabel("Embedding Similarity (higher = more robust)")
        axes[0].set_title("Noise Robustness: Embedding Similarity")
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(run_labels, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot robustness ratio
        ratios = {nt: [] for nt in noise_types}
        for r in results:
            for nt in noise_types:
                key = f"noise_{nt}_robustness_ratio"
                ratios[nt].append(r.metrics.get(key, 0))
        
        for i, nt in enumerate(noise_types):
            axes[1].bar(x + i * width, ratios[nt], width, label=nt)
        
        axes[1].axhline(y=1.0, color='r', linestyle='--', label='Baseline (ratio=1)')
        axes[1].set_xlabel("Run")
        axes[1].set_ylabel("Robustness Ratio (>1 = embedding more stable)")
        axes[1].set_title("Noise Robustness: Embedding vs Data Stability Ratio")
        axes[1].set_xticks(x + width * 1.5)
        axes[1].set_xticklabels(run_labels, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.plots_dir / "noise_robustness_comparison.png", dpi=150)
            print(f"[+] Saved noise robustness plot to {self.plots_dir}")
        
        plt.close()
    
    def plot_stack_similarity_comparison(self, results: List[EvalResult],
                                          save_plots: bool = True):
        """
        Plot stack similarity comparison across runs.
        """
        if not results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Create x-axis labels with loss info
        run_labels = []
        for r in results:
            best_loss = r.metrics.get("best_loss")
            if best_loss is not None:
                run_labels.append(f"{r.run_name}\n(loss: {best_loss:.3f})")
            else:
                run_labels.append(r.run_name)
        
        x = np.arange(len(run_labels))
        
        # Input vs Embedding stack matches
        input_matches = [r.metrics.get("stack_input_match_mean", 0) for r in results]
        emb_matches = [r.metrics.get("stack_emb_match_mean", 0) for r in results]
        
        width = 0.35
        axes[0].bar(x - width/2, input_matches, width, label='Input Space', alpha=0.8)
        axes[0].bar(x + width/2, emb_matches, width, label='Embedding Space', alpha=0.8)
        axes[0].set_xlabel("Run")
        axes[0].set_ylabel("Avg Same-Stack Matches in Top-k")
        axes[0].set_title("Stack Preservation: Input vs Embedding Space")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(run_labels, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Match scores
        match_scores = [r.metrics.get("stack_match_score_mean", 50) for r in results]
        colors = ['green' if s > 50 else 'orange' if s >= 45 else 'red' for s in match_scores]
        axes[1].bar(x, match_scores, color=colors, alpha=0.8)
        axes[1].axhline(y=50, color='gray', linestyle='--', label='Baseline (no improvement)')
        axes[1].set_xlabel("Run")
        axes[1].set_ylabel("Match Score (0-100)")
        axes[1].set_title("Stack Similarity Score (>50 = embedding preserves stacks better)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(run_labels, rotation=45, ha='right')
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.plots_dir / "stack_similarity_comparison.png", dpi=150)
            print(f"[+] Saved stack similarity plot to {self.plots_dir}")
        
        plt.close()
    
    def plot_embedding_similarity_histogram_comparison(self, results: List[EvalResult],
                                                       dataset_name: str = "valid",
                                                       save_plots: bool = True):
        """
        Plot histogram comparison of embedding and input similarity scores across checkpoints.
        
        Creates side-by-side histograms showing the distribution of pairwise cosine
        similarity scores for both input space and embedding space for each checkpoint,
        allowing comparison of similarity patterns.
        
        Args:
            results: List of EvalResult objects from multiple checkpoints
            dataset_name: Name of the dataset used for evaluation (e.g., "valid", "single_channel_10k")
            save_plots: Whether to save the plot
        """
        if not results:
            print("[!] No results provided for histogram comparison")
            return
        
        # Collect similarity scores for each result
        emb_similarity_scores_list = []
        input_similarity_scores_list = []
        run_names = []
        best_losses = []
        
        for r in results:
            run_name = r.run_name
            
            # Try to load embedding similarity scores from eval_data or numpy file
            emb_sims = None
            input_sims = None
            
            # First try eval_data (in-memory) with dataset_name
            emb_scores_key = f'embedding_similarity_scores_{dataset_name}'
            input_scores_key = f'input_similarity_scores_{dataset_name}'
            
            if run_name in self.eval_data and emb_scores_key in self.eval_data[run_name]:
                emb_sims = self.eval_data[run_name][emb_scores_key]
            else:
                # Try loading from numpy file (include dataset_name in filename)
                scores_path = self.data_dir_out / f"embedding_similarity_scores_{run_name}_{dataset_name}.npy"
                if scores_path.exists():
                    try:
                        emb_sims = np.load(scores_path)
                    except Exception as e:
                        print(f"[!] Warning: Could not load embedding similarity scores for {run_name}: {e}")
            
            # Try to load input similarity scores
            if run_name in self.eval_data and input_scores_key in self.eval_data[run_name]:
                input_sims = self.eval_data[run_name][input_scores_key]
            else:
                # Try loading from numpy file
                input_scores_path = self.data_dir_out / f"input_similarity_scores_{run_name}_{dataset_name}.npy"
                if input_scores_path.exists():
                    try:
                        input_sims = np.load(input_scores_path)
                    except Exception as e:
                        print(f"[!] Warning: Could not load input similarity scores for {run_name}: {e}")
            
            if emb_sims is not None and len(emb_sims) > 0:
                emb_similarity_scores_list.append(emb_sims)
                input_similarity_scores_list.append(input_sims)  # May be None
                run_names.append(run_name)
                best_loss = r.metrics.get("best_loss")
                best_losses.append(best_loss)
            else:
                print(f"[!] Warning: No embedding similarity scores found for {run_name} on dataset {dataset_name}, skipping histogram")
        
        if not emb_similarity_scores_list:
            print("[!] No similarity scores available for histogram comparison")
            return
        
        # Determine subplot layout (1 row, N columns)
        n_checkpoints = len(emb_similarity_scores_list)
        fig, axes = plt.subplots(1, n_checkpoints, figsize=(5 * n_checkpoints, 6))
        if n_checkpoints == 1:
            axes = [axes]
        
        # Determine consistent bins across all histograms for fair comparison
        bins = np.linspace(0, 1, 31)  # 30 bins from 0 to 1
        
        # Create histogram for each checkpoint
        for i, (emb_sims, input_sims, run_name, best_loss) in enumerate(
            zip(emb_similarity_scores_list, input_similarity_scores_list, run_names, best_losses)):
            
            # Calculate embedding statistics
            emb_mean = float(np.mean(emb_sims))
            emb_std = float(np.std(emb_sims))
            
            # Plot input histogram first (if available) so embedding is on top
            if input_sims is not None and len(input_sims) > 0:
                input_mean = float(np.mean(input_sims))
                input_std = float(np.std(input_sims))
                axes[i].hist(input_sims, bins=bins, alpha=0.5, color='orange', 
                           edgecolor='darkorange', linewidth=0.5,
                           label=f'Input (μ={input_mean:.3f}, σ={input_std:.3f})')
                axes[i].axvline(input_mean, color='darkorange', linestyle=':', linewidth=2)
            
            # Plot embedding histogram
            axes[i].hist(emb_sims, bins=bins, alpha=0.6, color='steelblue',
                        edgecolor='darkblue', linewidth=0.5,
                        label=f'Embedding (μ={emb_mean:.3f}, σ={emb_std:.3f})')
            axes[i].axvline(emb_mean, color='darkblue', linestyle='--', linewidth=2)
            
            # Set labels and title
            axes[i].set_xlabel('Cosine Similarity', fontsize=11)
            if i == 0:
                axes[i].set_ylabel('Frequency', fontsize=11)
            
            # Create title with run name and loss
            title = run_name
            if best_loss is not None:
                title += f"\n(loss: {best_loss:.3f})"
            axes[i].set_title(title, fontsize=10, fontweight='bold')
            
            axes[i].set_xlim(0, 1)
            axes[i].grid(alpha=0.3, axis='y')
            axes[i].legend(fontsize=8, loc='upper left')
        
        plt.suptitle(f'Input vs Embedding Similarity Score Distribution\nDataset: {dataset_name}', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_plots:
            filename = f"embedding_similarity_histogram_comparison_{dataset_name}.png"
            plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
            print(f"[+] Saved embedding similarity histogram comparison ({dataset_name}) to {self.plots_dir}")
        
        plt.close()
    
    def plot_embedding_similarity_matrix_comparison(self, results: List[EvalResult],
                                                    dataset_name: str = "valid",
                                                    save_plots: bool = True):
        """
        Plot side-by-side comparison of embedding similarity matrices across checkpoints.
        
        Creates side-by-side similarity matrix heatmaps for each checkpoint, allowing
        visual comparison of similarity patterns across models.
        
        Args:
            results: List of EvalResult objects from multiple checkpoints
            dataset_name: Name of the dataset used for evaluation (e.g., "valid", "single_channel_10k")
            save_plots: Whether to save the plot
        """
        if not results:
            print("[!] No results provided for similarity matrix comparison")
            return
        
        # Collect similarity matrices for each result
        similarity_matrices_list = []
        run_names = []
        best_losses = []
        
        for r in results:
            run_name = r.run_name
            
            # Try to get embeddings from eval_data to recompute similarity matrix
            # Embeddings are stored with key 'embeddings_{run_name}_{dataset_name}' in eval_data
            embeddings_key = f'embeddings_{run_name}_{dataset_name}'
            sim_matrix = None
            
            if embeddings_key in self.eval_data:
                embeddings = self.eval_data[embeddings_key]
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(embeddings)
                print(f"[+] Using stored embeddings for {run_name} on {dataset_name}: {embeddings.shape}")
            else:
                # Try loading from numpy file
                embeddings_path = self.data_dir_out / f"embeddings_{run_name}_{dataset_name}.npy"
                if embeddings_path.exists():
                    try:
                        embeddings = np.load(embeddings_path)
                        from sklearn.metrics.pairwise import cosine_similarity
                        sim_matrix = cosine_similarity(embeddings)
                        print(f"[+] Loaded embeddings from file for {run_name} on {dataset_name}: {embeddings.shape}")
                    except Exception as e:
                        print(f"[!] Warning: Could not load embeddings for {run_name}: {e}")
                else:
                    # Try recomputing from similarity scores (less ideal but works)
                    scores_key = f'embedding_similarity_scores_{dataset_name}'
                    if run_name in self.eval_data and scores_key in self.eval_data[run_name]:
                        # We have similarity scores but need to reconstruct the matrix
                        # This is approximate - we'd need the original sample count
                        print(f"[!] Warning: Cannot reconstruct similarity matrix from scores for {run_name}. Need embeddings.")
            
            if sim_matrix is not None and sim_matrix.shape[0] > 0:
                similarity_matrices_list.append(sim_matrix)
                run_names.append(run_name)
                best_loss = r.metrics.get("best_loss")
                best_losses.append(best_loss)
            else:
                print(f"[!] Warning: No similarity matrix found for {run_name} on dataset {dataset_name}, skipping")
        
        if not similarity_matrices_list:
            print("[!] No similarity matrices available for comparison")
            return
        
        # Determine subplot layout (1 row, N columns)
        n_checkpoints = len(similarity_matrices_list)
        fig, axes = plt.subplots(1, n_checkpoints, figsize=(5 * n_checkpoints, 5))
        if n_checkpoints == 1:
            axes = [axes]
        
        # Create similarity matrix heatmap for each checkpoint
        for i, (sim_matrix, run_name, best_loss) in enumerate(zip(similarity_matrices_list, run_names, best_losses)):
            # Calculate statistics (excluding diagonal)
            triu_indices = np.triu_indices_from(sim_matrix, k=1)
            sim_values = sim_matrix[triu_indices]
            sim_mean = float(np.mean(sim_values))
            sim_std = float(np.std(sim_values))
            
            # Create heatmap
            sns.heatmap(sim_matrix, ax=axes[i], cmap="viridis",
                       xticklabels=False, yticklabels=False,
                       vmin=0, vmax=1, cbar=True)
            
            # Set labels and title
            axes[i].set_xlabel(f'Sample Index (N={sim_matrix.shape[0]})', fontsize=10)
            if i == 0:
                axes[i].set_ylabel(f'Sample Index (N={sim_matrix.shape[0]})', fontsize=10)
            
            # Create title with run name and loss
            title = run_name
            if best_loss is not None:
                title += f"\n(loss: {best_loss:.3f})"
            title += f"\nmean={sim_mean:.3f}, std={sim_std:.3f}"
            axes[i].set_title(title, fontsize=10, fontweight='bold')
        
        plt.suptitle(f'Embedding Similarity Matrix Comparison\nDataset: {dataset_name}', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_plots:
            filename = f"embedding_similarity_matrix_comparison_{dataset_name}.png"
            plt.savefig(self.plots_dir / filename, dpi=150, bbox_inches='tight')
            print(f"[+] Saved embedding similarity matrix comparison ({dataset_name}) to {self.plots_dir}")
        
        plt.close()
    
    def plot_match_score_histogram(self, run_name: str, best_loss: Optional[float] = None, save_plots: bool = True):
        """
        Plot histogram of match scores comparing input vs embedding space.
        Similar to evaluate_stats_and_visualizations in evaluate.py.
        """
        key = f"match_df_{run_name}"
        if key not in self.eval_data:
            print(f"[!] No match_df data found for {run_name}")
            return
        
        match_df = self.eval_data[key]
        
        input_matches = match_df["input_stack_matches"].apply(len)
        emb_matches = match_df["embedding_stack_matches"].apply(len)
        
        max_matches = max(input_matches.max(), emb_matches.max())
        bins = np.arange(-0.5, max_matches + 1.5, 0.5)
        
        # Title with loss info
        loss_str = f" | Training Loss: {best_loss:.4f}" if best_loss is not None else ""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Run: {run_name}{loss_str}", fontsize=12, fontweight='bold')
        
        # Histogram of match counts
        counts_input, _, patches_input = axes[0].hist(
            input_matches, bins=bins, alpha=0.6, label="Input-space matches", color='steelblue'
        )
        counts_emb, _, patches_emb = axes[0].hist(
            emb_matches, bins=bins, alpha=0.6, label="Embedding-space matches", color='coral'
        )
        
        # Annotate counts
        for c, patch in zip(counts_input, patches_input):
            if c > 0:
                axes[0].text(patch.get_x() + patch.get_width()/2, c + 0.5, 
                           f"{int(c)}", ha='center', va='bottom', fontsize=8)
        
        axes[0].set_xlabel("Same-Stack Match Count")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Distribution: Input vs Embedding Stack Matches")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Match score distribution
        match_scores = match_df["match_score"].values
        axes[1].hist(match_scores, bins=20, alpha=0.7, color='mediumseagreen', edgecolor='darkgreen')
        axes[1].axvline(x=50, color='red', linestyle='--', linewidth=2, label='Baseline (50)')
        axes[1].axvline(x=match_scores.mean(), color='blue', linestyle='-', linewidth=2, 
                       label=f'Mean ({match_scores.mean():.1f})')
        axes[1].set_xlabel("Match Score (0-100)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"Match Score Distribution (>50 = embedding better)\nMean: {match_scores.mean():.1f}")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            run_plots_dir = self.plots_dir / run_name
            run_plots_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(run_plots_dir / "match_score_histogram.png", dpi=150)
            print(f"[+] Saved match score histogram to {run_plots_dir}")
        
        plt.close()
    
    def plot_noisy_vs_clean_spectrogram(self, run_name: str, best_loss: Optional[float] = None, 
                                         k: int = 3, save_plots: bool = True):
        """
        Plot best and worst noise robustness examples.
        Shows clean vs noisy spectrograms side by side.
        """
        key = f"noise_df_{run_name}"
        if key not in self.eval_data:
            print(f"[!] No noise data found for {run_name}")
            return
        
        noise_df = self.eval_data[key]
        noise_types = ["gaussian_std", "gaussian_mean", "gain_low", "gain_high"]
        
        run_plots_dir = self.plots_dir / run_name
        run_plots_dir.mkdir(parents=True, exist_ok=True)
        
        loss_str = f" | Training Loss: {best_loss:.4f}" if best_loss is not None else ""
        
        for noise_type in noise_types:
            emb_sim_col = f"{noise_type}_emb_sim"
            if emb_sim_col not in noise_df.columns:
                continue
            
            # Get best (highest similarity) and worst (lowest similarity) samples
            sorted_df = noise_df.sort_values(emb_sim_col, ascending=False)
            best_indices = sorted_df.head(k)["index"].values
            worst_indices = sorted_df.tail(k)["index"].values
            
            for status, indices in [("best", best_indices), ("worst", worst_indices)]:
                for idx in indices:
                    clean_key = f"clean_data_{run_name}_{idx}"
                    noisy_key = f"noisy_data_{run_name}_{idx}_{noise_type}"
                    
                    if clean_key not in self.eval_data or noisy_key not in self.eval_data:
                        continue
                    
                    clean_data = self.eval_data[clean_key]
                    noisy_data = self.eval_data[noisy_key]
                    
                    # Get similarity values
                    row = noise_df[noise_df["index"] == idx].iloc[0]
                    data_sim = row[f"{noise_type}_data_sim"]
                    emb_sim = row[emb_sim_col]
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(clean_data, label="Clean", color='black', linewidth=1.2)
                    ax.plot(noisy_data, label=f"Noisy ({noise_type})", alpha=0.7, linewidth=1)
                    ax.set_xlabel("Time / Position")
                    ax.set_ylabel("Amplitude")
                    ax.set_title(f"Run: {run_name}{loss_str}\n"
                               f"{status.upper()} | idx={idx} | {noise_type} | "
                               f"Data Sim: {data_sim:.4f} | Emb Sim: {emb_sim:.4f}")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    
                    plt.tight_layout()
                    
                    if save_plots:
                        fname = f"noisy_vs_clean_{noise_type}_{status}_idx{idx}.png"
                        plt.savefig(run_plots_dir / fname, dpi=150)
                    
                    plt.close()
        
        print(f"[+] Saved noisy vs clean spectrograms to {run_plots_dir}")
    
    def plot_embedding_vs_input_similarity_comparison(self, run_name: str, best_loss: Optional[float] = None,
                                                       k: int = 5, n_examples: int = 3, save_plots: bool = True):
        """
        Visualize similarity comparison for best and worst match score examples.
        Shows: embedding neighbors, input neighbors, same-stack neighbors.
        """
        key = f"match_df_{run_name}"
        if key not in self.eval_data:
            print(f"[!] No match_df data found for {run_name}")
            return
        
        match_df = self.eval_data[key]
        inputs_key = f"inputs_{run_name}"
        
        if inputs_key not in self.eval_data:
            print(f"[!] No input data found for {run_name}")
            return
        
        inputs = self.eval_data[inputs_key]
        
        run_plots_dir = self.plots_dir / run_name
        run_plots_dir.mkdir(parents=True, exist_ok=True)
        
        loss_str = f" | Training Loss: {best_loss:.4f}" if best_loss is not None else ""
        
        # Get best and worst examples
        sorted_df = match_df.sort_values("match_score", ascending=False)
        best_indices = sorted_df.head(n_examples)["index"].values
        worst_indices = sorted_df.tail(n_examples)["index"].values
        
        for status, indices in [("best", best_indices), ("worst", worst_indices)]:
            for query_idx in indices:
                row = match_df[match_df["index"] == query_idx].iloc[0]
                query_stack = row["stack_idx"]
                
                topk_emb_idx = row["embedding_neighbors"][:k]
                topk_input_idx = row["input_neighbors"][:k]
                emb_sims = row["embedding_similarities"][:k]
                input_sims = row["input_similarities"][:k]
                
                # Create visualization
                fig, axes = plt.subplots(3, k + 1, figsize=(3.5 * (k + 1), 9))
                fig.suptitle(
                    f"Run: {run_name}{loss_str}\n"
                    f"{status.upper()} | Query idx={query_idx} | stack={query_stack} | "
                    f"Match Score: {row['match_score']:.1f} | "
                    f"Emb matches: {len(row['embedding_stack_matches'])} | "
                    f"Input matches: {len(row['input_stack_matches'])}",
                    fontsize=11
                )
                
                row_titles = ["Embedding neighbors", "Input neighbors", "Query signal"]
                
                # Plot query in first column of each row
                for row_idx in range(3):
                    axes[row_idx, 0].plot(inputs[query_idx])
                    axes[row_idx, 0].set_title(f"QUERY\nidx={query_idx}")
                    axes[row_idx, 0].set_xticks([])
                    axes[row_idx, 0].set_ylabel(row_titles[row_idx])
                
                # Plot embedding neighbors
                for j, idx in enumerate(topk_emb_idx):
                    axes[0, j + 1].plot(inputs[idx])
                    axes[0, j + 1].set_title(f"idx={idx}\nsim={emb_sims[j]:.3f}")
                    axes[0, j + 1].set_xticks([])
                
                # Plot input neighbors
                for j, idx in enumerate(topk_input_idx):
                    axes[1, j + 1].plot(inputs[idx])
                    axes[1, j + 1].set_title(f"idx={idx}\nsim={input_sims[j]:.3f}")
                    axes[1, j + 1].set_xticks([])
                
                # Empty last row (or could show same-stack samples)
                for j in range(k):
                    axes[2, j + 1].axis('off')
                
                plt.tight_layout()
                
                if save_plots:
                    fname = f"similarity_comparison_{status}_query{query_idx}.png"
                    plt.savefig(run_plots_dir / fname, dpi=150)
                
                plt.close()
        
        print(f"[+] Saved embedding vs input similarity comparisons to {run_plots_dir}")
    
    def plot_signal_completion_comparison(self, results: List[EvalResult], 
                                           save_plots: bool = True):
        """
        Plot signal completion comparison across runs.
        """
        if not results:
            return
        
        strategies = ["causal_50", "causal_25", "random_30", "random_50"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Create x-axis labels with loss info
        run_labels = []
        for r in results:
            best_loss = r.metrics.get("best_loss")
            if best_loss is not None:
                run_labels.append(f"{r.run_name}\n(loss: {best_loss:.3f})")
            else:
                run_labels.append(r.run_name)
        
        x = np.arange(len(run_labels))
        width = 0.2
        
        # Plot cosine similarity for each strategy
        for i, strategy in enumerate(strategies):
            cos_sims = []
            for r in results:
                key = f"completion_{strategy}_cos_sim_mean"
                cos_sims.append(r.metrics.get(key, 0))
            axes[0].bar(x + i * width, cos_sims, width, label=strategy)
        
        axes[0].set_xlabel("Run")
        axes[0].set_ylabel("Cosine Similarity (higher = better completion)")
        axes[0].set_title("Signal Completion: Embedding Similarity at Masked Positions")
        axes[0].set_xticks(x + width * 1.5)
        axes[0].set_xticklabels(run_labels, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Plot overall completion score
        completion_scores = [r.metrics.get("completion_score", 50) for r in results]
        colors = ['green' if s > 70 else 'orange' if s >= 50 else 'red' for s in completion_scores]
        axes[1].bar(x, completion_scores, color=colors, alpha=0.8)
        axes[1].axhline(y=50, color='gray', linestyle='--', label='Baseline (random)')
        axes[1].set_xlabel("Run")
        axes[1].set_ylabel("Completion Score (0-100)")
        axes[1].set_title("Overall Signal Completion Score (>50 = better than random)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(run_labels, rotation=45, ha='right')
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.plots_dir / "signal_completion_comparison.png", dpi=150)
            print(f"[+] Saved signal completion plot to {self.plots_dir}")
        
        plt.close()
    
    def plot_signal_completion_histogram(self, run_name: str, best_loss: Optional[float] = None,
                                          save_plots: bool = True):
        """
        Plot histogram of signal completion scores for a single run.
        """
        key = f"completion_df_{run_name}"
        if key not in self.eval_data:
            print(f"[!] No completion data found for {run_name}")
            return
        
        completion_df = self.eval_data[key]
        strategies = ["causal_50", "causal_25", "random_30", "random_50"]
        
        loss_str = f" | Training Loss: {best_loss:.4f}" if best_loss is not None else ""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Signal Completion Analysis - Run: {run_name}{loss_str}", fontsize=12, fontweight='bold')
        
        for idx, strategy in enumerate(strategies):
            ax = axes[idx // 2, idx % 2]
            cos_sim_col = f"{strategy}_cos_sim"
            
            if cos_sim_col in completion_df.columns:
                values = completion_df[cos_sim_col].dropna().values
                if len(values) > 0:
                    ax.hist(values, bins=20, alpha=0.7, color='steelblue', edgecolor='darkblue')
                    ax.axvline(x=values.mean(), color='red', linestyle='-', linewidth=2, 
                               label=f'Mean ({values.mean():.3f})')
                    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Zero')
                    ax.set_xlabel("Cosine Similarity")
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"{strategy} (mean: {values.mean():.3f})")
                    ax.legend()
                    ax.grid(alpha=0.3)
                    ax.set_xlim(-1, 1)
        
        plt.tight_layout()
        
        if save_plots:
            run_plots_dir = self.plots_dir / run_name
            run_plots_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(run_plots_dir / "signal_completion_histogram.png", dpi=150)
            print(f"[+] Saved signal completion histogram to {run_plots_dir}")
        
        plt.close()
    
    def generate_evaluation_summary(self, results: List[EvalResult], 
                                     output_name: str = "evaluation_summary") -> str:
        """
        Generate a comprehensive text summary of all evaluations.
        Easy to read and analyze.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("SPECTRAL FM EVALUATION SUMMARY")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        
        # Overview
        lines.append("## OVERVIEW")
        lines.append(f"Total checkpoints evaluated: {len(results)}")
        lines.append(f"Output directory: {self.output_dir}")
        lines.append("")
        
        # Per-run summaries
        for result in results:
            lines.append("-" * 60)
            lines.append(f"### Run: {result.run_name}")
            lines.append(f"Checkpoint: {result.checkpoint_path}")
            lines.append(f"Evaluated: {result.timestamp}")
            lines.append("")
            
            # Key metrics
            m = result.metrics
            
            # Show which evaluation methods were run
            eval_methods_run = []
            if "eval_loss" in m:
                eval_methods_run.append("validation_loss")
            if "sim_variance_ratio" in m or "input_mean_sim" in m or "emb_mean_sim" in m:
                eval_methods_run.append("embedding_similarity")
            if "stack_match_score_mean" in m:
                eval_methods_run.append("stack_similarity")
            if any(f"noise_{nt}_emb_sim_mean" in m for nt in ["gaussian_std", "gaussian_mean", "gain_low", "gain_high"]):
                eval_methods_run.append("noise_robustness")
            if any(f"completion_{s}_cos_sim_mean" in m for s in ["causal_50", "causal_25", "random_30", "random_50"]):
                eval_methods_run.append("signal_completion")
            
            if eval_methods_run:
                lines.append(f"**Evaluation Methods Run:** {', '.join(eval_methods_run)}")
                lines.append("")
            
            # Best loss from training
            if "best_loss" in m and m["best_loss"] is not None:
                lines.append(f"**Training Best Loss:** {m['best_loss']:.6f}")
                lines.append("")
            
            # Validation loss (if validation_loss method was run)
            if "eval_loss" in m:
                lines.append("**Validation Loss:**")
                lines.append(f"  - Loss: {m.get('eval_loss', 'N/A'):.6f}")
                if "eval_loss_std" in m:
                    lines.append(f"  - Std: {m.get('eval_loss_std', 0):.6f}")
                if "eval_loss_min" in m:
                    lines.append(f"  - Min: {m.get('eval_loss_min', 0):.6f}")
                if "eval_loss_max" in m:
                    lines.append(f"  - Max: {m.get('eval_loss_max', 0):.6f}")
                if "eval_num_batches" in m:
                    lines.append(f"  - Batches: {m.get('eval_num_batches', 0)}")
                lines.append("")
            elif "validation_loss_error" in m:
                lines.append("**Validation Loss:**")
                lines.append(f"  - Error: {m.get('validation_loss_error', 'Unknown error')}")
                lines.append("")
            
            # Embedding Quality (skip Pearson/Spearman as requested)
            if "sim_variance_ratio" in m or "input_mean_sim" in m or "emb_mean_sim" in m:
                lines.append("**Embedding Quality:**")
                if "sim_variance_ratio" in m:
                    lines.append(f"  - Variance ratio: {m.get('sim_variance_ratio', 'N/A'):.4f}")
                if "input_mean_sim" in m:
                    lines.append(f"  - Input similarity: mean={m.get('input_mean_sim', 0):.4f}, std={m.get('input_std_sim', 0):.4f}")
                if "emb_mean_sim" in m:
                    lines.append(f"  - Embedding similarity: mean={m.get('emb_mean_sim', 0):.4f}, std={m.get('emb_std_sim', 0):.4f}")
                lines.append("")
            
            if "stack_match_score_mean" in m:
                lines.append("**Stack Preservation:**")
                lines.append(f"  - Match score (0-100): {m.get('stack_match_score_mean', 50):.1f}")
                lines.append(f"  - Input-space matches: {m.get('stack_input_match_mean', 0):.2f} ± {m.get('stack_input_match_std', 0):.2f}")
                lines.append(f"  - Embedding-space matches: {m.get('stack_emb_match_mean', 0):.2f} ± {m.get('stack_emb_match_std', 0):.2f}")
                lines.append(f"  - Improvement rate: {m.get('stack_match_improvement_pct', 0):.1f}%")
                lines.append("")
            
            noise_types = ["gaussian_std", "gaussian_mean", "gain_low", "gain_high"]
            has_noise = any(f"noise_{nt}_emb_sim_mean" in m for nt in noise_types)
            if has_noise:
                lines.append("**Noise Robustness:**")
                for nt in noise_types:
                    if f"noise_{nt}_emb_sim_mean" in m:
                        lines.append(f"  - {nt}:")
                        lines.append(f"      Data sim: {m.get(f'noise_{nt}_data_sim_mean', 0):.4f}")
                        lines.append(f"      Emb sim: {m.get(f'noise_{nt}_emb_sim_mean', 0):.4f}")
                        lines.append(f"      Robustness ratio: {m.get(f'noise_{nt}_robustness_ratio', 0):.4f}")
                lines.append("")
            
            completion_strategies = ["causal_50", "causal_25", "random_30", "random_50"]
            has_completion = any(f"completion_{s}_cos_sim_mean" in m for s in completion_strategies)
            if has_completion:
                lines.append("**Signal Completion:**")
                lines.append(f"  - Overall score (0-100): {m.get('completion_score', 50):.1f}")
                for s in completion_strategies:
                    if f"completion_{s}_cos_sim_mean" in m:
                        lines.append(f"  - {s}:")
                        lines.append(f"      Cosine sim: {m.get(f'completion_{s}_cos_sim_mean', 0):.4f}")
                        lines.append(f"      MSE: {m.get(f'completion_{s}_mse_mean', 0):.6f}")
                lines.append("")
            
            # Config summary
            if result.config_summary:
                lines.append("**Configuration:**")
                for k, v in result.config_summary.items():
                    if v:  # Only show non-empty values
                        lines.append(f"  - {k}: {v}")
                lines.append("")
        
        # Comparative analysis (if multiple runs)
        if len(results) > 1:
            lines.append("=" * 60)
            lines.append("## COMPARATIVE ANALYSIS")
            lines.append("")
            
            # Best performers (skip Pearson/Spearman as requested)
            # Best validation loss (lower is better)
            val_loss_scores = [(r.run_name, r.metrics.get("eval_loss", 999)) for r in results if "eval_loss" in r.metrics]
            if val_loss_scores:
                val_loss_scores.sort(key=lambda x: x[1])  # Lower is better
                lines.append("**Best Validation Loss:**")
                for i, (name, score) in enumerate(val_loss_scores[:3]):
                    lines.append(f"  {i+1}. {name}: {score:.6f}")
                lines.append("")
            
            match_scores = [(r.run_name, r.metrics.get("stack_match_score_mean", -999)) for r in results]
            match_scores.sort(key=lambda x: x[1], reverse=True)
            
            if match_scores[0][1] > -999:
                lines.append("**Best Stack Preservation:**")
                for i, (name, score) in enumerate(match_scores[:3]):
                    lines.append(f"  {i+1}. {name}: {score:.1f}")
                lines.append("")
            
            completion_scores = [(r.run_name, r.metrics.get("completion_score", -999)) for r in results]
            completion_scores.sort(key=lambda x: x[1], reverse=True)
            
            if completion_scores[0][1] > -999:
                lines.append("**Best Signal Completion:**")
                for i, (name, score) in enumerate(completion_scores[:3]):
                    lines.append(f"  {i+1}. {name}: {score:.1f}")
                lines.append("")
        
        # Interpretation guide
        lines.append("=" * 60)
        lines.append("## INTERPRETATION GUIDE")
        lines.append("")
        lines.append("**Validation Loss:**")
        lines.append("  Lower values indicate better model performance on validation set.")
        lines.append("  This is the same metric used during training.")
        lines.append("")
        lines.append("**Variance Ratio:**")
        lines.append("  Values near 0 suggest mode collapse (all embeddings similar).")
        lines.append("  Values near 1 suggest good diversity preservation.")
        lines.append("")
        lines.append("**Match Score (0-100):**")
        lines.append("  >50: Embedding space preserves group structure better than input space")
        lines.append("  =50: No improvement over input space")
        lines.append("  <50: Embedding space loses group structure")
        lines.append("")
        lines.append("**Noise Robustness Ratio:**")
        lines.append("  >1: Embeddings are more stable than raw data under noise")
        lines.append("  =1: Same stability as raw data")
        lines.append("  <1: Embeddings are less stable (more sensitive to noise)")
        lines.append("")
        lines.append("**Signal Completion Score (0-100):**")
        lines.append("  >70: Excellent completion capability (strong contextual understanding)")
        lines.append("  50-70: Good completion (model learns useful representations)")
        lines.append("  <50: Poor completion (model fails to capture context)")
        lines.append("  Causal strategies (causal_50, causal_25): Tests autoregressive prediction")
        lines.append("  Random strategies (random_30, random_50): Tests bidirectional context usage")
        lines.append("")
        lines.append("=" * 80)
        
        summary_text = "\n".join(lines)
        
        # Save to file
        summary_path = self.output_dir / f"{output_name}.txt"
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        print(f"[+] Saved evaluation summary to: {summary_path}")
        
        # Also print to console
        print("\n" + summary_text)
        
        return summary_text
    
    def generate_comparison_report_with_images(self, results: List[EvalResult],
                                                output_name: str = "comparison_report") -> str:
        """
        Generate a comprehensive markdown comparison report with embedded images.
        Includes: best loss, stack scores, overall score, and visualizations.
        Ordered by best loss (lowest first).
        """
        md_lines = []
        
        # Sort results by best_loss (lowest first)
        def get_loss(r):
            loss = r.metrics.get("best_loss")
            return loss if loss is not None else float('inf')
        
        sorted_results = sorted(results, key=get_loss)
        
        # Header
        md_lines.append("# SpectralFM Evaluation Comparison Report")
        md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append(f"**Total Runs:** {len(results)}")
        md_lines.append("**Ordered by:** Best Training Loss (lowest first)")
        md_lines.append("")
        
        # Summary Table (ordered by loss)
        md_lines.append("## Summary Table")
        md_lines.append("")
        md_lines.append("| Rank | Run | Best Loss | Stack Score | Pearson Corr | Variance Ratio |")
        md_lines.append("|------|-----|-----------|-------------|--------------|----------------|")
        
        for rank, r in enumerate(sorted_results, 1):
            m = r.metrics
            best_loss = m.get("best_loss", "N/A")
            if isinstance(best_loss, float):
                best_loss = f"{best_loss:.6f}"
            stack_score = m.get("stack_match_score_mean", "N/A")
            if isinstance(stack_score, float):
                stack_score = f"{stack_score:.1f}"
            pearson = m.get("pearson_corr", "N/A")
            if isinstance(pearson, float):
                pearson = f"{pearson:.4f}"
            var_ratio = m.get("sim_variance_ratio", "N/A")
            if isinstance(var_ratio, float):
                var_ratio = f"{var_ratio:.4f}"
            
            md_lines.append(f"| {rank} | {r.run_name} | {best_loss} | {stack_score} | {pearson} | {var_ratio} |")
        
        md_lines.append("")
        
        # Ranking Section
        md_lines.append("## Rankings")
        md_lines.append("")
        
        # By Best Loss (lower is better)
        loss_scores = [(r.run_name, r.metrics.get("best_loss", float('inf'))) for r in results]
        loss_scores = [(n, l) for n, l in loss_scores if l is not None and l != float('inf')]
        if loss_scores:
            loss_scores.sort(key=lambda x: x[1])
            md_lines.append("### Best Training Loss (Lower is Better)")
            for i, (name, score) in enumerate(loss_scores):
                md_lines.append(f"{i+1}. **{name}**: {score:.6f}")
            md_lines.append("")
        
        # By Stack Score (higher is better)
        stack_scores = [(r.run_name, r.metrics.get("stack_match_score_mean", -999)) for r in results]
        stack_scores = [(n, s) for n, s in stack_scores if s > -999]
        if stack_scores:
            stack_scores.sort(key=lambda x: x[1], reverse=True)
            md_lines.append("### Best Stack Preservation Score (Higher is Better)")
            for i, (name, score) in enumerate(stack_scores):
                md_lines.append(f"{i+1}. **{name}**: {score:.1f}")
            md_lines.append("")
        
        # By Pearson Correlation (higher is better)
        pearson_scores = [(r.run_name, r.metrics.get("pearson_corr", -999)) for r in results]
        pearson_scores = [(n, p) for n, p in pearson_scores if p > -999]
        if pearson_scores:
            pearson_scores.sort(key=lambda x: x[1], reverse=True)
            md_lines.append("### Best Embedding Correlation (Higher is Better)")
            for i, (name, score) in enumerate(pearson_scores):
                md_lines.append(f"{i+1}. **{name}**: {score:.4f}")
            md_lines.append("")
        
        # Per-Run Details with Images (ordered by loss)
        md_lines.append("## Per-Run Details (Ordered by Training Loss)")
        md_lines.append("")
        
        for rank, r in enumerate(sorted_results, 1):
            m = r.metrics
            loss_val = m.get("best_loss")
            loss_str = f" (Loss: {loss_val:.4f})" if loss_val is not None else ""
            
            md_lines.append(f"### #{rank}: {r.run_name}{loss_str}")
            md_lines.append("")
            
            # Training info
            if "best_loss" in m and m["best_loss"] is not None:
                md_lines.append(f"- **Best Training Loss:** {m['best_loss']:.6f}")
            
            # Embedding metrics
            if "pearson_corr" in m:
                md_lines.append(f"- **Pearson Correlation:** {m.get('pearson_corr', 'N/A'):.4f}")
                md_lines.append(f"- **Spearman Correlation:** {m.get('spearman_corr', 'N/A'):.4f}")
                md_lines.append(f"- **Variance Ratio:** {m.get('sim_variance_ratio', 'N/A'):.4f}")
            
            # Stack metrics
            if "stack_match_score_mean" in m:
                md_lines.append(f"- **Stack Match Score:** {m.get('stack_match_score_mean', 50):.1f}")
                md_lines.append(f"- **Improvement Rate:** {m.get('stack_match_improvement_pct', 0):.1f}%")
            
            md_lines.append("")
            
            # Images
            run_plots_dir = self.plots_dir / r.run_name
            
            # Similarity matrices
            sim_matrix_path = run_plots_dir / "similarity_matrices.png"
            if sim_matrix_path.exists():
                rel_path = f"plots/{r.run_name}/similarity_matrices.png"
                md_lines.append("#### Cosine Similarity Matrices")
                md_lines.append(f"![Similarity Matrices]({rel_path})")
                md_lines.append("")
            
            # Match score histogram
            match_hist_path = run_plots_dir / "match_score_histogram.png"
            if match_hist_path.exists():
                rel_path = f"plots/{r.run_name}/match_score_histogram.png"
                md_lines.append("#### Stack Match Score Histogram")
                md_lines.append(f"![Match Score Histogram]({rel_path})")
                md_lines.append("")
            
            md_lines.append("---")
            md_lines.append("")
        
        # Comparison Images (if multiple runs)
        if len(results) > 1:
            md_lines.append("## Comparison Visualizations")
            md_lines.append("")
            
            # Check for histogram comparison plots (may have dataset name in filename)
            embedding_hist_files = list(self.plots_dir.glob("embedding_similarity_histogram_comparison*.png"))
            if embedding_hist_files:
                for hist_file in embedding_hist_files:
                    dataset_suffix = hist_file.stem.replace("embedding_similarity_histogram_comparison_", "")
                    dataset_name_display = dataset_suffix if dataset_suffix else "valid"
                    md_lines.append(f"### Embedding Similarity Histogram Comparison ({dataset_name_display})")
                    md_lines.append("")
                    md_lines.append("Distribution of pairwise cosine similarity scores across checkpoints:")
                    md_lines.append("")
                    md_lines.append(f"![Embedding Similarity Histogram](plots/{hist_file.name})")
                    md_lines.append("")
                md_lines.append("### Embedding Similarity Histogram Comparison")
                md_lines.append("")
                md_lines.append("Distribution of pairwise cosine similarity scores across checkpoints:")
                md_lines.append("")
                md_lines.append("![Embedding Similarity Histogram](plots/embedding_similarity_histogram_comparison.png)")
                md_lines.append("")
            
            noise_comparison = self.plots_dir / "noise_robustness_comparison.png"
            if noise_comparison.exists():
                md_lines.append("### Noise Robustness Comparison")
                md_lines.append("![Noise Robustness](plots/noise_robustness_comparison.png)")
                md_lines.append("")
            
            stack_comparison = self.plots_dir / "stack_similarity_comparison.png"
            if stack_comparison.exists():
                md_lines.append("### Stack Similarity Comparison")
                md_lines.append("![Stack Comparison](plots/stack_similarity_comparison.png)")
                md_lines.append("")
        
        # Overall Scores
        md_lines.append("## Overall Score Computation")
        md_lines.append("")
        md_lines.append("The overall score combines multiple metrics (weighted):")
        md_lines.append("- Training Loss (inverted, lower is better): 30%")
        md_lines.append("- Stack Match Score: 35%")
        md_lines.append("- Pearson Correlation: 35%")
        md_lines.append("")
        
        # Compute overall scores
        overall_scores = []
        for r in results:
            m = r.metrics
            
            # Normalize metrics to 0-100 scale
            loss = m.get("best_loss")
            stack_score = m.get("stack_match_score_mean", 50)
            pearson = m.get("pearson_corr", 0)
            
            # Loss: lower is better, typical range 0-10, normalize to 0-100
            if loss is not None and loss < 100:
                loss_score = max(0, 100 - loss * 10)  # Approximate normalization
            else:
                loss_score = 50
            
            # Stack score already 0-100
            
            # Pearson: -1 to 1, normalize to 0-100
            pearson_score = (pearson + 1) * 50
            
            # Weighted combination
            overall = 0.30 * loss_score + 0.35 * stack_score + 0.35 * pearson_score
            overall_scores.append((r.run_name, overall, loss_score, stack_score, pearson_score))
        
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        
        md_lines.append("### Overall Rankings")
        md_lines.append("")
        md_lines.append("| Rank | Run | Overall Score | Loss Score | Stack Score | Correlation Score |")
        md_lines.append("|------|-----|---------------|------------|-------------|-------------------|")
        
        for i, (name, overall, loss_s, stack_s, pearson_s) in enumerate(overall_scores):
            md_lines.append(f"| {i+1} | {name} | {overall:.1f} | {loss_s:.1f} | {stack_s:.1f} | {pearson_s:.1f} |")
        
        md_lines.append("")
        
        # Winner
        if overall_scores:
            winner = overall_scores[0]
            md_lines.append(f"## 🏆 Best Overall: **{winner[0]}** (Score: {winner[1]:.1f})")
        
        md_lines.append("")
        
        md_report = "\n".join(md_lines)
        
        # Save markdown report
        report_path = self.output_dir / f"{output_name}.md"
        with open(report_path, 'w') as f:
            f.write(md_report)
        print(f"[+] Saved markdown comparison report to: {report_path}")
        
        return md_report
    
    def _extract_best_loss(self, checkpoint_path: str) -> Optional[float]:
        """Extract best_loss from a checkpoint file."""
        try:
            from fairseq import checkpoint_utils
            state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path, arg_overrides={})
            
            # Try multiple locations for best loss
            best_loss = state.get("best", None)
            
            # Also check extra_state where fairseq often stores this
            if best_loss is None and "extra_state" in state:
                extra_state = state["extra_state"]
                best_loss = extra_state.get("best", None)
                if best_loss is None:
                    best_loss = extra_state.get("val_loss", None)
            
            if best_loss is not None:
                print(f"[+] Best loss from checkpoint: {best_loss:.6f}")
            return float(best_loss) if best_loss is not None else None
        except Exception as e:
            print(f"[!] Could not extract best_loss: {e}")
            return None
    
    def _extract_config_summary(self, config: Dict) -> Dict[str, Any]:
        """Extract key configuration parameters for comparison."""
        summary = {}
        
        if not config:
            return summary
        
        # Model params
        model_cfg = config.get("model", {})
        summary["encoder_embed_dim"] = model_cfg.get("encoder_embed_dim", 768)
        summary["encoder_layers"] = model_cfg.get("encoder_layers", 12)
        summary["mask_prob"] = model_cfg.get("mask_prob", 0.65)
        summary["mask_length"] = model_cfg.get("mask_length", 10)
        summary["conv_feature_layers"] = str(model_cfg.get("conv_feature_layers", ""))
        summary["train_only_fe"] = model_cfg.get("train_only_fe", False)
        
        # Training params
        opt_cfg = config.get("optimization", {})
        summary["lr"] = opt_cfg.get("lr", [0.0001])
        summary["max_update"] = opt_cfg.get("max_update", 0)
        summary["max_epoch"] = opt_cfg.get("max_epoch", 0)
        
        # Dataset params
        ds_cfg = config.get("dataset", {})
        summary["batch_size"] = ds_cfg.get("batch_size", 0)
        
        # Task params
        task_cfg = config.get("task", {})
        summary["data_path"] = task_cfg.get("data", "")
        
        return summary


class ReportGenerator:
    """Generates comparison reports from evaluation results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comparison_report(self, results: List[EvalResult], 
                                  output_name: str = "comparison_report") -> pd.DataFrame:
        """Generate a comparison report as DataFrame and save to CSV."""
        if not results:
            print("[!] No results to report")
            return pd.DataFrame()
        
        rows = []
        for result in results:
            row = {
                "run_name": result.run_name,
                "checkpoint_path": result.checkpoint_path,
                "eval_timestamp": result.timestamp,
            }
            
            for k, v in result.metrics.items():
                row[f"metric_{k}"] = v
            
            for k, v in result.config_summary.items():
                row[f"config_{k}"] = v
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values("run_name", ascending=False)
        
        csv_path = self.output_dir / f"{output_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[+] Saved comparison report to: {csv_path}")
        
        self._print_summary(df)
        
        json_path = self.output_dir / f"{output_name}.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        print(f"[+] Saved detailed results to: {json_path}")
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print a summary of the comparison."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        metric_cols = [c for c in df.columns if c.startswith("metric_")]
        
        print(f"\nTotal runs evaluated: {len(df)}")
        
        # if metric_cols: # fixme
        #     print("\nMetrics Overview:")
        #     for col in metric_cols:
        #         metric_name = col.replace("metric_", "")
        #         values = df[col].dropna()
        #         if len(values) > 0 and values.dtype in ['float64', 'int64']:
        #             print(f"  {metric_name}:")
        #             print(f"    Mean: {values.mean():.4f}")
        #             print(f"    Std:  {values.std():.4f}")
        #             print(f"    Best: {values.max():.4f} (run: {df.loc[values.idxmax(), 'run_name']})")
        
        print("="*80 + "\n")


def _default_structured_similarity_json_path() -> Optional[str]:
    """Repo ``structured_similarity_full.json`` if present (100 exact manifest indices)."""
    p = (
        Path(__file__).resolve().parent.parent
        / "fairseq/examples/data2vec/config/audio/pretraining/recon_loss/structured_similarity_full.json"
    )
    return str(p) if p.is_file() else None


def main():
    parser = argparse.ArgumentParser(description="SpectralFM Evaluation Runner")
    
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="/mnt5/noy/fairseq/outputs",
                       help="Base directory for fairseq outputs")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to specific checkpoint file")
    parser.add_argument("--extra_checkpoints", type=str, nargs="*", default=None,
                       help="Additional checkpoint .pt files to include (from any directory)")
    parser.add_argument("--output_dir", type=str, 
                       default="/mnt5/noy/SpectralFM/code/eval_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--data_dir", type=str,
                       default="/mnt5/noy/fairseq/data/single_channel_1m/",
                       help="Directory containing evaluation data (stored but currently unused - use --eval_data_dir instead)")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                       help="Evaluation data directory override. Priority: --eval_data_dir > --data_dir > checkpoint's cfg.task.data")
    parser.add_argument("--eval_methods", type=str, nargs="+",
                       default=["embedding_similarity"],
                       help="Evaluation methods: embedding_similarity, structured_similarity, noise_robustness, stack_similarity, signal_completion, validation_loss")
    parser.add_argument(
        "--nova_data_dir",
        type=str,
        default=None,
        help="Parent of nova datasets (single_channel_all, multi_channel, …). "
        "Defaults to $SPECTRALFM_NOVA_DATA_DIR or /mnt5/noy/fairseq/data. Required for structured_similarity.",
    )
    parser.add_argument(
        "--structured_similarity_seed",
        type=int,
        default=42,
        help="RNG seed for build_structured_similarity_subset (default 42; must match offline tools).",
    )
    parser.add_argument(
        "--structured_similarity_entries_json",
        type=str,
        default=None,
        help="Path to 100-entry JSON (e.g. structured_similarity_full.json). "
        "Default: repo file under fairseq/.../recon_loss/ if it exists.",
    )
    parser.add_argument(
        "--structured_similarity_ignore_entries_json",
        action="store_true",
        help="Do not load default structured_similarity_full.json; build the panel with RNG + seed instead.",
    )
    parser.add_argument(
        "--structured_similarity_prefer_manifest",
        type=str,
        default="train",
        choices=("train", "valid"),
        help="Manifest for RNG-built panel line indices (default train). JSON panel uses prefer_manifest from the file.",
    )
    parser.add_argument(
        "--structured_similarity_allow_single_channel_fallback",
        action="store_true",
        help="If the full nova tree is incomplete, fall back to 10×10 stacks from single_channel_all only.",
    )
    parser.add_argument("--best_only", action="store_true",
                       help="Only evaluate checkpoint_best.pt files")
    parser.add_argument("--latest_only", action="store_true",
                       help="Only evaluate the most recent checkpoint")
    parser.add_argument("--run_names", type=str, nargs="+", default=None,
                       help="Specific run names to evaluate (e.g., 2026-01-07_21-50-07). Only these checkpoints will be evaluated.")
    parser.add_argument("--report_name", type=str, default=None,
                       help="Custom name for the report")
    parser.add_argument("--plot_matrices", action="store_true",
                       help="Generate similarity matrix visualizations")
    parser.add_argument("--all_methods", action="store_true",
                       help="Run all available evaluation methods")
    parser.add_argument("--custom_dataset_path", type=str, default=None,
                       help="Path to custom dataset for evaluation (will run 4-way comparison on both 'valid' and this dataset)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode: save plots of evaluated data samples to debug_plots directory")
    parser.add_argument("--include_random_weights", action="store_true",
                       help="Include pretrained transformers model (16k stretched) in embedding similarity comparison (5-way instead of 4-way)")
    parser.add_argument("--analyze_outliers", action="store_true",
                       help="Enable outlier analysis: find samples with lowest average similarity and visualize them")
    parser.add_argument("--outlier_run_name", type=str, default=None,
                       help="Specific run_name to analyze for outliers (required if --analyze_outliers). If not specified, uses all evaluated checkpoints.")
    parser.add_argument("--outlier_dataset", type=str, default=None,
                       help="Dataset name for outlier analysis (default: use custom_dataset_path name or 'valid')")
    parser.add_argument("--outlier_similarity_type", type=str, default="both",
                       choices=["embedding", "input", "both"],
                       help="Type of similarity for outlier analysis: 'embedding', 'input', or 'both' (default: 'both')")
    parser.add_argument("--analyze_inliers", action="store_true", default=True,
                       help="Also analyze inliers (highest avg similarity) in addition to outliers (default: True)")
    parser.add_argument("--no_analyze_inliers", dest="analyze_inliers", action="store_false",
                       help="Disable inlier analysis (only analyze outliers)")
    parser.add_argument("--mask_memory_path", type=str, default=None,
                       help="Path to mask memory file for fixed mask evaluation")
    
    args = parser.parse_args()
    
    # Create timestamped subdirectory for this evaluation run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_output_dir = Path(args.output_dir) / timestamp
    timestamped_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[+] Creating timestamped output directory: {timestamped_output_dir}")
    
    # Update output_dir to use timestamped subdirectory
    args.output_dir = str(timestamped_output_dir)
    
    if args.report_name is None:
        args.report_name = f"eval_report_{timestamp}"
    
    # If --all_methods, use all available eval methods
    if args.all_methods:
        args.eval_methods = ["embedding_similarity", "noise_robustness", "stack_similarity", "signal_completion", "validation_loss"]
    
    # Discover checkpoints
    if args.checkpoint:
        checkpoints = [CheckpointInfo(
            path=args.checkpoint,
            run_dir=str(Path(args.checkpoint).parent.parent),
            date=datetime.now().strftime("%Y-%m-%d"),
            time=datetime.now().strftime("%H-%M-%S"),
            checkpoint_type="manual"
        )]
    else:
        discovery = CheckpointDiscovery(args.checkpoint_dir)
        
        if args.latest_only:
            ckpt = discovery.find_latest_checkpoint()
            checkpoints = [ckpt] if ckpt else []
        elif args.best_only:
            checkpoints = discovery.find_best_checkpoints()
        else:
            checkpoints = discovery.find_all_checkpoints()
        
        # Filter by specific run names if provided
        if args.run_names:
            run_names_set = set(args.run_names)
            original_count = len(checkpoints)
            checkpoints = [c for c in checkpoints if c.run_name in run_names_set]
            print(f"[+] Filtered checkpoints: {len(checkpoints)}/{original_count} match the specified run names")
            if len(checkpoints) < len(run_names_set):
                found_names = {c.run_name for c in checkpoints}
                missing_names = run_names_set - found_names
                print(f"[!] Warning: Some specified run names were not found: {missing_names}")

    if args.extra_checkpoints:
        for cp_path in args.extra_checkpoints:
            cp = Path(cp_path)
            if not cp.exists():
                print(f"[!] Warning: extra checkpoint not found: {cp_path}")
                continue
            checkpoints.append(CheckpointInfo(
                path=str(cp.resolve()),
                run_dir=str(cp.parent.resolve()),
                date="unknown",
                time=cp.stem,
                checkpoint_type="manual",
            ))
            print(f"    [+] Added extra checkpoint: {cp.stem}")
    
    if not checkpoints:
        print("[!] No checkpoints found!")
        return
    
    print(f"[+] Found {len(checkpoints)} checkpoint(s)")
    for ckpt in checkpoints:
        print(f"    - {ckpt.run_name} ({ckpt.checkpoint_type})")
    
    print(f"[+] Running evaluation methods: {args.eval_methods}")
    
    nova_resolved = args.nova_data_dir
    if nova_resolved is None:
        nova_resolved = os.environ.get("SPECTRALFM_NOVA_DATA_DIR", "/mnt5/noy/fairseq/data")

    if args.structured_similarity_ignore_entries_json:
        resolved_struct_json = None
    elif args.structured_similarity_entries_json:
        resolved_struct_json = os.path.expanduser(args.structured_similarity_entries_json)
    else:
        resolved_struct_json = _default_structured_similarity_json_path()
    
    # Run evaluations
    runner = EvaluationRunner(args.output_dir, args.data_dir)
    results = runner.evaluate_all(checkpoints, args.eval_methods, 
                                  custom_dataset_path=args.custom_dataset_path,
                                  eval_data_dir=args.eval_data_dir,
                                  debug=args.debug,
                                  include_random_weights=args.include_random_weights,
                                  mask_memory_path=args.mask_memory_path,
                                  nova_data_dir=nova_resolved,
                                  structured_similarity_seed=args.structured_similarity_seed,
                                  structured_similarity_entries_json=resolved_struct_json,
                                  structured_similarity_prefer_manifest=args.structured_similarity_prefer_manifest,
                                  structured_similarity_allow_single_channel_fallback=args.structured_similarity_allow_single_channel_fallback)
    
    # Create lookup for best_loss by run_name
    loss_lookup = {r.run_name: r.metrics.get("best_loss") for r in results}
    
    # Generate visualizations for each checkpoint
    print("\n[+] Generating visualizations...")
    for ckpt in checkpoints:
        try:
            run_name = ckpt.run_name
            best_loss = loss_lookup.get(run_name)
            
            # Similarity matrices are generated inside evaluate_checkpoint when
            # embedding_similarity or stack_similarity methods are run.
            # Only generate here as a fallback if those methods weren't run but matrices are requested.
            embedding_methods_run = {"embedding_similarity", "stack_similarity"} & set(args.eval_methods)
            if (args.plot_matrices or args.all_methods) and len(embedding_methods_run) == 0:
                # Fallback: generate if embedding methods weren't run
                runner.plot_similarity_matrices(ckpt, best_loss=best_loss, save_plots=True)
            
            # Stack similarity visualizations
            if "stack_similarity" in args.eval_methods:
                runner.plot_match_score_histogram(run_name, best_loss=best_loss, save_plots=True)
                runner.plot_embedding_vs_input_similarity_comparison(run_name, best_loss=best_loss, k=5, n_examples=3, save_plots=True)
            
            # Noise robustness visualizations
            if "noise_robustness" in args.eval_methods:
                runner.plot_noisy_vs_clean_spectrogram(run_name, best_loss=best_loss, k=2, save_plots=True)
            
            # Signal completion visualizations
            if "signal_completion" in args.eval_methods:
                runner.plot_signal_completion_histogram(run_name, best_loss=best_loss, save_plots=True)
                
        except Exception as e:
            print(f"[!] Error generating visualizations for {ckpt.run_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate comparison plots for multi-checkpoint evaluations
    if len(results) > 1:
        print("\n[+] Generating comparison plots...")
        if "embedding_similarity" in args.eval_methods:
            # Determine dataset name for comparison plots
            # If custom_dataset_path is provided, use that dataset; otherwise use "valid"
            comparison_dataset_name = "valid"
            if args.custom_dataset_path:
                comparison_dataset_name = Path(args.custom_dataset_path).name
            
            # Histogram comparison
            runner.plot_embedding_similarity_histogram_comparison(results, dataset_name=comparison_dataset_name, save_plots=True)
            
            # Similarity matrix comparison (side-by-side heatmaps)
            runner.plot_embedding_similarity_matrix_comparison(results, dataset_name=comparison_dataset_name, save_plots=True)
            
            # Side-by-side embedding similarity comparison plots (from compare_checkpoints.py)
            try:
                from compare_checkpoints import create_side_by_side_plots
                checkpoints_for_comparison = [CheckpointInfo(
                    path=r.checkpoint_path,
                    run_dir=str(Path(r.checkpoint_path).parent.parent) if Path(r.checkpoint_path).parent.parent.exists() else str(Path(r.checkpoint_path).parent),
                    date=r.run_name.split("_")[0] if "_" in r.run_name else r.run_name.split("-")[0],
                    time=r.run_name.split("_", 1)[1].replace("-", ":") if "_" in r.run_name else r.run_name,
                    checkpoint_type="best"
                ) for r in results]
                create_side_by_side_plots(checkpoints_for_comparison, runner, Path(args.output_dir), 
                                         custom_dataset_path=args.custom_dataset_path)
            except ImportError as e:
                print(f"[!] Warning: Could not import create_side_by_side_plots: {e}")
            except Exception as e:
                print(f"[!] Warning: Could not create side-by-side comparison plots: {e}")
                import traceback
                traceback.print_exc()
        if "noise_robustness" in args.eval_methods:
            runner.plot_noise_robustness_comparison(results, save_plots=True)
        if "stack_similarity" in args.eval_methods:
            runner.plot_stack_similarity_comparison(results, save_plots=True)
        if "signal_completion" in args.eval_methods:
            runner.plot_signal_completion_comparison(results, save_plots=True)
    
    # Run outlier analysis if requested
    if args.analyze_outliers and "embedding_similarity" in args.eval_methods:
        print("\n[+] Running similarity outlier analysis...")
        
        # Determine dataset name for outlier analysis
        outlier_dataset_name = args.outlier_dataset
        if outlier_dataset_name is None:
            if args.custom_dataset_path:
                outlier_dataset_name = Path(args.custom_dataset_path).name
            else:
                outlier_dataset_name = "valid"
        
        # Determine which checkpoints to analyze
        checkpoints_to_analyze = results
        if args.outlier_run_name:
            checkpoints_to_analyze = [r for r in results if r.run_name == args.outlier_run_name]
            if not checkpoints_to_analyze:
                print(f"[!] Warning: No checkpoint found with run_name '{args.outlier_run_name}'")
                print(f"[!] Available run names: {[r.run_name for r in results]}")
        
        if checkpoints_to_analyze:
            for result in checkpoints_to_analyze:
                try:
                    runner.analyze_similarity_outliers(
                        run_name=result.run_name,
                        dataset_name=outlier_dataset_name,
                        similarity_type=args.outlier_similarity_type,
                        k_outliers=5,
                        k_neighbors=5,
                        save_plots=True,
                        analyze_inliers=args.analyze_inliers
                    )
                except Exception as e:
                    print(f"[!] Error analyzing outliers for {result.run_name}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("[!] No checkpoints available for outlier analysis")
    
    # Generate report
    reporter = ReportGenerator(args.output_dir)
    df = reporter.generate_comparison_report(results, args.report_name)
    
    # Generate comprehensive summary
    runner.generate_evaluation_summary(results, args.report_name + "_summary")
    
    # Generate markdown comparison report with images
    runner.generate_comparison_report_with_images(results, args.report_name + "_comparison")
    
    print(f"\n[+] Evaluation complete!")
    print(f"[+] All results saved to timestamped directory: {timestamped_output_dir}")
    print(f"[+] Plots saved to: {runner.plots_dir}")
    print(f"[+] Data saved to: {runner.data_dir_out}")
    print(f"[+] Summary: {args.output_dir}/{args.report_name}_summary.txt")
    print(f"[+] Comparison Report: {args.output_dir}/{args.report_name}_comparison.md")


if __name__ == "__main__":
    main()
