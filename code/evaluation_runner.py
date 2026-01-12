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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

# Suppress torchaudio/torio FFmpeg extension loading warnings
# These are non-critical - torchaudio will fall back to available FFmpeg versions
logging.getLogger("torio._extension.utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FFmpeg.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*libavutil.*", category=UserWarning)

# Add fairseq to path using relative path from this file's location
_FAIRSEQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fairseq")
if _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)

from model_loader import load_fairseq_checkpoint, load_fairseq_model_for_evaluation
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
    """Runs evaluations on checkpoints and collects results."""
    
    def __init__(self, output_dir: str, data_dir: str = "/mnt5/noy/fairseq/data/single_channel_1m/"):
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
        
    def evaluate_checkpoint(self, checkpoint_info: CheckpointInfo, 
                          eval_methods: List[str] = None) -> EvalResult:
        """
        Run evaluation on a single checkpoint.
        
        Args:
            checkpoint_info: Information about the checkpoint to evaluate
            eval_methods: List of evaluation methods to run
            
        Returns:
            EvalResult with metrics
        """
        print(f"[DEBUG] evaluate_checkpoint received eval_methods: {eval_methods} (type: {type(eval_methods)})")
        
        if eval_methods is None:
            eval_methods = ["embedding_similarity"]
            print(f"[DEBUG] eval_methods was None, using default: {eval_methods}")
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint_info.run_name}")
        print(f"Checkpoint: {checkpoint_info.checkpoint_type}")
        print(f"{'='*60}")
        
        # Store current run name for use in sub-methods
        self._current_run_name = checkpoint_info.run_name
        
        # Load model and extract best_loss from checkpoint
        try:
            model, cfg = load_fairseq_model_for_evaluation(checkpoint_info.path)
            
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
        
        # Run each evaluation method
        for method in eval_methods:
            try:
                if method == "embedding_similarity":
                    method_metrics = self._eval_embedding_similarity(model)
                elif method == "signal_completion":
                    method_metrics = self._eval_signal_completion(model)
                elif method == "noise_robustness":
                    method_metrics = self._eval_noise_robustness(model)
                elif method == "stack_similarity":
                    method_metrics = self._eval_stack_similarity(model)
                elif method == "validation_loss":
                    method_metrics = self._eval_validation_loss(model, cfg)
                else:
                    print(f"[!] Unknown eval method: {method}")
                    continue
                    
                if method_metrics:
                    metrics.update(method_metrics)
                    # Debug: print what metrics were added
                    if method == "validation_loss":
                        print(f"[DEBUG] Added validation_loss metrics: {list(method_metrics.keys())}")
                else:
                    print(f"[!] Warning: {method} returned empty metrics")
            except Exception as e:
                print(f"[!] Error in {method}: {e}")
                import traceback
                traceback.print_exc()
                metrics[f"{method}_error"] = str(e)
        
        # Extract key config parameters for comparison
        config_summary = self._extract_config_summary(checkpoint_info.config)
        
        result = EvalResult(
            checkpoint_path=checkpoint_info.path,
            run_name=checkpoint_info.run_name,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            config_summary=config_summary
        )
        
        # Debug: print metrics keys to help diagnose
        if eval_methods and "validation_loss" in eval_methods:
            if "eval_loss" in metrics:
                print(f"[DEBUG] ✓ eval_loss found in metrics: {metrics.get('eval_loss')}")
            elif "validation_loss_error" in metrics:
                print(f"[DEBUG] ✗ validation_loss failed with error: {metrics.get('validation_loss_error')}")
            else:
                print(f"[DEBUG] ⚠ validation_loss method run but no eval_loss or error found. Metrics keys: {list(metrics.keys())[:20]}")
        
        self.results.append(result)
        return result
    
    def evaluate_all(self, checkpoints: List[CheckpointInfo], 
                    eval_methods: List[str] = None) -> List[EvalResult]:
        """Evaluate multiple checkpoints."""
        print(f"[DEBUG] evaluate_all received eval_methods: {eval_methods} (type: {type(eval_methods)})")
        results = []
        for ckpt in tqdm(checkpoints, desc="Evaluating checkpoints"):
            print(f"[DEBUG] Calling evaluate_checkpoint with eval_methods: {eval_methods}")
            result = self.evaluate_checkpoint(ckpt, eval_methods)
            results.append(result)
        return results
    
    def _eval_embedding_similarity(self, model) -> Dict[str, float]:
        """
        Evaluate embedding quality by comparing input-space vs embedding-space similarity.
        
        A good model should:
        - Preserve relative similarities (similar inputs -> similar embeddings)
        - Have higher variance in embedding similarity than a collapsed model
        - Show correlation between input and embedding similarities
        """
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.stats import pearsonr, spearmanr
        import numpy as np
        import torchaudio
        import glob
        
        device = next(model.parameters()).device
        model.eval()
        
        # Load wav files directly from data_dir
        wav_files = glob.glob(os.path.join(self.data_dir, "*.wav"))
        
        if len(wav_files) == 0:
            return {"error": f"No wav files found in {self.data_dir}"}
        
        # Sample for efficiency
        sample_size = min(100, len(wav_files))
        import random
        random.seed(42)
        sampled_files = random.sample(wav_files, sample_size)
        
        print(f"[+] Loading {sample_size} wav files for evaluation...")
        
        inputs = []
        embeddings = []
        
        with torch.no_grad():
            for wav_path in tqdm(sampled_files, desc="Extracting embeddings"):
                try:
                    # Load wav file
                    waveform, sr = torchaudio.load(wav_path)
                    
                    # Resample if needed
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    
                    # Convert to mono if stereo
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    
                    # Store input for input-space similarity
                    input_flat = waveform.squeeze(0).cpu().numpy()
                    inputs.append(input_flat)
                    
                    # Prepare input: [batch, seq_len]
                    data = waveform.squeeze(0).to(device)  # [seq_len]
                    data = data.unsqueeze(0)  # [1, seq_len]
                    
                    # Get features
                    result = model.extract_features(data, padding_mask=None, mask=False)
                    emb = result["x"].mean(dim=1).cpu().numpy()  # [1, embed_dim]
                    embeddings.append(emb.squeeze())
                except Exception as e:
                    print(f"[!] Error processing {wav_path}: {e}")
                    continue
        
        if len(embeddings) < 2:
            return {"error": "Not enough valid samples for similarity computation"}
        
        # Stack arrays
        inputs = np.stack(inputs)
        embeddings = np.stack(embeddings)
        
        # Compute similarity matrices
        input_sim_matrix = cosine_similarity(inputs)
        emb_sim_matrix = cosine_similarity(embeddings)
        
        # Get upper triangle indices (excluding diagonal)
        triu_idx = np.triu_indices_from(input_sim_matrix, k=1)
        
        input_sims = input_sim_matrix[triu_idx]
        emb_sims = emb_sim_matrix[triu_idx]
        
        # Compute correlation between input and embedding similarities
        pearson_corr, pearson_p = pearsonr(input_sims, emb_sims)
        spearman_corr, spearman_p = spearmanr(input_sims, emb_sims)
        
        # Metrics
        metrics = {
            # Input space metrics
            "input_mean_sim": float(np.mean(input_sims)),
            "input_std_sim": float(np.std(input_sims)),
            
            # Embedding space metrics
            "emb_mean_sim": float(np.mean(emb_sims)),
            "emb_std_sim": float(np.std(emb_sims)),
            
            # Comparison metrics (key indicators of model quality)
            "pearson_corr": float(pearson_corr),  # How well embedding similarity tracks input similarity
            "pearson_p_value": float(pearson_p),
            "spearman_corr": float(spearman_corr),  # Rank correlation
            "spearman_p_value": float(spearman_p),
            
            # Similarity ratio (embedding variance / input variance)
            # Low ratio suggests mode collapse
            "sim_variance_ratio": float(np.std(emb_sims) / (np.std(input_sims) + 1e-8)),
            
            # Model info
            "emb_dim": embeddings.shape[1],
            "num_samples": len(embeddings),
            "num_pairs": len(input_sims),
        }
        
        print(f"\n[+] Embedding Similarity Analysis:")
        print(f"    Input space:  mean={metrics['input_mean_sim']:.4f}, std={metrics['input_std_sim']:.4f}")
        print(f"    Embed space:  mean={metrics['emb_mean_sim']:.4f}, std={metrics['emb_std_sim']:.4f}")
        print(f"    Pearson corr: {metrics['pearson_corr']:.4f} (p={metrics['pearson_p_value']:.2e})")
        print(f"    Spearman corr: {metrics['spearman_corr']:.4f}")
        print(f"    Variance ratio: {metrics['sim_variance_ratio']:.4f}")
        
        return metrics
    
    def load_eval_dataset_fairseq(self, cfg, split="valid", max_samples=None, verbose=True):
        """
        Load evaluation data using fairseq's task.load_dataset and __getitem__.
        This ensures evaluation uses the exact same data loading as training.
        
        Args:
            cfg: The saved config from the checkpoint
            split: Dataset split to load ("valid" or "train")
            max_samples: Maximum samples to load for efficiency (None = all)
            verbose: Whether to print loading messages
            
        Returns:
            task: The fairseq task
            samples: List of samples (if max_samples specified), else None
            dataset: The fairseq dataset
        """
        from fairseq import tasks
        
        # Build task from config
        task = tasks.setup_task(cfg.task)
        
        # Load dataset using the same method as training
        task.load_dataset(split, task_cfg=cfg.task)
        dataset = task.dataset(split)
        
        if verbose:
            print(f"[+] Loaded {len(dataset)} samples from fairseq dataset")
        
        # If max_samples specified, load individual samples
        samples = None
        if max_samples is not None:
            sample_size = min(max_samples, len(dataset))
            indices = list(range(sample_size))
            
            samples = []
            for idx in tqdm(indices, desc=f"Loading {split} data"):
                # This calls FileAudioDataset.__getitem__()
                sample = dataset[idx]
                samples.append(sample)
        
        return task, samples, dataset
    
    def _eval_validation_loss(self, model, cfg) -> Dict[str, float]:
        """
        Evaluate the model using trainer.valid_step (same as train.py's validate function).
        
        This mimics the exact validation done during training:
        - Uses trainer.get_valid_iterator() for batching
        - Uses trainer.valid_step(sample) which handles GPU, metrics, etc.
        
        Returns:
            Dictionary with validation loss metrics
        """
        from fairseq.trainer import Trainer
        from fairseq.logging import metrics as fairseq_metrics
        
        print("[+] Evaluating validation loss using trainer.valid_step...")
        
        try:
            print("[DEBUG] Step 1: Loading dataset...")
            # Load dataset using our helper (allows custom paths later)
            # Note: Prints may come from:
            # 1. model.build_model() if model_path is set in config (data2vec_audio.py lines 243-261)
            # 2. fairseq's internal task/trainer initialization
            # 3. load_eval_dataset_fairseq (now suppressed with verbose=False)
            task, _, dataset = self.load_eval_dataset_fairseq(cfg, split="valid", max_samples=None, verbose=False)
            print(f"[DEBUG] Step 1 complete: Loaded dataset with {len(dataset)} samples")
            
            print("[DEBUG] Step 2: Building criterion...")
            # Build criterion
            criterion = task.build_criterion(cfg.criterion)
            print("[DEBUG] Step 2 complete: Criterion built")
            
            print("[DEBUG] Step 3: Creating trainer...")
            # Ensure model is on the correct device
            # Trainer will move data to its device, so model must match
            # Check what device Trainer will use (same logic as Trainer.__init__)
            use_cuda = torch.cuda.is_available() and not getattr(cfg.common, 'cpu', False)
            trainer_device = torch.device("cuda" if use_cuda else "cpu")
            model_device = next(model.parameters()).device
            print(f"[DEBUG] Trainer will use device: {trainer_device}, Model currently on: {model_device}")
            print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}, cfg.common.cpu: {getattr(cfg.common, 'cpu', 'not set')}")
            
            if model_device != trainer_device:
                print(f"[DEBUG] Moving model from {model_device} to {trainer_device}")
                model = model.to(trainer_device)
                # Verify move was successful
                new_model_device = next(model.parameters()).device
                print(f"[DEBUG] Model now on device: {new_model_device}")
            
            # Ensure EMA model is also on the correct device
            # EMA model is stored separately in model.ema.model and might not be moved with model.to()
            if hasattr(model, 'ema') and model.ema is not None and hasattr(model.ema, 'model') and model.ema.model is not None:
                try:
                    ema_model_device = next(model.ema.model.parameters()).device
                    print(f"[DEBUG] EMA model device before: {ema_model_device}")
                    if ema_model_device != trainer_device:
                        print(f"[DEBUG] Moving EMA model from {ema_model_device} to {trainer_device}")
                        model.ema.model = model.ema.model.to(trainer_device)
                        # Verify EMA move
                        new_ema_device = next(model.ema.model.parameters()).device
                        print(f"[DEBUG] EMA model now on device: {new_ema_device}")
                except Exception as e:
                    print(f"[DEBUG] Could not check/move EMA model: {e}")
            
            # Create trainer (same as in training)
            trainer = Trainer(cfg, task, model, criterion)
            print(f"[DEBUG] Trainer created with device: {trainer.device}")
            print("[DEBUG] Step 3 complete: Trainer created")
            
            print("[DEBUG] Step 4: Getting valid iterator...")
            # Get valid iterator (same as train.py validate function)
            itr = trainer.get_valid_iterator("valid").next_epoch_itr(shuffle=False)
            print("[DEBUG] Step 4 complete: Iterator obtained")
            
            print(f"[+] Running validation...")
            
            all_losses = []
            num_batches = 0
            
            # Validation loop (same as train.py validate function)
            with fairseq_metrics.aggregate(new_root=True) as agg:
                for sample in tqdm(itr, desc="Computing validation loss"):
                    # trainer.valid_step handles: move to GPU, model.eval(), torch.no_grad()
                    log_output = trainer.valid_step(sample)
                    
                    if log_output:
                        # Print full log_output structure for first batch (debugging)
                        if num_batches == 0:
                            print(f"\n[DEBUG] First batch log_output keys: {list(log_output.keys())}")
                            print(f"[DEBUG] Full log_output: {log_output}")
                        
                        loss_val = log_output.get("loss", 0)
                        sample_size = log_output.get("sample_size", 0)
                        nsentences = log_output.get("nsentences", 0)
                        
                        # Print detailed loss info for debugging
                        print(f"  Batch {num_batches + 1}: loss={loss_val:.6f}, sample_size={sample_size}, nsentences={nsentences}")
                        if "regression" in log_output:
                            print(f"    regression loss: {log_output.get('regression', 0):.6f}")
                        
                        all_losses.append(float(loss_val))
                        num_batches += 1
                    else:
                        print(f"  Batch {num_batches + 1}: log_output is None or empty")
                
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
    
    def _eval_signal_completion(self, model) -> Dict[str, float]:
        """
        Evaluate signal completion/reconstruction capability.
        
        Tests how well the model can predict masked portions of signals by:
        1. Getting embeddings for full (unmasked) signals as ground truth
        2. Masking portions of signals (second half for causal, random spans)
        3. Comparing model's contextualized representations at masked positions
        
        Key metrics:
        - Cosine similarity between predicted and target embeddings at masked positions
        - MSE loss at masked positions
        - Comparison across different masking strategies (causal vs random)
        """
        from scipy.stats import pearsonr, spearmanr
        
        device = next(model.parameters()).device
        model.eval()
        
        # Load dataset using fairseq infrastructure
        _, samples, dataset = self.load_eval_dataset_fairseq(self._current_cfg, split="valid", max_samples=100)
        sample_size = len(samples)
        
        print(f"[+] Evaluating signal completion on {sample_size} samples...")
        
        # Results storage
        completion_results = []
        
        masking_strategies = {
            "causal_50": {"type": "causal", "ratio": 0.5},  # Mask second half
            "causal_25": {"type": "causal", "ratio": 0.25},  # Mask last quarter
            "random_30": {"type": "random", "ratio": 0.3},   # Random 30% masking
            "random_50": {"type": "random", "ratio": 0.5},   # Random 50% masking
        }
        
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
                    
                    sample_result = {
                        "index": idx,
                        "sample_id": int(sample_id),
                        "seq_len": gt_embeddings.shape[1],
                    }
                    
                    # Test each masking strategy
                    for strategy_name, strategy_config in masking_strategies.items():
                        mask_type = strategy_config["type"]
                        mask_ratio = strategy_config["ratio"]
                        
                        T = gt_embeddings.shape[1]  # Sequence length after feature extraction
                        
                        # Create mask indices
                        if mask_type == "causal":
                            # Mask the last portion (simulate completion task)
                            mask_start = int(T * (1 - mask_ratio))
                            mask_indices = torch.zeros(1, T, dtype=torch.bool, device=device)
                            mask_indices[0, mask_start:] = True
                        else:  # random
                            # Random masking
                            num_masked = int(T * mask_ratio)
                            perm = torch.randperm(T, device=device)[:num_masked]
                            mask_indices = torch.zeros(1, T, dtype=torch.bool, device=device)
                            mask_indices[0, perm] = True
                        
                        # Get masked embeddings
                        # The model applies masking internally, so we use mask=True
                        # But we need custom mask_indices to control what's masked
                        masked_result = model.forward(
                            data, 
                            padding_mask=None, 
                            mask=True,
                            features_only=True,
                            mask_indices=mask_indices.cpu().numpy()
                        )
                        masked_embeddings = masked_result["x"]  # [1, T, embed_dim]
                        
                        # Compute metrics at masked positions
                        gt_masked = gt_embeddings[mask_indices]  # [num_masked, embed_dim]
                        pred_masked = masked_embeddings[mask_indices]  # [num_masked, embed_dim]
                        
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
                            
                            sample_result[f"{strategy_name}_cos_sim"] = cos_sim
                            sample_result[f"{strategy_name}_mse"] = mse
                            sample_result[f"{strategy_name}_l1"] = l1
                            sample_result[f"{strategy_name}_num_masked"] = mask_indices.sum().item()
                    
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
        
        # Aggregate metrics
        metrics = {}
        for strategy_name in masking_strategies.keys():
            cos_sim_col = f"{strategy_name}_cos_sim"
            mse_col = f"{strategy_name}_mse"
            l1_col = f"{strategy_name}_l1"
            
            if cos_sim_col in completion_df.columns:
                cos_sims = completion_df[cos_sim_col].dropna().values
                mse_vals = completion_df[mse_col].dropna().values
                l1_vals = completion_df[l1_col].dropna().values
                
                if len(cos_sims) > 0:
                    metrics[f"completion_{strategy_name}_cos_sim_mean"] = float(np.mean(cos_sims))
                    metrics[f"completion_{strategy_name}_cos_sim_std"] = float(np.std(cos_sims))
                    metrics[f"completion_{strategy_name}_mse_mean"] = float(np.mean(mse_vals))
                    metrics[f"completion_{strategy_name}_mse_std"] = float(np.std(mse_vals))
                    metrics[f"completion_{strategy_name}_l1_mean"] = float(np.mean(l1_vals))
        
        # Overall completion score (average cosine similarity across strategies)
        all_cos_sims = []
        for strategy_name in masking_strategies.keys():
            key = f"completion_{strategy_name}_cos_sim_mean"
            if key in metrics:
                all_cos_sims.append(metrics[key])
        
        if all_cos_sims:
            metrics["completion_overall_cos_sim"] = float(np.mean(all_cos_sims))
            # Convert to 0-100 score (cos sim ranges from -1 to 1)
            metrics["completion_score"] = float((np.mean(all_cos_sims) + 1) * 50)
        
        print(f"\n[+] Signal Completion Analysis:")
        for strategy_name in masking_strategies.keys():
            key = f"completion_{strategy_name}_cos_sim_mean"
            if key in metrics:
                print(f"    {strategy_name}: cos_sim={metrics[key]:.4f}, "
                      f"mse={metrics[f'completion_{strategy_name}_mse_mean']:.4f}")
        if "completion_score" in metrics:
            print(f"    Overall completion score: {metrics['completion_score']:.1f}/100")
        
        return metrics
    
    def _eval_noise_robustness(self, model) -> Dict[str, float]:
        """
        Evaluate embedding robustness to various noise types.
        Compares clean vs noisy embeddings using cosine similarity.
        Saves detailed noise data and generates noisy vs clean spectrograms.
        """
        import torchaudio
        import glob
        
        device = next(model.parameters()).device
        model.eval()
        
        # Load wav files
        wav_files = glob.glob(os.path.join(self.data_dir, "*.wav"))
        if len(wav_files) == 0:
            return {"error": f"No wav files found in {self.data_dir}"}
        
        # Sample for efficiency
        sample_size = min(50, len(wav_files))
        import random
        random.seed(42)
        sampled_files = random.sample(wav_files, sample_size)
        
        print(f"[+] Evaluating noise robustness on {sample_size} samples...")
        
        noise_types = {
            "gaussian_std": lambda x: x + np.random.normal(0, 0.01, size=len(x)),
            "gaussian_mean": lambda x: x + np.random.normal(0.02, 0.001, size=len(x)),
            "gain_low": lambda x: x * np.random.normal(1, 0.05),
            "gain_high": lambda x: x * np.random.normal(1, 0.1),
        }
        
        # Store per-sample results for detailed analysis
        noise_results = []
        
        with torch.no_grad():
            for idx, wav_path in enumerate(tqdm(sampled_files, desc="Noise robustness eval")):
                try:
                    # Load wav file
                    waveform, sr = torchaudio.load(wav_path)
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    
                    clean_data = waveform.squeeze(0).numpy()
                    
                    # Get clean embedding
                    data_tensor = waveform.squeeze(0).unsqueeze(0).to(device)
                    clean_result = model.extract_features(data_tensor, padding_mask=None, mask=False)
                    clean_emb = clean_result["x"].mean(dim=1).cpu().numpy().squeeze()
                    
                    sample_result = {
                        "index": idx,
                        "file": os.path.basename(wav_path),
                    }
                    
                    # Evaluate each noise type
                    for noise_type, noise_fn in noise_types.items():
                        noisy_data = noise_fn(clean_data).astype(np.float32)
                        noisy_tensor = torch.tensor(noisy_data).unsqueeze(0).to(device)
                        
                        noisy_result = model.extract_features(noisy_tensor, padding_mask=None, mask=False)
                        noisy_emb = noisy_result["x"].mean(dim=1).cpu().numpy().squeeze()
                        
                        # Compute similarities using helper function
                        data_sim = compute_cosine_similarity(clean_data, noisy_data)
                        emb_sim = compute_cosine_similarity(clean_emb, noisy_emb)
                        
                        sample_result[f"{noise_type}_data_sim"] = data_sim
                        sample_result[f"{noise_type}_emb_sim"] = emb_sim
                        
                        # Store noisy data for best/worst plotting (first few samples only)
                        if idx < 10:
                            if f"clean_data_{idx}" not in self.eval_data:
                                self.eval_data[f"clean_data_{self._current_run_name}_{idx}"] = clean_data
                            self.eval_data[f"noisy_data_{self._current_run_name}_{idx}_{noise_type}"] = noisy_data
                    
                    noise_results.append(sample_result)
                        
                except Exception as e:
                    print(f"[!] Error processing {wav_path}: {e}")
                    continue
        
        # Create noise results dataframe and save
        noise_df = pd.DataFrame(noise_results)
        noise_df_path = self.data_dir_out / f"noise_robustness_{self._current_run_name}.csv"
        noise_df.to_csv(noise_df_path, index=False)
        print(f"[+] Saved noise robustness data to: {noise_df_path}")
        
        # Store for visualization
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
                # Robustness ratio: how much better embedding preserves identity than raw data
                metrics[f"noise_{noise_type}_robustness_ratio"] = float(
                    np.mean(emb_sims) / (np.mean(data_sims) + 1e-8)
                )
        
        print(f"\n[+] Noise Robustness Analysis:")
        for noise_type in noise_types:
            if f"noise_{noise_type}_emb_sim_mean" in metrics:
                print(f"    {noise_type}: data_sim={metrics[f'noise_{noise_type}_data_sim_mean']:.4f}, "
                      f"emb_sim={metrics[f'noise_{noise_type}_emb_sim_mean']:.4f}, "
                      f"ratio={metrics[f'noise_{noise_type}_robustness_ratio']:.4f}")
        
        return metrics

    def _eval_stack_similarity(self, model) -> Dict[str, float]:
        """
        Evaluate similarity comparison by stack membership.
        Compares how well embedding-space neighbors match input-space neighbors,
        particularly for samples from the same "stack" (group).
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import torchaudio
        import glob
        
        device = next(model.parameters()).device
        model.eval()
        
        # Load wav files
        wav_files = sorted(glob.glob(os.path.join(self.data_dir, "*.wav")))
        if len(wav_files) == 0:
            return {"error": f"No wav files found in {self.data_dir}"}
        
        sample_size = min(100, len(wav_files))
        # Use first N samples to maintain stack structure (every 10 samples = 1 stack)
        sampled_files = wav_files[:sample_size]
        
        print(f"[+] Evaluating stack similarity on {sample_size} samples...")
        
        inputs = []
        embeddings = []
        stack_indices = []
        
        with torch.no_grad():
            for idx, wav_path in enumerate(tqdm(sampled_files, desc="Stack similarity eval")):
                try:
                    waveform, sr = torchaudio.load(wav_path)
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    
                    input_flat = waveform.squeeze(0).cpu().numpy()
                    inputs.append(input_flat)
                    
                    data = waveform.squeeze(0).unsqueeze(0).to(device)
                    result = model.extract_features(data, padding_mask=None, mask=False)
                    emb = result["x"].mean(dim=1).cpu().numpy().squeeze()
                    embeddings.append(emb)
                    
                    # Assign stack index (every 10 samples = 1 stack)
                    stack_indices.append(idx // 10)
                    
                except Exception as e:
                    print(f"[!] Error processing {wav_path}: {e}")
                    continue
        
        if len(embeddings) < 10:
            return {"error": "Not enough valid samples for stack similarity"}
        
        inputs = np.stack(inputs)
        embeddings = np.stack(embeddings)
        stack_indices = np.array(stack_indices)
        
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
                                  best_loss: Optional[float] = None,
                                  save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate and optionally save similarity matrix visualizations.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import torchaudio
        import glob
        
        # Load model
        model, cfg = load_fairseq_model_for_evaluation(checkpoint_info.path)
        device = next(model.parameters()).device
        model.eval()
        
        # Load samples
        wav_files = glob.glob(os.path.join(self.data_dir, "*.wav"))[:50]
        
        inputs = []
        embeddings = []
        
        with torch.no_grad():
            for wav_path in tqdm(wav_files, desc="Loading for visualization"):
                try:
                    waveform, sr = torchaudio.load(wav_path)
                    if sr != 16000:
                        waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    
                    inputs.append(waveform.squeeze(0).cpu().numpy())
                    
                    data = waveform.squeeze(0).unsqueeze(0).to(device)
                    result = model.extract_features(data, padding_mask=None, mask=False)
                    emb = result["x"].mean(dim=1).cpu().numpy().squeeze()
                    embeddings.append(emb)
                except Exception:
                    continue
        
        if len(embeddings) < 2:
            return {}
        
        inputs = np.stack(inputs)
        embeddings = np.stack(embeddings)
        
        input_sim = cosine_similarity(inputs)
        emb_sim = cosine_similarity(embeddings)
        
        loss_str = f" | Training Loss: {best_loss:.4f}" if best_loss is not None else ""
        
        if save_plots:
            run_plots_dir = self.plots_dir / checkpoint_info.run_name
            run_plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot input similarity matrix
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"Run: {checkpoint_info.run_name}{loss_str}", fontsize=12, fontweight='bold')
            
            sns.heatmap(input_sim, ax=axes[0], cmap="viridis", 
                       xticklabels=False, yticklabels=False)
            axes[0].set_title("Input Space Cosine Similarity")
            axes[0].set_xlabel("Sample Index")
            axes[0].set_ylabel("Sample Index")
            
            sns.heatmap(emb_sim, ax=axes[1], cmap="viridis",
                       xticklabels=False, yticklabels=False)
            axes[1].set_title("Embedding Space Cosine Similarity")
            axes[1].set_xlabel("Sample Index")
            axes[1].set_ylabel("Sample Index")
            
            plt.tight_layout()
            plt.savefig(run_plots_dir / "similarity_matrices.png", dpi=150)
            plt.close()
            
            print(f"[+] Saved similarity matrix plot to {run_plots_dir}")
        
        return {
            "input_sim_matrix": input_sim,
            "emb_sim_matrix": emb_sim,
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
            
            # Debug: print all metric keys to help diagnose
            if not any(key.startswith("eval_") for key in m.keys()):
                print(f"[DEBUG] Run {result.run_name}: No eval_* keys found. Available keys: {list(m.keys())[:10]}...")
            
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
        
        if metric_cols:
            print("\nMetrics Overview:")
            for col in metric_cols:
                metric_name = col.replace("metric_", "")
                values = df[col].dropna()
                if len(values) > 0 and values.dtype in ['float64', 'int64']:
                    print(f"  {metric_name}:")
                    print(f"    Mean: {values.mean():.4f}")
                    print(f"    Std:  {values.std():.4f}")
                    print(f"    Best: {values.max():.4f} (run: {df.loc[values.idxmax(), 'run_name']})")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SpectralFM Evaluation Runner")
    
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="/mnt5/noy/fairseq/outputs",
                       help="Base directory for fairseq outputs")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to specific checkpoint file")
    parser.add_argument("--output_dir", type=str, 
                       default="/mnt5/noy/SpectralFM/code/eval_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--data_dir", type=str,
                       default="/mnt5/noy/fairseq/data/single_channel_1m/",
                       help="Directory containing evaluation data")
    parser.add_argument("--eval_methods", type=str, nargs="+",
                       default=["embedding_similarity"],
                       help="Evaluation methods to run: embedding_similarity, noise_robustness, stack_similarity, signal_completion, validation_loss")
    parser.add_argument("--best_only", action="store_true",
                       help="Only evaluate checkpoint_best.pt files")
    parser.add_argument("--latest_only", action="store_true",
                       help="Only evaluate the most recent checkpoint")
    parser.add_argument("--report_name", type=str, default=None,
                       help="Custom name for the report")
    parser.add_argument("--plot_matrices", action="store_true",
                       help="Generate similarity matrix visualizations")
    parser.add_argument("--all_methods", action="store_true",
                       help="Run all available evaluation methods")
    
    args = parser.parse_args()
    
    if args.report_name is None:
        args.report_name = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    
    if not checkpoints:
        print("[!] No checkpoints found!")
        return
    
    print(f"[+] Found {len(checkpoints)} checkpoint(s)")
    for ckpt in checkpoints:
        print(f"    - {ckpt.run_name} ({ckpt.checkpoint_type})")
    
    print(f"[+] Running evaluation methods: {args.eval_methods}")
    print(f"[DEBUG] args.eval_methods type: {type(args.eval_methods)}, value: {args.eval_methods}")
    
    # Run evaluations
    runner = EvaluationRunner(args.output_dir, args.data_dir)
    print(f"[DEBUG] About to call evaluate_all with eval_methods: {args.eval_methods}")
    results = runner.evaluate_all(checkpoints, args.eval_methods)
    
    # Create lookup for best_loss by run_name
    loss_lookup = {r.run_name: r.metrics.get("best_loss") for r in results}
    
    # Generate visualizations for each checkpoint
    print("\n[+] Generating visualizations...")
    for ckpt in checkpoints:
        try:
            run_name = ckpt.run_name
            best_loss = loss_lookup.get(run_name)
            
            # Similarity matrices (if requested or all_methods)
            if args.plot_matrices or args.all_methods:
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
        if "noise_robustness" in args.eval_methods:
            runner.plot_noise_robustness_comparison(results, save_plots=True)
        if "stack_similarity" in args.eval_methods:
            runner.plot_stack_similarity_comparison(results, save_plots=True)
        if "signal_completion" in args.eval_methods:
            runner.plot_signal_completion_comparison(results, save_plots=True)
    
    # Generate report
    reporter = ReportGenerator(args.output_dir)
    df = reporter.generate_comparison_report(results, args.report_name)
    
    # Generate comprehensive summary
    runner.generate_evaluation_summary(results, args.report_name + "_summary")
    
    # Generate markdown comparison report with images
    runner.generate_comparison_report_with_images(results, args.report_name + "_comparison")
    
    print(f"\n[+] Evaluation complete!")
    print(f"[+] Results saved to: {args.output_dir}")
    print(f"[+] Plots saved to: {runner.plots_dir}")
    print(f"[+] Data saved to: {runner.data_dir_out}")
    print(f"[+] Summary: {args.output_dir}/{args.report_name}_summary.txt")
    print(f"[+] Comparison Report: {args.output_dir}/{args.report_name}_comparison.md")


if __name__ == "__main__":
    main()
