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
from tqdm import tqdm

# Add fairseq to path using relative path from this file's location
_FAIRSEQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fairseq")
if _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)

from model_loader import load_fairseq_checkpoint, load_fairseq_model_for_evaluation
from omegaconf import OmegaConf


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
        
        # Walk through date directories
        for date_dir in sorted(self.base_dir.iterdir()):
            if not date_dir.is_dir() or not self._is_date_dir(date_dir.name):
                continue
                
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
        if eval_methods is None:
            eval_methods = ["embedding_similarity"]
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {checkpoint_info.run_name}")
        print(f"Checkpoint: {checkpoint_info.checkpoint_type}")
        print(f"{'='*60}")
        
        # Load model
        try:
            model, cfg = load_fairseq_model_for_evaluation(checkpoint_info.path)
        except Exception as e:
            print(f"[!] Failed to load checkpoint: {e}")
            return EvalResult(
                checkpoint_path=checkpoint_info.path,
                run_name=checkpoint_info.run_name,
                timestamp=datetime.now().isoformat(),
                metrics={"error": str(e)}
            )
        
        metrics = {}
        
        # Run each evaluation method
        for method in eval_methods:
            try:
                if method == "embedding_similarity":
                    method_metrics = self._eval_embedding_similarity(model)
                elif method == "signal_completion":
                    method_metrics = self._eval_signal_completion(model)
                elif method == "noise_robustness":
                    method_metrics = self._eval_noise_robustness(model)
                else:
                    print(f"[!] Unknown eval method: {method}")
                    continue
                    
                metrics.update(method_metrics)
            except Exception as e:
                print(f"[!] Error in {method}: {e}")
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
        
        self.results.append(result)
        return result
    
    def evaluate_all(self, checkpoints: List[CheckpointInfo], 
                    eval_methods: List[str] = None) -> List[EvalResult]:
        """Evaluate multiple checkpoints."""
        results = []
        for ckpt in tqdm(checkpoints, desc="Evaluating checkpoints"):
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
    
    def _eval_signal_completion(self, model) -> Dict[str, float]:
        """Evaluate signal completion/reconstruction capability."""
        return {"signal_completion": 0.0}
    
    def _eval_noise_robustness(self, model) -> Dict[str, float]:
        """Evaluate robustness to noise."""
        return {"noise_robustness": 0.0}
    
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
                       help="Evaluation methods to run")
    parser.add_argument("--best_only", action="store_true",
                       help="Only evaluate checkpoint_best.pt files")
    parser.add_argument("--latest_only", action="store_true",
                       help="Only evaluate the most recent checkpoint")
    parser.add_argument("--report_name", type=str, default=None,
                       help="Custom name for the report")
    
    args = parser.parse_args()
    
    if args.report_name is None:
        args.report_name = f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    
    # Run evaluations
    runner = EvaluationRunner(args.output_dir, args.data_dir)
    results = runner.evaluate_all(checkpoints, args.eval_methods)
    
    # Generate report
    reporter = ReportGenerator(args.output_dir)
    df = reporter.generate_comparison_report(results, args.report_name)
    
    print(f"\n[+] Evaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
