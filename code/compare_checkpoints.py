"""
Script to evaluate multiple checkpoints and generate comparison visualizations.

Usage:
    python compare_checkpoints.py --checkpoints <path1> <path2> ... --custom_dataset_path <path>
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add fairseq to path
_FAIRSEQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fairseq")
if _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)

from evaluation_runner import (
    EvaluationRunner, 
    CheckpointInfo, 
    EvalResult
)


def create_checkpoint_info(checkpoint_path: str) -> CheckpointInfo:
    """Create a CheckpointInfo object from a checkpoint path (file or directory)."""
    path = Path(checkpoint_path)
    
    # If it's a directory, look for checkpoint_best.pt inside
    if path.is_dir():
        checkpoint_file = path / "checkpoint_best.pt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"checkpoint_best.pt not found in directory: {checkpoint_path}")
        path = checkpoint_file
    elif not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Extract date and time from path: .../2026-01-07_21-50-07/checkpoint_best.pt
    parent_dir = path.parent.name
    if "_" in parent_dir:
        date_part, time_part = parent_dir.split("_", 1)
        date = date_part
        time = time_part.replace("-", ":")
    else:
        date = datetime.now().strftime("%Y-%m-%d")
        time = datetime.now().strftime("%H:%M:%S")
    
    return CheckpointInfo(
        path=str(path),
        run_dir=str(path.parent.parent),
        date=date,
        time=time,
        checkpoint_type="best",
        config={}
    )


def plot_validation_loss_comparison(results: List[EvalResult], output_dir: Path):
    """Create a comparison plot of validation loss vs best loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    run_names = []
    best_losses = []
    validation_losses = []
    
    for result in results:
        run_name = result.run_name
        best_loss = result.metrics.get("best_loss")
        validation_loss = result.metrics.get("eval_loss")  # Use eval_loss from _eval_validation_loss
        
        if best_loss is not None and validation_loss is not None:
            run_names.append(run_name)
            best_losses.append(best_loss)
            validation_losses.append(validation_loss)
    
    if not run_names:
        print("[!] No validation loss data available for comparison")
        return
    
    # Bar plot comparison
    x = np.arange(len(run_names))
    width = 0.35
    
    axes[0].bar(x - width/2, best_losses, width, label='Best Loss (Training)', alpha=0.8)
    axes[0].bar(x + width/2, validation_losses, width, label='Validation Loss', alpha=0.8)
    axes[0].set_xlabel('Run', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Best Loss vs Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(run_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(best_losses, validation_losses, s=100, alpha=0.6)
    for i, name in enumerate(run_names):
        axes[1].annotate(name, (best_losses[i], validation_losses[i]), 
                        fontsize=8, alpha=0.7)
    axes[1].plot([min(best_losses + validation_losses), max(best_losses + validation_losses)],
                [min(best_losses + validation_losses), max(best_losses + validation_losses)],
                'r--', alpha=0.5, label='y=x')
    axes[1].set_xlabel('Best Loss (Training)', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title('Best Loss vs Validation Loss (Scatter)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "validation_loss_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved validation loss comparison to: {output_path}")


def plot_embedding_metrics_comparison(results: List[EvalResult], output_dir: Path):
    """Create comparison plots for embedding metrics."""
    # Collect metrics
    metrics_data = []
    
    for result in results:
        row = {"run_name": result.run_name}
        
        # Best loss
        row["best_loss"] = result.metrics.get("best_loss")
        row["validation_loss"] = result.metrics.get("eval_loss")  # Use eval_loss from _eval_validation_loss
        
        # Valid dataset metrics
        row["valid_pearson_corr"] = result.metrics.get("valid_pearson_corr")
        row["valid_spearman_corr"] = result.metrics.get("valid_spearman_corr")
        row["valid_sim_variance_ratio"] = result.metrics.get("valid_sim_variance_ratio")
        row["valid_emb_mean_sim"] = result.metrics.get("valid_emb_mean_sim")
        row["valid_emb_std_sim"] = result.metrics.get("valid_emb_std_sim")
        
        # Custom dataset metrics (if available)
        row["custom_pearson_corr"] = result.metrics.get("custom_pearson_corr")
        row["custom_spearman_corr"] = result.metrics.get("custom_spearman_corr")
        row["custom_sim_variance_ratio"] = result.metrics.get("custom_sim_variance_ratio")
        row["custom_emb_mean_sim"] = result.metrics.get("custom_emb_mean_sim")
        row["custom_emb_std_sim"] = result.metrics.get("custom_emb_std_sim")
        
        metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Key metrics to plot
    metric_configs = [
        ("valid_pearson_corr", "Pearson Correlation (Valid)", "Higher is better"),
        ("valid_spearman_corr", "Spearman Correlation (Valid)", "Higher is better"),
        ("valid_sim_variance_ratio", "Variance Ratio (Valid)", "Higher is better"),
        ("custom_pearson_corr", "Pearson Correlation (Custom)", "Higher is better"),
        ("custom_spearman_corr", "Spearman Correlation (Custom)", "Higher is better"),
        ("custom_sim_variance_ratio", "Variance Ratio (Custom)", "Higher is better"),
    ]
    
    for idx, (metric_key, title, note) in enumerate(metric_configs):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        values = df[metric_key].dropna().tolist()
        run_names = df.loc[df[metric_key].notna(), "run_name"].tolist()
        
        if values:
            bars = ax.bar(range(len(values)), values, alpha=0.7)
            ax.set_xticks(range(len(run_names)))
            ax.set_xticklabels(run_names, rotation=45, ha='right')
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(f"{title}\n{note}", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'No data for {metric_key}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "embedding_metrics_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved embedding metrics comparison to: {output_path}")
    
    # Save metrics table
    csv_path = output_dir / "embedding_metrics_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved embedding metrics table to: {csv_path}")


def _extract_fe_outputs_from_checkpoint(
    checkpoint_path: str,
    samples: list,
    device,
    checkpoint_name: str = "",
) -> Optional[np.ndarray]:
    """
    Extract CNN feature extractor output vectors from a fairseq checkpoint.

    Runs each sample through the CNN frontend (feature_extractor), applies the
    post-CNN layer norm, and optionally post_extract_proj, then mean-pools over
    the time dimension.  Returns an array of shape (N, D).
    """
    from model_loader import load_fairseq_checkpoint
    import torch

    try:
        model, _, _ = load_fairseq_checkpoint(checkpoint_path)
        model = model.to(device)
        model.eval()

        fe_outputs = []
        with torch.no_grad():
            for sample in samples:
                if isinstance(sample, dict):
                    source = sample.get("source") or (
                        sample.get("net_input", {}).get("source")
                    )
                else:
                    source = sample

                if source is None:
                    continue

                if not isinstance(source, torch.Tensor):
                    source = torch.tensor(source, dtype=torch.float32)

                if source.dim() == 1:
                    source = source.unsqueeze(0)  # [1, T]
                source = source.to(device)

                fe = model.feature_extractor(source)   # [1, 512, T']
                fe = fe.transpose(1, 2)                # [1, T', 512]
                fe = model.layer_norm(fe)              # [1, T', 512]
                if model.post_extract_proj is not None:
                    fe = model.post_extract_proj(fe)   # [1, T', D]
                fe_vec = fe.mean(dim=1)                # [1, D]
                fe_outputs.append(fe_vec.squeeze(0).cpu().numpy())

        if fe_outputs:
            return np.stack(fe_outputs)
        return None
    except Exception as e:
        print(f"    [!] Error extracting FE outputs from {checkpoint_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_side_by_side_plots(checkpoint_infos: List[CheckpointInfo], runner: EvaluationRunner, 
                               output_dir: Path, custom_dataset_path: Optional[str] = None):
    """
    Create comparison plots showing cosine similarity matrices for each checkpoint.
    Creates 2 plots: one for valid dataset, one for custom dataset.
    Each plot has two rows per checkpoint: embeddings (top) and CNN feature
    extractor outputs (bottom).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from model_loader import load_fairseq_checkpoint
    import torch
    
    # Use the runner's plots_dir (timestamped evaluation run directory) instead of comparison_plots subdirectory
    # This matches where the histogram comparison plots are saved
    plots_dir = runner.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[+] Creating embedding similarity comparison plots...")
    
    # Use pre-loaded valid samples from runner if available (from evaluate_all/evaluate_checkpoint)
    # This ensures the same samples are used in evaluation and comparison plots
    samples_valid = runner._preloaded_samples_valid
    
    if samples_valid is None:
        print("[!] Warning: No pre-loaded valid samples found in runner.")
        print("[!] This should not happen if side-by-side plots are called from evaluation_runner.")
        print("[!] Falling back to loading samples, but they may not match evaluation samples...")
        # Use the first checkpoint's config to load the dataset
        if not checkpoint_infos:
            print("[!] No checkpoints provided")
            return
        
        from model_loader import load_fairseq_checkpoint
        model, model_cfg, checkpoint_info_loaded = load_fairseq_checkpoint(checkpoint_infos[0].path)
        cfg = checkpoint_info_loaded["cfg"]
        
        # Use the same loading logic as evaluate_all: fixed seed, same max_samples
        task_valid, _, dataset_valid = runner.load_eval_dataset_fairseq(
            cfg, split="valid", max_samples=None, verbose=True
        )
        
        # Randomly select 100 samples with fixed seed (same as evaluate_all)
        dataset_size = len(dataset_valid)
        num_samples = min(100, dataset_size)
        
        if dataset_size > 0:
            print(f"[+] Randomly selecting {num_samples} samples from {dataset_size} total samples...")
            random.seed(42)  # Fixed seed for reproducibility (same as evaluate_all)
            
            # Randomly select indices
            selected_indices = random.sample(range(dataset_size), num_samples)
            selected_indices.sort()  # Sort for easier debugging/tracking
            
            # Load only the selected samples
            from tqdm import tqdm
            samples_valid = []
            for idx in tqdm(selected_indices, desc=f"Loading {num_samples} selected samples"):
                sample = dataset_valid[idx]
                samples_valid.append(sample)
            
            print(f"[+] Loaded {len(samples_valid)} valid samples (indices: {selected_indices[:5]}...{selected_indices[-5:]})")
        else:
            print("[!] Failed to load valid dataset samples")
            return
    else:
        print(f"[+] Using pre-loaded valid samples from evaluation: {len(samples_valid)} samples")
        if runner._preloaded_indices_valid is not None:
            print(f"[+] Using the same selected indices: {runner._preloaded_indices_valid[:5]}...{runner._preloaded_indices_valid[-5:]}")
    
    # Use pre-loaded custom samples from runner if available (from evaluate_all)
    # This ensures the same samples are used in evaluation and comparison plots
    samples_custom = runner._preloaded_samples_custom
    custom_dataset_name = None
    
    if custom_dataset_path:
        custom_dataset_name = Path(custom_dataset_path).name
        
        if samples_custom is None:
            print(f"[!] Warning: No pre-loaded custom samples found in runner.")
            print(f"[!] Falling back to loading custom dataset: {custom_dataset_path}...")
            # Use the first checkpoint's config if not already loaded
            if 'cfg' not in locals():
                from model_loader import load_fairseq_checkpoint
                model, model_cfg, checkpoint_info_loaded = load_fairseq_checkpoint(checkpoint_infos[0].path)
                cfg = checkpoint_info_loaded["cfg"]
            
            # First, load the full dataset to get its size
            task_custom, _, dataset_custom = runner.load_eval_dataset_fairseq(
                cfg, split="valid", max_samples=None, verbose=True,
                custom_dataset_path=custom_dataset_path
            )
            
            # Randomly select 100 samples with fixed seed (same as evaluate_all)
            dataset_size = len(dataset_custom)
            num_samples = min(100, dataset_size)
            
            if dataset_size > 0:
                print(f"[+] Randomly selecting {num_samples} samples from {dataset_size} total samples...")
                random.seed(42)  # Fixed seed for reproducibility (same as evaluate_all)
                
                # Randomly select indices
                selected_indices = random.sample(range(dataset_size), num_samples)
                selected_indices.sort()  # Sort for easier debugging/tracking
                
                # Load only the selected samples
                from tqdm import tqdm
                samples_custom = []
                for idx in tqdm(selected_indices, desc=f"Loading {num_samples} selected samples"):
                    sample = dataset_custom[idx]
                    samples_custom.append(sample)
                
                print(f"[+] Loaded {len(samples_custom)} samples (indices: {selected_indices[:5]}...{selected_indices[-5:]})")
            else:
                print(f"[!] Custom dataset is empty!")
                samples_custom = None
        else:
            print(f"[+] Using pre-loaded custom samples from evaluation: {len(samples_custom)} samples")
            if runner._preloaded_indices_custom is not None:
                print(f"[+] Using the same selected indices: {runner._preloaded_indices_custom[:5]}...{runner._preloaded_indices_custom[-5:]}")
    
    # Determine device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Extract embeddings and FE outputs from each checkpoint for valid dataset
    # Use stored embeddings from evaluation if available (ensures exact same embeddings)
    print("[+] Getting embeddings from checkpoints for valid dataset...")
    valid_embeddings_list = []
    valid_fe_list = []
    valid_run_names = []
    
    for ckpt_info in checkpoint_infos:
        run_name = ckpt_info.run_name
        try:
            print(f"    Loading {run_name}...")
            
            # First try to get stored embeddings from evaluation (exact same as used in evaluation)
            embeddings_key = f'embeddings_{run_name}_valid'
            embeddings = None
            
            if embeddings_key in runner.eval_data:
                embeddings = runner.eval_data[embeddings_key]
                print(f"        Using stored embeddings from evaluation: {embeddings.shape}")
            else:
                # Fallback: try loading from numpy file
                embeddings_path = runner.data_dir_out / f"embeddings_{run_name}_valid.npy"
                if embeddings_path.exists():
                    try:
                        embeddings = np.load(embeddings_path)
                        print(f"        Loaded embeddings from file: {embeddings.shape}")
                    except Exception as e:
                        print(f"        [!] Could not load embeddings from file: {e}")
            
            # Last resort: extract embeddings (may differ from evaluation)
            if embeddings is None:
                print(f"        [!] Stored embeddings not found, extracting from checkpoint (may differ from evaluation)...")
                embeddings = runner._load_embeddings_from_checkpoint(
                    ckpt_info.path, samples_valid, device, checkpoint_name=run_name
                )
            
            if embeddings is not None and len(embeddings) > 0:
                valid_embeddings_list.append(embeddings)
                valid_run_names.append(run_name)
            else:
                print(f"    [!] Failed to get/extract embeddings from {run_name}")
                continue

            # Extract FE outputs (try stored, then fresh extraction)
            fe_outputs = None
            fe_key = f'fe_outputs_{run_name}_valid'
            if fe_key in runner.eval_data:
                fe_outputs = runner.eval_data[fe_key]
                print(f"        Using stored FE outputs from evaluation: {fe_outputs.shape}")
            else:
                fe_path = runner.data_dir_out / f"fe_outputs_{run_name}_valid.npy"
                if fe_path.exists():
                    try:
                        fe_outputs = np.load(fe_path)
                        print(f"        Loaded FE outputs from file: {fe_outputs.shape}")
                    except Exception as e:
                        print(f"        [!] Could not load FE outputs from file: {e}")

            if fe_outputs is None:
                print(f"        Extracting FE outputs from checkpoint...")
                fe_outputs = _extract_fe_outputs_from_checkpoint(
                    ckpt_info.path, samples_valid, device, checkpoint_name=run_name
                )
            valid_fe_list.append(fe_outputs)

        except Exception as e:
            print(f"    [!] Error loading {run_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Extract embeddings and FE outputs from each checkpoint for custom dataset
    # Use stored embeddings from evaluation if available (ensures exact same embeddings)
    custom_embeddings_list = []
    custom_fe_list = []
    custom_run_names = []
    
    if samples_custom is not None and len(samples_custom) > 0:
        print(f"[+] Getting embeddings from checkpoints for custom dataset ({custom_dataset_name})...")
        for ckpt_info in checkpoint_infos:
            run_name = ckpt_info.run_name
            try:
                print(f"    Loading {run_name}...")
                
                # First try to get stored embeddings from evaluation (exact same as used in evaluation)
                embeddings_key = f'embeddings_{run_name}_{custom_dataset_name}'
                embeddings = None
                
                if embeddings_key in runner.eval_data:
                    embeddings = runner.eval_data[embeddings_key]
                    print(f"        Using stored embeddings from evaluation: {embeddings.shape}")
                else:
                    # Fallback: try loading from numpy file
                    embeddings_path = runner.data_dir_out / f"embeddings_{run_name}_{custom_dataset_name}.npy"
                    if embeddings_path.exists():
                        try:
                            embeddings = np.load(embeddings_path)
                            print(f"        Loaded embeddings from file: {embeddings.shape}")
                        except Exception as e:
                            print(f"        [!] Could not load embeddings from file: {e}")
                
                # Last resort: extract embeddings (may differ from evaluation)
                if embeddings is None:
                    print(f"        [!] Stored embeddings not found, extracting from checkpoint (may differ from evaluation)...")
                    embeddings = runner._load_embeddings_from_checkpoint(
                        ckpt_info.path, samples_custom, device, checkpoint_name=run_name
                    )
                
                if embeddings is not None and len(embeddings) > 0:
                    custom_embeddings_list.append(embeddings)
                    custom_run_names.append(run_name)
                else:
                    print(f"    [!] Failed to get/extract embeddings from {run_name}")
                    continue

                # Extract FE outputs (try stored, then fresh extraction)
                fe_outputs = None
                fe_key = f'fe_outputs_{run_name}_{custom_dataset_name}'
                if fe_key in runner.eval_data:
                    fe_outputs = runner.eval_data[fe_key]
                    print(f"        Using stored FE outputs from evaluation: {fe_outputs.shape}")
                else:
                    fe_path = runner.data_dir_out / f"fe_outputs_{run_name}_{custom_dataset_name}.npy"
                    if fe_path.exists():
                        try:
                            fe_outputs = np.load(fe_path)
                            print(f"        Loaded FE outputs from file: {fe_outputs.shape}")
                        except Exception as e:
                            print(f"        [!] Could not load FE outputs from file: {e}")

                if fe_outputs is None:
                    print(f"        Extracting FE outputs from checkpoint...")
                    fe_outputs = _extract_fe_outputs_from_checkpoint(
                        ckpt_info.path, samples_custom, device, checkpoint_name=run_name
                    )
                custom_fe_list.append(fe_outputs)

            except Exception as e:
                print(f"    [!] Error loading {run_name}: {e}")
                import traceback
                traceback.print_exc()

    def _plot_similarity_rows(embeddings_list, fe_list, run_names, title, output_path):
        """Plot a 2-row (or 1-row fallback) similarity heatmap figure."""
        n_models = len(embeddings_list)
        has_fe = any(fe is not None for fe in fe_list)
        n_rows = 2 if has_fe else 1
        fig, axes = plt.subplots(n_rows, n_models,
                                 figsize=(5 * n_models, 5 * n_rows),
                                 squeeze=False)

        for idx, (embeddings, run_name) in enumerate(zip(embeddings_list, run_names)):
            # --- Row 0: Embeddings ---
            sim_matrix = cosine_similarity(embeddings)
            triu = np.triu_indices_from(sim_matrix, k=1)
            sim_mean = np.mean(sim_matrix[triu])
            sim_std = np.std(sim_matrix[triu])

            ax = axes[0, idx]
            sns.heatmap(sim_matrix, ax=ax, cmap="viridis",
                        xticklabels=False, yticklabels=False, vmin=0, vmax=1)
            ax.set_title(f"{run_name}\n(mean={sim_mean:.3f}, std={sim_std:.3f})",
                         fontsize=11, fontweight='bold')
            ax.set_xlabel(f"Sample Index (N={len(embeddings)})", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Embeddings\nSample Index", fontsize=10)

            # --- Row 1: FE outputs ---
            if has_fe:
                fe = fe_list[idx] if idx < len(fe_list) else None
                ax_fe = axes[1, idx]
                if fe is not None and len(fe) > 1:
                    fe_sim = cosine_similarity(fe)
                    fe_triu = np.triu_indices_from(fe_sim, k=1)
                    fe_mean = np.mean(fe_sim[fe_triu])
                    fe_std = np.std(fe_sim[fe_triu])

                    sns.heatmap(fe_sim, ax=ax_fe, cmap="viridis",
                                xticklabels=False, yticklabels=False, vmin=0, vmax=1)
                    ax_fe.set_title(f"FE Output\n(mean={fe_mean:.3f}, std={fe_std:.3f})",
                                    fontsize=11)
                else:
                    ax_fe.text(0.5, 0.5, "FE outputs\nnot available",
                               ha='center', va='center', fontsize=11)
                    ax_fe.axis('off')

                ax_fe.set_xlabel(f"Sample Index (N={len(embeddings)})", fontsize=10)
                if idx == 0:
                    ax_fe.set_ylabel("FE Output\nSample Index", fontsize=10)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved comparison to: {output_path}")

    # Create plot for valid dataset
    if valid_embeddings_list:
        print("[+] Creating valid dataset comparison plot...")
        _plot_similarity_rows(
            valid_embeddings_list, valid_fe_list, valid_run_names,
            title="Embedding & FE Output Similarity Comparison - Valid Dataset",
            output_path=plots_dir / "embedding_similarity_comparison_valid.png",
        )
    
    # Create plot for custom dataset
    if custom_embeddings_list:
        print("[+] Creating custom dataset comparison plot...")
        _plot_similarity_rows(
            custom_embeddings_list, custom_fe_list, custom_run_names,
            title=f"Embedding & FE Output Similarity Comparison - {custom_dataset_name}",
            output_path=plots_dir / f"embedding_similarity_comparison_{custom_dataset_name}.png",
        )


def calculate_validation_loss_for_split(runner: EvaluationRunner, model, cfg, split: str, 
                                       custom_dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper function to calculate validation loss for a specific dataset split.
    """
    from omegaconf import OmegaConf
    
    # Save original cfg
    original_data = cfg.task.data
    
    # Temporarily modify cfg if custom_dataset_path is provided
    if custom_dataset_path:
        task_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task))
        task_cfg.data = custom_dataset_path
        cfg.task = task_cfg
    
    try:
        # Load dataset for the split
        task, _, _ = runner.load_eval_dataset_fairseq(
            cfg, split=split, max_samples=None, verbose=False,
            custom_dataset_path=custom_dataset_path
        )
        
        # Calculate validation loss
        val_metrics = runner._eval_validation_loss(model, cfg, debug=False)
        return val_metrics
    except Exception as e:
        print(f"    [!] Error: {e}")
        return {"eval_loss": "Error", "eval_loss_std": "N/A"}
    finally:
        # Restore original cfg
        if custom_dataset_path:
            cfg.task.data = original_data


def calculate_validation_losses(checkpoint_infos: List[CheckpointInfo], runner: EvaluationRunner,
                                custom_dataset_path: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate validation loss on sanity, valid, and custom datasets for each checkpoint.
    Returns a comparison table.
    """
    print("\n[+] Calculating validation losses on all datasets...")
    
    rows = []
    
    for ckpt_info in checkpoint_infos:
        print(f"\n[+] Processing {ckpt_info.run_name}...")
        row = {"Run Name": ckpt_info.run_name, "Checkpoint": ckpt_info.path}
        
        try:
            # Load model and config
            from model_loader import load_fairseq_checkpoint
            from omegaconf import OmegaConf
            model, model_cfg, checkpoint_info_loaded = load_fairseq_checkpoint(ckpt_info.path)
            cfg = checkpoint_info_loaded["cfg"]
            
            # Create a copy of cfg to avoid modifying the original
            cfg = OmegaConf.create(OmegaConf.to_container(cfg))
            runner._current_cfg = cfg
            
            # Prepare model
            model, device = runner._prepare_model_for_eval(model, cfg)
            
            # Calculate validation loss on sanity (train dataset)
            print(f"    Calculating validation loss on sanity (train) dataset...")
            val_metrics_sanity = calculate_validation_loss_for_split(
                runner, model, cfg, split="sanity", custom_dataset_path=None
            )
            row["Sanity Loss"] = val_metrics_sanity.get("eval_loss", "N/A")
            row["Sanity Loss Std"] = val_metrics_sanity.get("eval_loss_std", "N/A")
            
            # Calculate validation loss on valid dataset
            print(f"    Calculating validation loss on valid dataset...")
            val_metrics_valid = calculate_validation_loss_for_split(
                runner, model, cfg, split="valid", custom_dataset_path=None
            )
            row["Valid Loss"] = val_metrics_valid.get("eval_loss", "N/A")
            row["Valid Loss Std"] = val_metrics_valid.get("eval_loss_std", "N/A")
            
            # Calculate validation loss on custom dataset if provided
            if custom_dataset_path:
                print(f"    Calculating validation loss on custom dataset: {custom_dataset_path}...")
                val_metrics_custom = calculate_validation_loss_for_split(
                    runner, model, cfg, split="valid", custom_dataset_path=custom_dataset_path
                )
                row["Custom Loss"] = val_metrics_custom.get("eval_loss", "N/A")
                row["Custom Loss Std"] = val_metrics_custom.get("eval_loss_std", "N/A")
            else:
                row["Custom Loss"] = "N/A"
                row["Custom Loss Std"] = "N/A"
            
        except Exception as e:
            print(f"[!] Error processing {ckpt_info.run_name}: {e}")
            import traceback
            traceback.print_exc()
            row["Sanity Loss"] = "Error"
            row["Valid Loss"] = "Error"
            row["Custom Loss"] = "Error"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def print_validation_loss_table(df: pd.DataFrame, output_dir: Path):
    """Print and save validation loss comparison table."""
    print("\n" + "="*120)
    print("VALIDATION LOSS COMPARISON")
    print("="*120)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*120)
    
    # Save to CSV
    csv_path = output_dir / "validation_loss_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved validation loss comparison table to: {csv_path}")
    
    # Save to text file
    txt_path = output_dir / "validation_loss_comparison.txt"
    with open(txt_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write("VALIDATION LOSS COMPARISON\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*120 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*120 + "\n")
    print(f"[+] Saved validation loss comparison text to: {txt_path}")


def print_comparison_summary(results: List[EvalResult]):
    """Print a summary comparison table."""
    print("\n" + "="*100)
    print("CHECKPOINT COMPARISON SUMMARY")
    print("="*100)
    
    # Create comparison table
    rows = []
    for result in results:
        row = {
            "Run Name": result.run_name,
            "Best Loss": result.metrics.get("best_loss", "N/A"),
            "Validation Loss": result.metrics.get("eval_loss", "N/A"),  # Use eval_loss from _eval_validation_loss
            "Valid Pearson": result.metrics.get("valid_pearson_corr", "N/A"),
            "Valid Spearman": result.metrics.get("valid_spearman_corr", "N/A"),
            "Valid Var Ratio": result.metrics.get("valid_sim_variance_ratio", "N/A"),
        }
        
        # Add custom metrics if available
        if result.metrics.get("custom_pearson_corr") is not None:
            row["Custom Pearson"] = result.metrics.get("custom_pearson_corr", "N/A")
            row["Custom Spearman"] = result.metrics.get("custom_spearman_corr", "N/A")
            row["Custom Var Ratio"] = result.metrics.get("custom_sim_variance_ratio", "N/A")
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description="Compare multiple checkpoints")
    
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                       help="Paths to checkpoint files")
    parser.add_argument("--output_dir", type=str,
                       default="/mnt5/noy/SpectralFM/code/eval_results/comparison",
                       help="Directory to save comparison results")
    parser.add_argument("--data_dir", type=str,
                       default="/mnt5/noy/fairseq/data/single_channel_1m/",
                       help="Directory containing evaluation data")
    parser.add_argument("--custom_dataset_path", type=str, default=None,
                       help="Path to custom dataset for evaluation")
    parser.add_argument("--eval_methods", type=str, nargs="+",
                       default=["validation_loss", "embedding_similarity"],
                       help="Evaluation methods to run")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode: save plots of evaluated data samples to debug_plots directory")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[+] Evaluating {len(args.checkpoints)} checkpoints...")
    print(f"[+] Output directory: {output_dir}")
    
    # Create checkpoint info objects
    checkpoint_infos = []
    for ckpt_path in args.checkpoints:
        try:
            ckpt_info = create_checkpoint_info(ckpt_path)
            checkpoint_infos.append(ckpt_info)
            print(f"    - {ckpt_info.run_name}: {ckpt_path}")
        except Exception as e:
            print(f"[!] Error creating checkpoint info for {ckpt_path}: {e}")
            continue
    
    if not checkpoint_infos:
        print("[!] No valid checkpoints found!")
        return
    
    # Run evaluations
    runner = EvaluationRunner(str(output_dir), args.data_dir)
    results = runner.evaluate_all(
        checkpoint_infos, 
        args.eval_methods,
        custom_dataset_path=args.custom_dataset_path,
        debug=args.debug
    )
    
    if not results:
        print("[!] No evaluation results!")
        return
    
    # Print summary
    print_comparison_summary(results)
    
    # Generate comparison visualizations
    print("\n[+] Generating comparison visualizations...")
    
    # Validation loss comparison
    plot_validation_loss_comparison(results, output_dir)
    
    # Embedding metrics comparison
    plot_embedding_metrics_comparison(results, output_dir)
    
    # Side-by-side plots (embedding similarity comparison)
    create_side_by_side_plots(checkpoint_infos, runner, output_dir, 
                              custom_dataset_path=args.custom_dataset_path)
    
    # Histogram comparison plots (input + embedding similarity distributions)
    print("\n[+] Creating embedding similarity comparison plots...")
    runner.plot_embedding_similarity_histogram_comparison(results, dataset_name="valid")
    if args.custom_dataset_path:
        custom_name = Path(args.custom_dataset_path).name
        runner.plot_embedding_similarity_histogram_comparison(results, dataset_name=custom_name)
    
    # Validation loss comparison table (only if validation_loss in eval_methods)
    if "validation_loss" in args.eval_methods:
        print("\n[+] Calculating validation losses on all datasets...")
        val_loss_df = calculate_validation_losses(checkpoint_infos, runner, 
                                                   custom_dataset_path=args.custom_dataset_path)
        print_validation_loss_table(val_loss_df, output_dir)
    else:
        print("\n[+] Skipping validation loss calculation (not in eval_methods)")
    
    # Save results summary
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("CHECKPOINT COMPARISON SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        
        for result in results:
            f.write(f"\nRun: {result.run_name}\n")
            f.write(f"Checkpoint: {result.checkpoint_path}\n")
            f.write(f"Best Loss: {result.metrics.get('best_loss', 'N/A')}\n")
            f.write(f"Validation Loss: {result.metrics.get('eval_loss', 'N/A')}\n")  # Use eval_loss from _eval_validation_loss
            f.write(f"Valid Pearson: {result.metrics.get('valid_pearson_corr', 'N/A')}\n")
            f.write(f"Valid Spearman: {result.metrics.get('valid_spearman_corr', 'N/A')}\n")
            f.write(f"Valid Var Ratio: {result.metrics.get('valid_sim_variance_ratio', 'N/A')}\n")
            if result.metrics.get('custom_pearson_corr') is not None:
                f.write(f"Custom Pearson: {result.metrics.get('custom_pearson_corr', 'N/A')}\n")
                f.write(f"Custom Spearman: {result.metrics.get('custom_spearman_corr', 'N/A')}\n")
                f.write(f"Custom Var Ratio: {result.metrics.get('custom_sim_variance_ratio', 'N/A')}\n")
            f.write("-"*100 + "\n")
    
    print(f"\n[+] Comparison complete!")
    print(f"[+] Results saved to: {output_dir}")
    print(f"[+] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
