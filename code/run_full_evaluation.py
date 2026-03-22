#!/usr/bin/env python3
"""
SpectralFM Full Evaluation Orchestrator

Coordinates all evaluation methods across checkpoints, datasets, and splits.
Produces final results, scaling law plots, transfer matrices, and summary reports.

Usage:
    # Full evaluation on a single checkpoint:
    python run_full_evaluation.py --checkpoint /path/to/checkpoint_best.pt

    # Scaling law sweep across dataset sizes:
    python run_full_evaluation.py --mode scaling_laws

    # Transfer evaluation between datasets:
    python run_full_evaluation.py --mode transfer

    # Light mode (no fairseq):
    python run_full_evaluation.py --mode light --data_dir /path/to/wavs/

    # All evaluations:
    python run_full_evaluation.py --mode full --checkpoint /path/to/checkpoint_best.pt
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ #
#  Dataset registry                                                   #
# ------------------------------------------------------------------ #

DATASETS = {
    "single_channel_100": "/mnt5/noy/fairseq/data/single_channel_100/",
    "single_channel_1k": "/mnt5/noy/fairseq/data/single_channel_1k/",
    "single_channel_10k": "/mnt5/noy/fairseq/data/single_channel_10k/",
    "single_channel_1m": "/mnt5/noy/fairseq/data/single_channel_1m/",
    "single_channel_5m": "/mnt5/noy/fairseq/data/single_channel_5m/",
    "single_channel_all": "/mnt5/noy/fairseq/data/single_channel_all/",
    "multi_channel": "/mnt5/noy/fairseq/data/multi_channel/",
    "labeled_data": "/mnt5/noy/fairseq/data/labeled_data/",
    "sampled_data": "/mnt5/noy/fairseq/data/sampled_data/",
}

# Datasets for scaling law sweep (ordered by size)
SCALING_DATASETS = [
    ("single_channel_100", 100),
    ("single_channel_1k", 1000),
    ("single_channel_10k", 10000),
    ("single_channel_1m", 1000000),
    ("single_channel_5m", 5000000),
    ("single_channel_all", 9100000),
]

# Methods that require labels.tsv (parameter_0)
LABELED_METHODS = [
    "label_regression", "downstream_probing", "fewshot",
    "parameter_verification", "failure_analysis",
]

# Methods that require component IDs in filenames
COMPONENT_METHODS = [
    "component_clustering", "component_detection",
]

# Methods that work on any data
UNIVERSAL_METHODS = [
    "embedding_similarity", "knn_retrieval", "repr_geometry",
    "clustering", "noise_robustness", "signal_completion",
    "efficiency", "attention", "augmentation_invariance",
]

# Methods for baselines comparison
BASELINE_METHODS = ["baselines"]


def get_available_methods(data_dir: str) -> List[str]:
    """Determine which eval methods are applicable for a dataset."""
    methods = list(UNIVERSAL_METHODS)

    # Check for labels.tsv
    if os.path.isfile(os.path.join(data_dir, "labels.tsv")):
        methods.extend(LABELED_METHODS)
        methods.extend(BASELINE_METHODS)

    # Check for component IDs in filenames
    import glob
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))[:10]
    from eval_utils import extract_component_id
    has_components = any(
        extract_component_id(os.path.basename(f)) is not None
        for f in wav_files
    )
    if has_components:
        methods.extend(COMPONENT_METHODS)

    return methods


# ------------------------------------------------------------------ #
#  Mode: Full evaluation on single checkpoint                        #
# ------------------------------------------------------------------ #

def run_full_evaluation(
    checkpoint_path: str,
    output_dir: str,
    eval_data_dir: Optional[str] = None,
    max_samples: int = 500,
    include_random_weights: bool = True,
) -> Dict[str, Any]:
    """Run all applicable evaluations on a single checkpoint."""
    from evaluation_runner import (
        EvaluationRunner, CheckpointInfo,
    )

    data_dir = eval_data_dir or DATASETS["single_channel_1m"]
    methods = get_available_methods(data_dir)

    print(f"\n{'='*70}")
    print(f"  FULL EVALUATION")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Dataset: {data_dir}")
    print(f"  Methods ({len(methods)}): {methods}")
    print(f"  Max samples: {max_samples}")
    print(f"{'='*70}\n")

    runner = EvaluationRunner(output_dir, data_dir)
    runner._max_eval_samples = max_samples
    runner._dim_reduction_methods = ["pca"]

    ckpt_info = CheckpointInfo(
        path=checkpoint_path,
        run_dir=str(Path(checkpoint_path).parent.parent),
        date=datetime.now().strftime("%Y-%m-%d"),
        time=datetime.now().strftime("%H-%M-%S"),
        checkpoint_type="manual",
    )

    result = runner.evaluate_checkpoint(
        ckpt_info,
        eval_methods=methods,
        eval_data_dir=eval_data_dir,
        include_random_weights=include_random_weights,
    )

    # Save comprehensive JSON
    output_path = os.path.join(output_dir, "full_evaluation_results.json")
    results_dict = {
        "checkpoint_path": checkpoint_path,
        "eval_data_dir": data_dir,
        "n_methods": len(methods),
        "methods": methods,
        "max_samples": max_samples,
        "timestamp": datetime.now().isoformat(),
        "metrics": result.metrics,
        "metric_descriptions": result.metric_descriptions,
        "plot_paths": result.plot_paths,
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)

    print(f"\n[+] Full evaluation results saved to: {output_path}")
    return results_dict


# ------------------------------------------------------------------ #
#  Mode: Scaling laws                                                #
# ------------------------------------------------------------------ #

def run_scaling_laws(
    checkpoint_dir: str,
    output_dir: str,
    max_samples: int = 200,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run evaluations across dataset size sweep checkpoints.
    Kaplan et al. (2020) style.
    """
    from evaluation_runner import (
        EvaluationRunner, CheckpointDiscovery, CheckpointInfo,
    )
    from eval_plots import plot_scaling_laws

    if methods is None:
        methods = ["embedding_similarity", "knn_retrieval", "repr_geometry", "clustering"]

    print(f"\n{'='*70}")
    print(f"  SCALING LAW EVALUATION")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Methods: {methods}")
    print(f"{'='*70}\n")

    # Discover checkpoints
    discovery = CheckpointDiscovery(checkpoint_dir)
    all_checkpoints = discovery.find_best_checkpoints()

    dataset_sizes = []
    all_metrics = []

    for dataset_name, size in SCALING_DATASETS:
        data_dir = DATASETS.get(dataset_name)
        if not data_dir or not os.path.isdir(data_dir):
            print(f"[!] Skipping {dataset_name}: not found")
            continue

        # Find checkpoint trained on this dataset
        matching = [c for c in all_checkpoints if dataset_name in c.run_dir]
        if not matching:
            print(f"[!] No checkpoint found for {dataset_name}")
            continue

        ckpt = matching[0]
        print(f"\n[+] Evaluating {dataset_name} (size={size})...")

        runner = EvaluationRunner(output_dir, data_dir)
        runner._max_eval_samples = max_samples

        result = runner.evaluate_checkpoint(
            ckpt,
            eval_methods=methods,
            eval_data_dir=data_dir,
        )

        dataset_sizes.append(size)
        all_metrics.append(result.metrics)

    if len(all_metrics) < 2:
        print("[!] Not enough checkpoints for scaling laws")
        return {"error": "Not enough checkpoints"}

    # Aggregate metrics for plotting
    metric_keys = set()
    for m in all_metrics:
        metric_keys.update(k for k, v in m.items() if isinstance(v, (int, float)))

    metrics_per_size = {}
    for key in sorted(metric_keys):
        values = [m.get(key, None) for m in all_metrics]
        if all(v is not None for v in values):
            metrics_per_size[key] = values

    # Generate scaling law plots
    plot_path = os.path.join(output_dir, "scaling_laws.png")
    # Select key metrics for the plot
    key_metrics = {k: v for k, v in metrics_per_size.items()
                   if any(prefix in k for prefix in [
                       "pearson_corr", "knn_neighbor_overlap",
                       "repr_cka_linear", "repr_effective_rank",
                       "cluster_", "comp_cluster_ari",
                   ])}

    if key_metrics:
        plot_scaling_laws(dataset_sizes, key_metrics, plot_path)
        print(f"[+] Scaling law plots saved to: {plot_path}")

    # Save results
    results = {
        "dataset_sizes": dataset_sizes,
        "metrics_per_size": metrics_per_size,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = os.path.join(output_dir, "scaling_law_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


# ------------------------------------------------------------------ #
#  Mode: Transfer evaluation                                         #
# ------------------------------------------------------------------ #

def run_transfer_evaluation(
    checkpoint_path: str,
    output_dir: str,
    max_samples: int = 200,
) -> Dict[str, Any]:
    """
    Cross-dataset transfer evaluation.

    Train probe on one dataset, evaluate on another.
    Tests generalization of learned representations.
    """
    from evaluation_runner import EvaluationRunner, CheckpointInfo
    from eval_utils import DatasetSplitter, extract_component_ids_from_directory

    print(f"\n{'='*70}")
    print(f"  TRANSFER EVALUATION")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*70}\n")

    transfer_pairs = [
        ("multi_channel", "sampled_data", "multi→sampled (28 comps, many unseen)"),
        ("single_channel_1m", "multi_channel", "single→multi (cross-domain)"),
    ]

    all_results = {}

    for train_ds, eval_ds, description in transfer_pairs:
        train_dir = DATASETS.get(train_ds)
        eval_dir = DATASETS.get(eval_ds)

        if not train_dir or not eval_dir:
            continue
        if not os.path.isdir(train_dir) or not os.path.isdir(eval_dir):
            continue

        print(f"\n[+] Transfer: {description}")

        # Get component overlap
        try:
            comp_train, _ = extract_component_ids_from_directory(train_dir, max_files=1000)
            comp_eval, _ = extract_component_ids_from_directory(eval_dir, max_files=1000)

            shared, only_train, only_eval = DatasetSplitter.get_overlapping_components(
                comp_train, comp_eval
            )

            print(f"    Shared components: {shared}")
            print(f"    Only in train: {only_train}")
            print(f"    Only in eval: {only_eval}")

            all_results[f"{train_ds}→{eval_ds}"] = {
                "description": description,
                "shared_components": len(shared),
                "unique_to_eval": len(only_eval),
            }
        except Exception as e:
            print(f"[!] Component analysis failed: {e}")

        # Run evaluation on the eval dataset using the checkpoint
        try:
            runner = EvaluationRunner(output_dir, eval_dir)
            runner._max_eval_samples = max_samples

            ckpt_info = CheckpointInfo(
                path=checkpoint_path,
                run_dir=str(Path(checkpoint_path).parent.parent),
                date=datetime.now().strftime("%Y-%m-%d"),
                time=datetime.now().strftime("%H-%M-%S"),
                checkpoint_type="manual",
            )

            methods = ["embedding_similarity", "knn_retrieval", "component_clustering", "repr_geometry"]
            result = runner.evaluate_checkpoint(
                ckpt_info,
                eval_methods=methods,
                eval_data_dir=eval_dir,
            )

            all_results[f"{train_ds}→{eval_ds}"]["metrics"] = result.metrics
        except Exception as e:
            print(f"[!] Transfer evaluation failed: {e}")

    # Save results
    output_path = os.path.join(output_dir, "transfer_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[+] Transfer results saved to: {output_path}")

    return all_results


# ------------------------------------------------------------------ #
#  Mode: Light evaluation (no fairseq)                               #
# ------------------------------------------------------------------ #

def run_light_evaluation(
    data_dir: str,
    output_dir: str,
    model_path: Optional[str] = None,
    max_samples: int = 200,
) -> Dict[str, Any]:
    """
    Light mode evaluation: no fairseq dependency.

    Uses torchaudio for data loading and HuggingFace for model loading.
    Runs all metrics that only need (inputs, embeddings) arrays.
    """
    from eval_utils import load_wav_files_torchaudio, extract_component_id
    from eval_metrics import (
        compute_knn_metrics, compute_repr_geometry_metrics,
        compute_component_clustering_metrics,
    )
    from eval_plots import plot_component_clustering, plot_repr_geometry

    print(f"\n{'='*70}")
    print(f"  LIGHT MODE EVALUATION")
    print(f"  Data: {data_dir}")
    print(f"  Model: {model_path or 'None (input-only metrics)'}")
    print(f"{'='*70}\n")

    # Load data
    print("[+] Loading WAV files with torchaudio...")
    inputs, filenames = load_wav_files_torchaudio(data_dir, max_samples=max_samples)
    print(f"[+] Loaded {len(inputs)} samples, shape: {inputs.shape}")

    # Extract component IDs
    component_ids = np.array([
        extract_component_id(fn) if extract_component_id(fn) is not None else -1
        for fn in filenames
    ])
    has_components = np.sum(component_ids >= 0) > 10

    all_metrics: Dict[str, float] = {}

    # If model provided, extract embeddings
    embeddings = None
    if model_path is not None:
        print("[+] Loading model and extracting embeddings...")
        try:
            from model_loader import load_custom_data2vec_audio_model
            import torch

            model = load_custom_data2vec_audio_model(model_path)
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            emb_list = []
            with torch.no_grad():
                for i in range(len(inputs)):
                    x = torch.tensor(inputs[i], dtype=torch.float32).unsqueeze(0).to(device)
                    output = model(x)
                    if hasattr(output, "last_hidden_state"):
                        emb = output.last_hidden_state.mean(dim=1)
                    elif isinstance(output, tuple):
                        emb = output[0].mean(dim=1) if output[0].dim() == 3 else output[0]
                    else:
                        emb = output.mean(dim=1) if output.dim() == 3 else output
                    emb_list.append(emb.cpu().numpy().flatten())
            embeddings = np.array(emb_list)
            print(f"[+] Extracted embeddings: {embeddings.shape}")
        except Exception as e:
            print(f"[!] Model loading failed: {e}")
            print("[+] Falling back to input-only metrics")

    if embeddings is None:
        # Use PCA as a simple embedding
        from sklearn.decomposition import PCA
        n_comp = min(50, inputs.shape[1], inputs.shape[0])
        pca = PCA(n_components=n_comp)
        embeddings = pca.fit_transform(inputs)
        print(f"[+] Using PCA embeddings as fallback: {embeddings.shape}")
        all_metrics["light_mode_embedding"] = "pca_fallback"

    # Run metrics
    print("[+] Computing KNN retrieval metrics...")
    knn_metrics = compute_knn_metrics(inputs, embeddings, k=10)
    all_metrics.update(knn_metrics)

    print("[+] Computing representation geometry metrics...")
    labels = component_ids if has_components else None
    repr_metrics = compute_repr_geometry_metrics(inputs, embeddings, labels=labels)
    all_metrics.update(repr_metrics)

    if has_components:
        print("[+] Computing component clustering metrics...")
        valid = component_ids >= 0
        comp_metrics = compute_component_clustering_metrics(
            embeddings[valid], component_ids[valid], inputs=inputs[valid]
        )
        all_metrics.update(comp_metrics)

        # Plot
        try:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, "light_component_clustering.png")
            plot_component_clustering(
                embeddings[valid], component_ids[valid],
                comp_metrics, plot_path,
                title="Light Mode: Component Clustering",
            )
        except Exception as e:
            print(f"[!] Plot failed: {e}")

    # Repr geometry plot
    try:
        from eval_metrics import compute_effective_rank
        _, _, sv = compute_effective_rank(embeddings)
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "light_repr_geometry.png")
        plot_repr_geometry(repr_metrics, sv, plot_path, title="Light Mode: Repr Geometry")
    except Exception as e:
        print(f"[!] Plot failed: {e}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "light_evaluation_results.json")
    results = {
        "data_dir": data_dir,
        "model_path": model_path,
        "n_samples": len(inputs),
        "has_components": bool(has_components),
        "timestamp": datetime.now().isoformat(),
        "metrics": {k: v for k, v in all_metrics.items() if isinstance(v, (int, float, str))},
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[+] Light mode results saved to: {output_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  LIGHT MODE RESULTS")
    print(f"{'='*50}")
    for k, v in sorted(all_metrics.items()):
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        elif isinstance(v, int):
            print(f"    {k}: {v}")

    return results


# ------------------------------------------------------------------ #
#  Mode: Training dynamics (intermediate checkpoints)                #
# ------------------------------------------------------------------ #

def run_training_dynamics(
    checkpoint_dir: str,
    output_dir: str,
    eval_data_dir: Optional[str] = None,
    max_samples: int = 200,
) -> Dict[str, Any]:
    """
    Evaluate intermediate checkpoints to track metric trajectories.
    """
    from evaluation_runner import EvaluationRunner, CheckpointDiscovery

    print(f"\n{'='*70}")
    print(f"  TRAINING DYNAMICS")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*70}\n")

    data_dir = eval_data_dir or DATASETS["single_channel_1m"]
    methods = ["embedding_similarity", "knn_retrieval", "repr_geometry"]

    discovery = CheckpointDiscovery(checkpoint_dir)
    all_checkpoints = discovery.find_all_checkpoints()

    if not all_checkpoints:
        return {"error": "No checkpoints found"}

    # Sort by checkpoint type (step number)
    def sort_key(c):
        try:
            return int(c.checkpoint_type.replace("_", ""))
        except (ValueError, AttributeError):
            if c.checkpoint_type == "best":
                return float("inf")
            return 0

    all_checkpoints.sort(key=sort_key)

    steps = []
    all_metrics = []

    for ckpt in all_checkpoints:
        print(f"\n[+] Evaluating checkpoint: {ckpt.checkpoint_type}...")
        try:
            runner = EvaluationRunner(output_dir, data_dir)
            runner._max_eval_samples = max_samples

            result = runner.evaluate_checkpoint(
                ckpt, eval_methods=methods, eval_data_dir=eval_data_dir,
            )

            step = sort_key(ckpt)
            steps.append(step)
            all_metrics.append(result.metrics)
        except Exception as e:
            print(f"[!] Failed: {e}")

    # Save results
    results = {
        "steps": steps,
        "metrics": all_metrics,
        "checkpoint_dir": checkpoint_dir,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = os.path.join(output_dir, "training_dynamics_results.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[+] Training dynamics results saved to: {output_path}")
    return results


# ------------------------------------------------------------------ #
#  Mode: Ablation studies                                            #
# ------------------------------------------------------------------ #

def generate_ablation_configs() -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate ablation study configurations.

    Each ablation varies one hyperparameter while keeping others fixed.
    Returns dict of ablation_name -> list of config overrides.
    """
    configs = {}

    configs["mask_prob"] = [
        {"mask_prob": p} for p in [0.1, 0.3, 0.5, 0.65, 0.8]
    ]

    configs["mask_length"] = [
        {"mask_length": l} for l in [5, 10, 20, 40]
    ]

    configs["encoder_layers_used"] = [
        {"encoder_layers_used": n} for n in [1, 3, 6, 9, 12]
    ]

    configs["embedding_aggregation"] = [
        {"aggregation": agg} for agg in ["mean", "last_token", "cls", "weighted_sum"]
    ]

    return configs


# ------------------------------------------------------------------ #
#  CLI Entry Point                                                    #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="SpectralFM Full Evaluation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full              Run all applicable evaluations on a single checkpoint
  scaling_laws      Run evaluations across dataset size sweep checkpoints  
  transfer          Cross-dataset transfer evaluation
  light             Light mode: no fairseq, uses torchaudio + HuggingFace
  training_dynamics Evaluate intermediate checkpoints for metric trajectories
  ablation          Generate ablation study configs

Examples:
  python run_full_evaluation.py --mode full --checkpoint /path/to/ckpt.pt
  python run_full_evaluation.py --mode light --data_dir /path/to/wavs/
  python run_full_evaluation.py --mode scaling_laws --checkpoint_dir /path/to/outputs/
        """,
    )

    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "scaling_laws", "transfer",
                                 "light", "training_dynamics", "ablation"],
                        help="Evaluation mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific checkpoint file")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/mnt5/noy/fairseq/outputs",
                        help="Base directory for fairseq outputs")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt5/noy/SpectralFM/code/eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory for evaluation")
    parser.add_argument("--eval_data_dir", type=str, default=None,
                        help="Override evaluation data directory")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Max samples for evaluation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to HuggingFace model (for light mode)")
    parser.add_argument("--include_random_weights", action="store_true",
                        help="Include random weight baseline comparison")

    args = parser.parse_args()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.mode}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    if args.mode == "full":
        if not args.checkpoint:
            parser.error("--checkpoint is required for full mode")
        run_full_evaluation(
            args.checkpoint, output_dir,
            eval_data_dir=args.eval_data_dir or args.data_dir,
            max_samples=args.max_samples,
            include_random_weights=args.include_random_weights,
        )

    elif args.mode == "scaling_laws":
        run_scaling_laws(
            args.checkpoint_dir, output_dir,
            max_samples=args.max_samples,
        )

    elif args.mode == "transfer":
        if not args.checkpoint:
            parser.error("--checkpoint is required for transfer mode")
        run_transfer_evaluation(
            args.checkpoint, output_dir,
            max_samples=args.max_samples,
        )

    elif args.mode == "light":
        data_dir = args.data_dir or args.eval_data_dir
        if not data_dir:
            parser.error("--data_dir is required for light mode")
        run_light_evaluation(
            data_dir, output_dir,
            model_path=args.model_path,
            max_samples=args.max_samples,
        )

    elif args.mode == "training_dynamics":
        run_training_dynamics(
            args.checkpoint_dir, output_dir,
            eval_data_dir=args.eval_data_dir or args.data_dir,
            max_samples=args.max_samples,
        )

    elif args.mode == "ablation":
        configs = generate_ablation_configs()
        output_path = os.path.join(output_dir, "ablation_configs.json")
        with open(output_path, "w") as f:
            json.dump(configs, f, indent=2)
        print(f"[+] Ablation configs saved to: {output_path}")
        for name, cfgs in configs.items():
            print(f"  {name}: {len(cfgs)} configurations")

    elapsed = time.time() - start_time
    print(f"\n[+] Total evaluation time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"[+] Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
