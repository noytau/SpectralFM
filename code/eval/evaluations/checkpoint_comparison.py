"""
Checkpoint Comparison Evaluation
----------------------------------
Runs the per-model evals across multiple checkpoints — on multiple datasets (E4).
Shows how evaluation scores evolve during training.

This is NOT a standalone evaluation — it wraps the other evaluations and produces
a comparative view across training steps/epochs.

`datasets` maps dataset alias → {"df": normalized_df, "loader": dataloader}.
Which evals run on which alias comes from `eval_datasets` (the dataset × eval
matrix, see runner.EVAL_DATASET_MATRIX). Metric columns and per-checkpoint result
keys carry the alias suffix, e.g. `embedding_stack_match_rate_in_dist`,
`noise_robustness_sanity`. label_regression (labeled set only) and
structured_similarity (run-level panel) keep unsuffixed keys.
"""
from __future__ import annotations
import datetime
import os
from typing import Dict, Optional

import pandas as pd

from . import embedding_similarity as emb_eval
from . import noise_robustness as noise_eval
from . import structured_similarity as struct_eval
from . import clustering as clust_eval
from . import label_regression as labelreg_eval


def run(
    checkpoints: list,             # [(label, model, path), ...] from CheckpointLoader.load_multiple
    datasets: Dict[str, dict],     # alias → {"df": DataFrame, "loader": DataLoader}
    eval_datasets: Dict[str, list],  # eval name → list of dataset aliases
    device: str = "cpu",
    run_noise: bool = True,
    run_clustering: bool = True,
    run_label_regression: bool = True,
    k: int = 5,
    nova_data_dir: Optional[str] = None,
    labeled_data_dir: Optional[str] = None,
    label_reg_max_samples: int = 1000,
) -> dict:
    """
    For each checkpoint, run each eval on its matrix datasets.

    Returns dict with:
      - comparison_df: one row per checkpoint, metric columns suffixed per dataset
      - per_checkpoint: {label: {f"{eval}_{alias}": results, ...}}
    """
    rows = []
    per_checkpoint = {}

    def _aliases(eval_name):
        return [a for a in eval_datasets.get(eval_name, []) if a in datasets]

    for item in checkpoints:
        label, model = item[0], item[1]
        ckpt_path    = item[2] if len(item) > 2 else None

        # File modification time as checkpoint date
        if ckpt_path and os.path.isfile(ckpt_path):
            mtime = os.path.getmtime(ckpt_path)
            ckpt_date = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        else:
            ckpt_date = "N/A"

        print(f"\n{'='*50}")
        print(f"[CheckpointComparison] Evaluating: {label}  ({ckpt_date})")
        print(f"{'='*50}")

        row = {"checkpoint": label, "date": ckpt_date}
        checkpoint_results = {}

        # ── Stack query (embedding similarity) ────────────────────────────────
        for alias in _aliases("embedding_similarity"):
            ds = datasets[alias]
            print(f"[CheckpointComparison] embedding_similarity on {alias}")
            emb_results = emb_eval.run(df=ds["df"], model=model,
                                       dataloader=ds["loader"], k=k, device=device)
            checkpoint_results[f"embedding_similarity_{alias}"] = emb_results
            row[f"embedding_stack_match_rate_{alias}"] = emb_results["embedding_stack_match_rate"]
            row[f"input_stack_match_rate_{alias}"] = emb_results["input_stack_match_rate"]
            row[f"match_score_avg_{alias}"] = emb_results["match_score_avg"]

        # ── Noise robustness ──────────────────────────────────────────────────
        if run_noise:
            for alias in _aliases("noise_robustness"):
                ds = datasets[alias]
                print(f"[CheckpointComparison] noise_robustness on {alias}")
                noise_results = noise_eval.run(df=ds["df"], model=model, device=device)
                checkpoint_results[f"noise_robustness_{alias}"] = noise_results
                for noise_type, mean_sim in noise_results["summary"].items():
                    row[f"noise_{noise_type}_{alias}"] = mean_sim

        # ── Clustering ────────────────────────────────────────────────────────
        if run_clustering:
            for alias in _aliases("clustering"):
                ds = datasets[alias]
                try:
                    print(f"[CheckpointComparison] clustering on {alias}")
                    emb_key = f"embedding_similarity_{alias}"
                    precomputed = (checkpoint_results.get(emb_key) or {}).get("embeddings")
                    clust_results = clust_eval.run(
                        df=ds["df"], model=model, device=device, k=k,
                        embeddings=precomputed,
                    )
                    checkpoint_results[f"clustering_{alias}"] = clust_results
                    row[f"clustering_ari_{alias}"] = clust_results.get("comp_cluster_ari")
                    row[f"clustering_nmi_{alias}"] = clust_results.get("comp_cluster_nmi")
                    row[f"clustering_silhouette_{alias}"] = clust_results.get("comp_cluster_silhouette")
                except Exception as e:
                    print(f"[CheckpointComparison] Clustering ({alias}) skipped: {e}")

        # ── Label regression (labeled set only — unsuffixed keys) ─────────────
        if run_label_regression and labeled_data_dir and os.path.isdir(labeled_data_dir):
            try:
                lr_results = labelreg_eval.run(
                    model=model, labeled_data_dir=labeled_data_dir, device=device,
                    max_samples=label_reg_max_samples,
                )
                checkpoint_results["label_regression"] = lr_results
                for sfx in ("", "_2c", "_3c"):
                    for key in ("label_reg_input_r2", "label_reg_emb_r2",
                                "label_reg_improvement_r2"):
                        if f"{key}{sfx}" in lr_results:
                            row[f"{key}{sfx}"] = lr_results[f"{key}{sfx}"]
            except Exception as e:
                print(f"[CheckpointComparison] Label regression skipped: {e}")

        # ── Structured similarity (run-level canonical panel) ─────────────────
        if nova_data_dir and os.path.isdir(nova_data_dir):
            try:
                print(f"[CheckpointComparison] Structured similarity: {label}")
                ss_results = struct_eval.run(model, nova_data_dir, device=device)
                checkpoint_results["structured_similarity"] = ss_results
            except Exception as e:
                print(f"[CheckpointComparison] Structured similarity skipped: {e}")

        rows.append(row)
        checkpoint_results["_date"] = ckpt_date
        per_checkpoint[label] = checkpoint_results

    comparison_df = pd.DataFrame(rows)
    print("\n[CheckpointComparison] Summary:")
    print(comparison_df.to_string(index=False))

    return {
        "comparison_df": comparison_df,
        "per_checkpoint": per_checkpoint,
    }
