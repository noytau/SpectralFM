"""
Checkpoint Comparison Evaluation
----------------------------------
Runs embedding_similarity (and optionally noise_robustness) across multiple checkpoints.
Shows how evaluation scores evolve during training.

This is NOT a standalone evaluation — it wraps the other evaluations and produces
a comparative view across training steps/epochs.
"""
from __future__ import annotations
import datetime
import os
from typing import Optional
import pandas as pd

from . import embedding_similarity as emb_eval
from . import noise_robustness as noise_eval
from . import structured_similarity as struct_eval
from . import clustering as clust_eval
from . import label_regression as labelreg_eval


def run(
    checkpoints: list,          # [(label, model), ...] from CheckpointLoader.load_multiple
    df: pd.DataFrame,
    dataloader,
    device: str = "cpu",
    run_noise: bool = True,
    run_clustering: bool = True,
    run_label_regression: bool = True,
    k: int = 5,
    nova_data_dir: Optional[str] = None,
    labeled_data_dir: Optional[str] = None,
    label_reg_max_samples: int = 2000,
) -> dict:
    """
    For each checkpoint, run embedding similarity (+ optional noise robustness).

    Returns dict with:
      - comparison_df: one row per checkpoint with key metrics
      - per_checkpoint: {label: full eval results dict}
    """
    rows = []
    per_checkpoint = {}

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

        emb_results = emb_eval.run(df=df, model=model, dataloader=dataloader, k=k, device=device)

        row = {
            "checkpoint": label,
            "date":       ckpt_date,
            "embedding_stack_match_rate": emb_results["embedding_stack_match_rate"],
            "input_stack_match_rate": emb_results["input_stack_match_rate"],
            "match_score_avg": emb_results["match_score_avg"],
        }

        checkpoint_results = {"embedding_similarity": emb_results}

        if run_noise:
            noise_results = noise_eval.run(df=df, model=model, device=device)
            checkpoint_results["noise_robustness"] = noise_results
            for noise_type, mean_sim in noise_results["summary"].items():
                row[f"noise_{noise_type}"] = mean_sim

        if run_clustering:
            try:
                clust_results = clust_eval.run(
                    df=df, model=model, device=device, k=k,
                    embeddings=emb_results.get("embeddings"),
                )
                checkpoint_results["clustering"] = clust_results
                row["clustering_ari"] = clust_results.get("comp_cluster_ari")
                row["clustering_nmi"] = clust_results.get("comp_cluster_nmi")
                row["clustering_silhouette"] = clust_results.get("comp_cluster_silhouette")
            except Exception as e:
                print(f"[CheckpointComparison] Clustering skipped: {e}")

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
