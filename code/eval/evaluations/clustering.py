"""
Clustering Evaluation
----------------------
KMeans on embeddings with stack_idx as ground-truth labels. Measures how well
the model groups samples from the same stack (observation) together.

Thin wrapper around metrics.compute_component_clustering_metrics (ported from
Geoffrey eval_metrics.py). Metrics: ARI, NMI, V-measure, silhouette,
KNN precision, retrieval mAP, variance ratio, embedding-input alignment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..metrics import compute_component_clustering_metrics


def _embed(model, data: np.ndarray, device: str, batch_size: int = 32) -> np.ndarray:
    model.eval()
    model.to(device)
    embs = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i : i + batch_size], dtype=torch.float32).to(device)
            out = model(input_values=batch)
            embs.append(out.last_hidden_state.mean(dim=1).cpu().numpy())
    return np.concatenate(embs, axis=0)


def run(
    df: pd.DataFrame,
    model,
    device: str = "cpu",
    batch_size: int = 32,
    k: int = 10,
    embeddings: np.ndarray | None = None,
) -> dict:
    """
    Run clustering evaluation on df ('data' + 'stack_idx' columns).

    Args:
        embeddings  Pre-computed embeddings [N, D]; skips model inference if provided
    Returns:
        dict of comp_cluster_* metrics + embeddings/labels for plotting
    """
    inputs = np.stack(df["data"].apply(np.array).values).astype(np.float32)
    stack_ids = df["stack_idx"].values

    if embeddings is None:
        embeddings = _embed(model, inputs, device, batch_size)

    metrics = compute_component_clustering_metrics(
        embeddings=embeddings, component_ids=stack_ids, inputs=inputs, k=k
    )

    if "comp_cluster_error" not in metrics:
        print(f"[Clustering] n_clusters={metrics['comp_cluster_n_components']}  "
              f"ARI={metrics['comp_cluster_ari']:.4f}  "
              f"NMI={metrics['comp_cluster_nmi']:.4f}  "
              f"silhouette={metrics['comp_cluster_silhouette']:.4f}")
    else:
        print(f"[Clustering] Skipped: {metrics['comp_cluster_error']}")

    return {
        **metrics,
        "embeddings": embeddings,
        "true_labels": stack_ids,
    }
