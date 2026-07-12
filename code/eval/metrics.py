"""
Pure metric computation functions for SpectralFM evaluation.

Trimmed port of Geoffrey's /mnt5/noy/SpectralFM/code/eval_metrics.py — only the
functions needed by the eval package. Stateless: numpy in, dict of floats out.
Zero fairseq dependency (numpy / scipy / sklearn only).
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats as scipy_stats


# ------------------------------------------------------------------ #
#  KNN retrieval helper                                               #
# ------------------------------------------------------------------ #

def knn_retrieval(
    query_space: np.ndarray,
    search_space: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each sample, find k nearest neighbors in search_space.

    Args:
        query_space: [N, D] array to compute queries from (same indices as search_space)
        search_space: [N, D] array to search in
        k: number of neighbors (excluding self)
        metric: 'cosine' or 'euclidean'

    Returns:
        indices: [N, k] neighbor indices
        scores: [N, k] similarity/distance scores
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="auto")
    nn.fit(search_space)
    distances, indices = nn.kneighbors(query_space)
    # Remove self-match (first column)
    neighbor_indices = indices[:, 1:]
    neighbor_distances = distances[:, 1:]
    if metric == "cosine":
        # sklearn cosine distance = 1 - cosine_similarity
        neighbor_scores = 1.0 - neighbor_distances
    else:
        neighbor_scores = -neighbor_distances  # lower distance = more similar
    return neighbor_indices, neighbor_scores


# ------------------------------------------------------------------ #
#  Component-Based Clustering Metrics                                 #
# ------------------------------------------------------------------ #

def compute_component_clustering_metrics(
    embeddings: np.ndarray,
    component_ids: np.ndarray,
    inputs: Optional[np.ndarray] = None,
    k: int = 10,
) -> Dict[str, float]:
    """
    Evaluate embeddings using component IDs as ground-truth labels.

    Metrics (Hubert & Arabie 1985; Strehl & Ghosh 2002; Rosenberg & Hirschberg 2007):
        comp_cluster_ari: Adjusted Rand Index
        comp_cluster_nmi: Normalized Mutual Information
        comp_cluster_silhouette: Silhouette Score
        comp_cluster_vmeasure: V-Measure
        comp_cluster_knn_precision: Same-component K-NN precision
        comp_cluster_retrieval_map: Component retrieval mean average precision
        comp_cluster_variance_ratio: Inter/intra component distance ratio

    Args:
        embeddings: [N, D] model embeddings
        component_ids: [N,] integer component IDs
        inputs: [N, 245] optional raw inputs for comparison
        k: number of neighbors for KNN precision

    Returns:
        Dict of metrics
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        silhouette_score, v_measure_score,
    )
    from sklearn.metrics.pairwise import euclidean_distances

    metrics: Dict[str, float] = {}
    unique_components = np.unique(component_ids)
    n_components = len(unique_components)
    n = len(embeddings)

    if n_components < 2:
        return {"comp_cluster_error": f"Need >= 2 components, got {n_components}"}

    metrics["comp_cluster_n_components"] = n_components
    metrics["comp_cluster_n_samples"] = n

    # K-Means clustering with k = n_components
    km = KMeans(n_clusters=n_components, random_state=42, n_init=10)
    pred_labels = km.fit_predict(embeddings)

    # ARI, NMI, V-Measure
    metrics["comp_cluster_ari"] = float(adjusted_rand_score(component_ids, pred_labels))
    metrics["comp_cluster_nmi"] = float(normalized_mutual_info_score(component_ids, pred_labels))
    metrics["comp_cluster_vmeasure"] = float(v_measure_score(component_ids, pred_labels))

    # Silhouette using true labels
    try:
        metrics["comp_cluster_silhouette"] = float(
            silhouette_score(embeddings, component_ids)
        )
    except Exception:
        metrics["comp_cluster_silhouette"] = 0.0

    # KNN precision: fraction of k-NN sharing same component
    k_actual = min(k, n - 1)
    if k_actual >= 1:
        emb_indices, _ = knn_retrieval(embeddings, embeddings, k=k_actual, metric="cosine")
        precisions = []
        for i in range(n):
            neighbors = emb_indices[i]
            same = np.sum(component_ids[neighbors] == component_ids[i])
            precisions.append(same / k_actual)
        metrics["comp_cluster_knn_precision"] = float(np.mean(precisions))

        # Component retrieval mAP
        aps = []
        for i in range(n):
            neighbors = emb_indices[i]
            relevant = (component_ids[neighbors] == component_ids[i]).astype(float)
            if relevant.sum() == 0:
                continue
            cum_relevant = np.cumsum(relevant)
            precisions_at_k = cum_relevant / np.arange(1, k_actual + 1)
            ap = np.sum(precisions_at_k * relevant) / relevant.sum()
            aps.append(ap)
        metrics["comp_cluster_retrieval_map"] = float(np.mean(aps)) if aps else 0.0

    # Variance ratio: inter / intra component distances
    try:
        dists = euclidean_distances(embeddings)
        intra_dists, inter_dists = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if component_ids[i] == component_ids[j]:
                    intra_dists.append(dists[i, j])
                else:
                    inter_dists.append(dists[i, j])
        if intra_dists and inter_dists:
            metrics["comp_cluster_variance_ratio"] = float(
                np.var(inter_dists) / (np.var(intra_dists) + 1e-12)
            )
        else:
            metrics["comp_cluster_variance_ratio"] = 0.0
    except Exception:
        metrics["comp_cluster_variance_ratio"] = 0.0

    # Embedding-input alignment (Spearman) if inputs provided
    if inputs is not None:
        try:
            idx = np.random.RandomState(42).choice(n, min(n, 200), replace=False)
            input_dists = euclidean_distances(inputs[idx])
            emb_dists = euclidean_distances(embeddings[idx])
            triu = np.triu_indices_from(input_dists, k=1)
            rho, _ = scipy_stats.spearmanr(input_dists[triu], emb_dists[triu])
            metrics["comp_cluster_emb_input_align"] = float(rho)
        except Exception:
            metrics["comp_cluster_emb_input_align"] = 0.0

    return metrics


# ------------------------------------------------------------------ #
#  Linear Probing (label regression)                                  #
# ------------------------------------------------------------------ #

def compute_linear_probing_metrics(
    X: np.ndarray,
    y: np.ndarray,
    probe_type: str = "ridge",
    n_folds: int = 5,
    task: str = "regression",
    return_predictions: bool = False,
) -> Dict[str, float]:
    """
    Linear probing evaluation.

    Alain & Bengio (2017); Chen et al./SimCLR (2020).

    Args:
        X: [N, D] features (inputs or embeddings)
        y: [N,] labels (float for regression, int for classification)
        probe_type: 'ridge', 'knn', 'mlp', 'lasso', 'svr'
        n_folds: number of CV folds
        task: 'regression' or 'classification'

    Returns:
        Dict of metrics
    """
    from sklearn.model_selection import cross_val_predict

    metrics: Dict[str, float] = {}
    prefix = f"downstream_{probe_type}_probe"

    n_folds = min(n_folds, len(y))
    if len(y) < 10:
        return {f"{prefix}_error": "Not enough samples"}

    if task == "regression":
        if probe_type == "ridge":
            from sklearn.linear_model import RidgeCV
            model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=n_folds)
        elif probe_type == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            model = KNeighborsRegressor(n_neighbors=min(5, len(y) - 1))
        elif probe_type == "mlp":
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(
                hidden_layer_sizes=(256,), max_iter=500,
                random_state=42, early_stopping=True,
            )
        elif probe_type == "lasso":
            from sklearn.linear_model import LassoCV
            model = LassoCV(cv=n_folds, random_state=42)
        elif probe_type == "svr":
            from sklearn.svm import SVR
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            model = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
        else:
            return {f"{prefix}_error": f"Unknown probe type: {probe_type}"}

        y_pred = cross_val_predict(model, X, y, cv=n_folds)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        mse = float(np.mean((y - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y - y_pred)))
        pearson_r, _ = scipy_stats.pearsonr(y, y_pred)
        spearman_rho, _ = scipy_stats.spearmanr(y, y_pred)

        metrics[f"{prefix}_r2"] = float(r2)
        metrics[f"{prefix}_mse"] = mse
        metrics[f"{prefix}_rmse"] = rmse
        metrics[f"{prefix}_mae"] = mae
        metrics[f"{prefix}_pearson_r"] = float(pearson_r)
        metrics[f"{prefix}_spearman_rho"] = float(spearman_rho)
        if return_predictions:
            metrics["predictions"] = y_pred

    elif task == "classification":
        if probe_type == "ridge":
            from sklearn.linear_model import RidgeClassifier
            model = RidgeClassifier(alpha=1.0)
        elif probe_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=min(5, len(y) - 1))
        elif probe_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(
                hidden_layer_sizes=(256,), max_iter=500,
                random_state=42, early_stopping=True,
            )
        else:
            return {f"{prefix}_error": f"Unsupported probe for classification: {probe_type}"}

        from sklearn.metrics import accuracy_score, f1_score
        y_pred = cross_val_predict(model, X, y, cv=n_folds)
        metrics[f"{prefix}_accuracy"] = float(accuracy_score(y, y_pred))
        metrics[f"{prefix}_f1_macro"] = float(f1_score(y, y_pred, average="macro", zero_division=0))

    return metrics


# ------------------------------------------------------------------ #
#  Extended Signal Completion                                         #
# ------------------------------------------------------------------ #

def compute_extended_signal_completion(
    gt_embeddings: np.ndarray,
    pred_embeddings: np.ndarray,
    mask_indices: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Extended signal completion analysis with per-bin error.

    Divides the masked positions into bins along the sequence dimension
    and computes error metrics per bin. Useful for understanding if
    errors concentrate at specific positions (e.g., edges vs center).

    Args:
        gt_embeddings: [T, D] ground truth embeddings
        pred_embeddings: [T, D] predicted embeddings
        mask_indices: [T,] boolean mask
        n_bins: number of position bins

    Returns:
        Dict with per-bin and aggregate metrics
    """
    from scipy.stats import pearsonr

    metrics: Dict[str, float] = {}
    seq_len = len(mask_indices)
    masked_positions = np.where(mask_indices)[0]

    if len(masked_positions) == 0:
        return {"ext_completion_error": "No masked positions"}

    gt_masked = gt_embeddings[mask_indices]
    pred_masked = pred_embeddings[mask_indices]

    # Per-position cosine similarity
    cos_sims = np.array([
        np.dot(gt_masked[i], pred_masked[i]) /
        (np.linalg.norm(gt_masked[i]) * np.linalg.norm(pred_masked[i]) + 1e-10)
        for i in range(len(gt_masked))
    ])

    # Per-position MSE
    per_pos_mse = np.mean((gt_masked - pred_masked) ** 2, axis=1)

    metrics["ext_completion_cos_sim_mean"] = float(np.mean(cos_sims))
    metrics["ext_completion_cos_sim_median"] = float(np.median(cos_sims))
    metrics["ext_completion_mse_mean"] = float(np.mean(per_pos_mse))
    metrics["ext_completion_mse_p25"] = float(np.percentile(per_pos_mse, 25))
    metrics["ext_completion_mse_p75"] = float(np.percentile(per_pos_mse, 75))

    # Per-bin analysis: divide sequence into n_bins and compute error per bin
    bin_edges = np.linspace(0, seq_len, n_bins + 1, dtype=int)
    for b in range(n_bins):
        bin_mask = (masked_positions >= bin_edges[b]) & (masked_positions < bin_edges[b + 1])
        if bin_mask.sum() > 0:
            bin_cos = cos_sims[bin_mask]
            bin_mse = per_pos_mse[bin_mask]
            metrics[f"ext_completion_bin{b}_cos_sim"] = float(np.mean(bin_cos))
            metrics[f"ext_completion_bin{b}_mse"] = float(np.mean(bin_mse))
            metrics[f"ext_completion_bin{b}_n_positions"] = int(bin_mask.sum())

    # Position-error correlation: do errors increase with position?
    if len(masked_positions) > 5:
        corr, p_val = pearsonr(masked_positions.astype(float), per_pos_mse)
        metrics["ext_completion_position_error_corr"] = float(corr)
        metrics["ext_completion_position_error_pval"] = float(p_val)

    return metrics
