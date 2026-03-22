"""
Pure metric computation functions for SpectralFM evaluation.

This module contains stateless functions that compute metrics given numpy arrays.
All functions follow the signature: f(inputs, embeddings, ...) -> Dict[str, float]

Separated from evaluation_runner.py for maintainability (see efficiency-tips.mdc).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats as scipy_stats
from sklearn.metrics.pairwise import cosine_similarity
import warnings


# ------------------------------------------------------------------ #
#  KNN Retrieval & Neighbor Overlap (Phase 0A: audit-existing-evals) #
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


def neighbor_overlap(
    indices_a: np.ndarray, indices_b: np.ndarray
) -> float:
    """
    Fraction of KNN neighbors shared between two spaces.

    Args:
        indices_a: [N, k] neighbor indices from space A
        indices_b: [N, k] neighbor indices from space B

    Returns:
        Mean overlap fraction in [0, 1]
    """
    n = len(indices_a)
    overlaps = []
    for i in range(n):
        set_a = set(indices_a[i])
        set_b = set(indices_b[i])
        k = len(set_a)
        if k == 0:
            continue
        overlaps.append(len(set_a & set_b) / k)
    return float(np.mean(overlaps)) if overlaps else 0.0


def compute_knn_metrics(
    inputs: np.ndarray,
    embeddings: np.ndarray,
    k: int = 10,
) -> Dict[str, float]:
    """
    KNN-based embedding quality metrics.

    Ported from compute_stats.py: retrieve_k_similar_inputs_by_embedding,
    compare_embedding_vs_input_similarity.

    Metrics:
        knn_neighbor_overlap: fraction of shared neighbors between input and embedding space
        knn_emb_avg_sim: average cosine similarity to k-NN in embedding space
        knn_input_avg_sim: average cosine similarity to k-NN in input space
        knn_retrieval_improvement: emb_avg_sim - input_avg_sim

    Args:
        inputs: [N, 245] raw spectrograms
        embeddings: [N, 768] model embeddings
        k: number of neighbors

    Returns:
        Dict of metrics
    """
    n = len(inputs)
    k_actual = min(k, n - 1)
    if k_actual < 1:
        return {"knn_error": "Not enough samples for KNN"}

    input_indices, input_scores = knn_retrieval(inputs, inputs, k=k_actual, metric="cosine")
    emb_indices, emb_scores = knn_retrieval(embeddings, embeddings, k=k_actual, metric="cosine")

    overlap = neighbor_overlap(input_indices, emb_indices)
    emb_avg_sim = float(np.mean(emb_scores))
    input_avg_sim = float(np.mean(input_scores))

    return {
        "knn_neighbor_overlap": overlap,
        "knn_emb_avg_sim": emb_avg_sim,
        "knn_input_avg_sim": input_avg_sim,
        "knn_retrieval_improvement": emb_avg_sim - input_avg_sim,
        "knn_k": k_actual,
    }


# ------------------------------------------------------------------ #
#  Component-Based Clustering Metrics (Phase 4)                      #
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
        from sklearn.metrics.pairwise import euclidean_distances
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
            # Subsample for efficiency
            max_pairs = 5000
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
#  Representation Geometry (Phase 3)                                  #
# ------------------------------------------------------------------ #

def compute_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two representation matrices.
    Kornblith et al. (ICML 2019).

    Args:
        X: [N, D1] representation matrix
        Y: [N, D2] representation matrix

    Returns:
        CKA score in [0, 1]
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # HSIC
    XtX = X.T @ X  # [D1, D1]
    YtY = Y.T @ Y  # [D2, D2]
    XtY = X.T @ Y  # [D1, D2]

    hsic_xy = np.sum(XtY ** 2)
    hsic_xx = np.sum(XtX ** 2)
    hsic_yy = np.sum(YtY ** 2)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def compute_rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """
    RBF kernel CKA.
    Kornblith et al. (ICML 2019).
    """
    from sklearn.metrics.pairwise import rbf_kernel

    if sigma is None:
        # Use median heuristic
        from scipy.spatial.distance import pdist
        dists_x = pdist(X, metric="euclidean")
        dists_y = pdist(Y, metric="euclidean")
        sigma_x = np.median(dists_x) if len(dists_x) > 0 else 1.0
        sigma_y = np.median(dists_y) if len(dists_y) > 0 else 1.0
        sigma = (sigma_x + sigma_y) / 2.0
        if sigma < 1e-10:
            sigma = 1.0

    gamma = 1.0 / (2 * sigma ** 2)
    K = rbf_kernel(X, gamma=gamma)
    L = rbf_kernel(Y, gamma=gamma)

    # Center kernels
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H

    hsic_kl = np.sum(K_c * L_c)
    hsic_kk = np.sum(K_c * K_c)
    hsic_ll = np.sum(L_c * L_c)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-12:
        return 0.0
    return float(hsic_kl / denom)


def compute_effective_rank(embeddings: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Effective rank and collapse ratio of embedding matrix.
    Jing et al. (ICML 2022).

    Returns:
        (effective_rank, collapse_ratio, singular_values)
    """
    # Center embeddings
    centered = embeddings - embeddings.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)

    # Normalize singular values to form a probability distribution
    s_norm = s / (s.sum() + 1e-12)
    s_norm = s_norm[s_norm > 1e-12]  # Remove zeros

    # Effective rank = exp(entropy)
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    effective_rank = float(np.exp(entropy))

    # Collapse ratio = sigma_1 / sigma_k (last nonzero)
    if len(s) > 1 and s[-1] > 1e-12:
        collapse_ratio = float(s[0] / s[-1])
    else:
        collapse_ratio = float("inf")

    return effective_rank, collapse_ratio, s


def compute_uniformity_alignment(
    embeddings: np.ndarray,
    labels: np.ndarray,
    t: float = 2.0,
) -> Tuple[float, float]:
    """
    Uniformity and alignment scores.
    Wang & Isola (ICML 2020).

    Args:
        embeddings: [N, D] embeddings (will be L2-normalized)
        labels: [N,] integer labels for positive pairs (same label = positive)
        t: temperature for uniformity kernel

    Returns:
        (alignment, uniformity)
        Good representations: low alignment, high (less negative) uniformity
    """
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    z = embeddings / norms

    # Alignment: average squared L2 distance between positive pairs
    alignment_sum = 0.0
    n_pos = 0
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue
        # Subsample for efficiency
        if len(idx) > 100:
            idx = np.random.RandomState(42).choice(idx, 100, replace=False)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                alignment_sum += np.sum((z[idx[i]] - z[idx[j]]) ** 2)
                n_pos += 1

    alignment = float(alignment_sum / max(n_pos, 1))

    # Uniformity: log of average pairwise Gaussian kernel
    # Subsample for efficiency
    n = len(z)
    if n > 500:
        sub_idx = np.random.RandomState(42).choice(n, 500, replace=False)
        z_sub = z[sub_idx]
    else:
        z_sub = z

    sq_dists = np.sum((z_sub[:, None] - z_sub[None, :]) ** 2, axis=-1)
    # Exclude diagonal
    mask = ~np.eye(len(z_sub), dtype=bool)
    uniformity = float(np.log(np.mean(np.exp(-t * sq_dists[mask])) + 1e-12))

    return alignment, uniformity


def compute_vendi_score(embeddings: np.ndarray, q: float = 1.0) -> float:
    """
    Vendi Score: diversity of embedding distribution.
    Friedman & Dieng (ICML 2023).

    Args:
        embeddings: [N, D]
        q: order of the Vendi Score (1.0 = exp(entropy))

    Returns:
        Vendi Score (higher = more diverse)
    """
    # Compute cosine similarity kernel
    sim = cosine_similarity(embeddings)
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(sim)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    eigenvalues = eigenvalues / eigenvalues.sum()

    if q == 1.0:
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-12))
        return float(np.exp(entropy))
    else:
        return float(np.sum(eigenvalues ** q) ** (1.0 / (1.0 - q)))


def compute_repr_geometry_metrics(
    inputs: np.ndarray,
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all representation geometry metrics.

    Metrics:
        repr_cka_linear: Linear CKA(input, embedding)
        repr_cka_rbf: RBF CKA(input, embedding)
        repr_effective_rank: Effective rank of embedding matrix
        repr_collapse_ratio: sigma_1 / sigma_k
        repr_uniformity: Uniformity score
        repr_alignment: Alignment score (requires labels)
        repr_vendi_score: Vendi Score

    References:
        Kornblith et al. (ICML 2019), Wang & Isola (ICML 2020),
        Jing et al. (ICML 2022), Friedman & Dieng (ICML 2023)
    """
    metrics: Dict[str, float] = {}

    # Subsample for CKA efficiency
    n = len(inputs)
    if n > 1000:
        idx = np.random.RandomState(42).choice(n, 1000, replace=False)
        inputs_sub = inputs[idx]
        embeddings_sub = embeddings[idx]
        labels_sub = labels[idx] if labels is not None else None
    else:
        inputs_sub = inputs
        embeddings_sub = embeddings
        labels_sub = labels

    # CKA
    try:
        metrics["repr_cka_linear"] = compute_linear_cka(inputs_sub, embeddings_sub)
    except Exception as e:
        metrics["repr_cka_linear"] = 0.0

    try:
        metrics["repr_cka_rbf"] = compute_rbf_cka(inputs_sub, embeddings_sub)
    except Exception as e:
        metrics["repr_cka_rbf"] = 0.0

    # Effective rank
    try:
        eff_rank, collapse_ratio, _ = compute_effective_rank(embeddings)
        metrics["repr_effective_rank"] = eff_rank
        metrics["repr_collapse_ratio"] = min(collapse_ratio, 1e6)  # Cap for JSON
    except Exception:
        metrics["repr_effective_rank"] = 0.0
        metrics["repr_collapse_ratio"] = 0.0

    # Uniformity & Alignment
    if labels_sub is not None and len(np.unique(labels_sub)) >= 2:
        try:
            alignment, uniformity = compute_uniformity_alignment(embeddings_sub, labels_sub)
            metrics["repr_alignment"] = alignment
            metrics["repr_uniformity"] = uniformity
        except Exception:
            metrics["repr_alignment"] = 0.0
            metrics["repr_uniformity"] = 0.0
    else:
        # Compute uniformity only (no labels for alignment)
        try:
            _, uniformity = compute_uniformity_alignment(
                embeddings_sub,
                np.zeros(len(embeddings_sub), dtype=int),  # dummy labels
            )
            metrics["repr_uniformity"] = uniformity
        except Exception:
            metrics["repr_uniformity"] = 0.0

    # Vendi Score
    try:
        metrics["repr_vendi_score"] = compute_vendi_score(embeddings_sub)
    except Exception:
        metrics["repr_vendi_score"] = 0.0

    return metrics


# ------------------------------------------------------------------ #
#  Downstream Probing (Phase 1A)                                     #
# ------------------------------------------------------------------ #

def compute_linear_probing_metrics(
    X: np.ndarray,
    y: np.ndarray,
    probe_type: str = "ridge",
    n_folds: int = 5,
    task: str = "regression",
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
    from sklearn.model_selection import cross_val_predict, cross_val_score

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

        # Compute metrics
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


def compute_fewshot_metrics(
    X: np.ndarray,
    y: np.ndarray,
    fractions: List[float] = None,
    n_seeds: int = 5,
    task: str = "regression",
) -> Dict[str, float]:
    """
    Few-shot evaluation at different label fractions.
    Oquab et al./DINOv2 (2024).

    Args:
        X: [N, D] features
        y: [N,] labels
        fractions: label fractions to test
        n_seeds: number of random seeds per fraction
        task: 'regression' or 'classification'

    Returns:
        Dict of metrics with keys like downstream_fewshot_1pct_r2
    """
    if fractions is None:
        fractions = [0.01, 0.05, 0.10, 0.50]

    metrics: Dict[str, float] = {}
    n = len(y)

    for frac in fractions:
        n_labeled = max(10, int(frac * n))
        if n_labeled >= n:
            n_labeled = n

        scores = []
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            idx = rng.choice(n, n_labeled, replace=False)
            X_sub, y_sub = X[idx], y[idx]

            if task == "regression":
                from sklearn.linear_model import RidgeCV
                n_folds = min(5, n_labeled)
                model = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=n_folds)
                from sklearn.model_selection import cross_val_predict
                y_pred = cross_val_predict(model, X_sub, y_sub, cv=n_folds)
                ss_res = np.sum((y_sub - y_pred) ** 2)
                ss_tot = np.sum((y_sub - np.mean(y_sub)) ** 2)
                r2 = 1.0 - ss_res / (ss_tot + 1e-12)
                scores.append(r2)
            else:
                from sklearn.linear_model import RidgeClassifier
                from sklearn.model_selection import cross_val_score
                model = RidgeClassifier(alpha=1.0)
                n_folds = min(5, n_labeled)
                cv_scores = cross_val_score(model, X_sub, y_sub, cv=n_folds, scoring="accuracy")
                scores.append(float(np.mean(cv_scores)))

        pct_str = f"{int(frac * 100)}pct"
        metric_key = "r2" if task == "regression" else "acc"
        metrics[f"downstream_fewshot_{pct_str}_{metric_key}"] = float(np.mean(scores))
        metrics[f"downstream_fewshot_{pct_str}_{metric_key}_std"] = float(np.std(scores))

    return metrics


# ------------------------------------------------------------------ #
#  Spectral-Domain Tasks (Phase 2)                                   #
# ------------------------------------------------------------------ #

def compute_parameter_verification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    threshold_quantile: float = 0.1,
) -> Dict[str, float]:
    """
    Parameter verification task: determine if two spectrograms share
    similar parameter_0 values based on embedding similarity.

    Analogue of speaker verification (EER, ROC-AUC).

    Args:
        embeddings: [N, D]
        labels: [N,] continuous parameter_0 values
        threshold_quantile: quantile for defining "same parameter" pairs

    Returns:
        Dict with spectral_param_verify_eer, spectral_param_verify_auc
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    n = len(labels)
    # Create pairs
    max_pairs = 10000
    rng = np.random.RandomState(42)

    # Generate random pairs
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    idx_a = rng.randint(0, n, n_pairs)
    idx_b = rng.randint(0, n, n_pairs)
    # Remove self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    # Label distance
    label_dists = np.abs(labels[idx_a] - labels[idx_b])
    threshold = np.quantile(label_dists, threshold_quantile)

    # Binary: same parameter if distance < threshold
    y_true = (label_dists <= threshold).astype(int)

    # Embedding similarity as score
    emb_a = embeddings[idx_a]
    emb_b = embeddings[idx_b]
    scores = np.sum(emb_a * emb_b, axis=1) / (
        np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1) + 1e-12
    )

    metrics: Dict[str, float] = {}

    if len(np.unique(y_true)) < 2:
        metrics["spectral_param_verify_eer"] = 0.0
        metrics["spectral_param_verify_auc"] = 0.0
        return metrics

    # ROC-AUC
    try:
        metrics["spectral_param_verify_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        metrics["spectral_param_verify_auc"] = 0.0

    # EER
    try:
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        metrics["spectral_param_verify_eer"] = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    except Exception:
        metrics["spectral_param_verify_eer"] = 0.0

    return metrics


def compute_component_detection_metrics(
    embeddings: np.ndarray,
    component_ids: np.ndarray,
    n_folds: int = 5,
) -> Dict[str, float]:
    """
    Component detection: predict component type from embedding.
    Analogue of sound event detection.

    Metrics: F1-score (macro), mAP.

    Args:
        embeddings: [N, D]
        component_ids: [N,] integer component IDs
        n_folds: number of CV folds
    """
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import f1_score, average_precision_score
    from sklearn.preprocessing import label_binarize

    metrics: Dict[str, float] = {}
    unique_components = np.unique(component_ids)

    if len(unique_components) < 2:
        return {"spectral_comp_detect_error": "Need >= 2 components"}

    n_folds_actual = min(n_folds, len(component_ids))

    model = RidgeClassifier(alpha=1.0)
    y_pred = cross_val_predict(model, embeddings, component_ids, cv=n_folds_actual)
    metrics["spectral_comp_detect_f1"] = float(
        f1_score(component_ids, y_pred, average="macro", zero_division=0)
    )

    # mAP using one-vs-rest
    try:
        model.fit(embeddings, component_ids)
        decision = model.decision_function(embeddings)
        y_bin = label_binarize(component_ids, classes=unique_components)
        if y_bin.ndim == 1:
            y_bin = np.column_stack([1 - y_bin, y_bin])
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        metrics["spectral_comp_detect_map"] = float(
            average_precision_score(y_bin, decision, average="macro")
        )
    except Exception:
        metrics["spectral_comp_detect_map"] = 0.0

    return metrics


# ------------------------------------------------------------------ #
#  Statistical Significance (Phase 6A)                               #
# ------------------------------------------------------------------ #

def bootstrap_confidence_interval(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric.

    Args:
        values: [N,] sample values
        n_bootstrap: number of bootstrap samples
        confidence: confidence level
        seed: random seed

    Returns:
        (mean, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot_means.append(np.mean(values[idx]))

    boot_means = np.array(boot_means)
    alpha = (1 - confidence) / 2
    ci_low = float(np.quantile(boot_means, alpha))
    ci_high = float(np.quantile(boot_means, 1 - alpha))
    return float(np.mean(values)), ci_low, ci_high


def paired_bootstrap_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """
    Paired bootstrap hypothesis test.

    Returns p-value: probability that values_a - values_b <= 0
    """
    rng = np.random.RandomState(seed)
    n = len(values_a)
    diff = values_a - values_b
    count_leq_zero = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if np.mean(diff[idx]) <= 0:
            count_leq_zero += 1
    return float(count_leq_zero / n_bootstrap)


# ------------------------------------------------------------------ #
#  Efficiency Metrics (Phase 6B)                                     #
# ------------------------------------------------------------------ #

def compute_efficiency_metrics(
    model,
    sample_input: np.ndarray,
    device: str = "cpu",
    n_warmup: int = 5,
    n_measure: int = 20,
) -> Dict[str, float]:
    """
    Measure inference throughput and memory usage.

    Args:
        model: PyTorch model
        sample_input: [1, L] sample input tensor
        device: 'cpu' or 'cuda'
        n_warmup: warmup iterations
        n_measure: measurement iterations

    Returns:
        Dict with efficiency_throughput_samples_per_sec, efficiency_gpu_memory_mb
    """
    import torch
    import time

    metrics: Dict[str, float] = {}

    model.eval()
    tensor_input = torch.tensor(sample_input, dtype=torch.float32).to(device)
    if tensor_input.dim() == 1:
        tensor_input = tensor_input.unsqueeze(0)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                model(tensor_input)
            except Exception:
                break

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            start = time.perf_counter()
            try:
                model(tensor_input)
            except Exception:
                break
            if device != "cpu":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    if times:
        avg_time = np.mean(times)
        metrics["efficiency_throughput_samples_per_sec"] = float(1.0 / avg_time) if avg_time > 0 else 0.0
        metrics["efficiency_avg_inference_ms"] = float(avg_time * 1000)

    # GPU memory
    if device != "cpu" and torch.cuda.is_available():
        metrics["efficiency_gpu_memory_mb"] = float(
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )
    else:
        metrics["efficiency_gpu_memory_mb"] = 0.0

    return metrics


# ------------------------------------------------------------------ #
#  Domain-Specific Baselines (Phase 7)                               #
# ------------------------------------------------------------------ #

def compute_signal_processing_features(
    inputs: np.ndarray,
    method: str = "fft",
) -> np.ndarray:
    """
    Compute signal processing baseline features from spectrograms.

    Args:
        inputs: [N, 245] raw spectrograms
        method: 'fft', 'wavelet', 'moments', 'peaks', 'centroid'

    Returns:
        [N, D] feature matrix
    """
    n = len(inputs)

    if method == "fft":
        # FFT magnitudes (already spectral, so just normalize)
        return inputs / (np.linalg.norm(inputs, axis=1, keepdims=True) + 1e-12)

    elif method == "wavelet":
        try:
            import pywt
        except ImportError:
            # Fallback: use multi-scale averaging
            features = []
            for scale in [2, 4, 8, 16, 32]:
                kernel_size = min(scale, inputs.shape[1])
                kernel = np.ones(kernel_size) / kernel_size
                smoothed = np.array([np.convolve(x, kernel, mode="same") for x in inputs])
                features.append(smoothed)
            return np.concatenate(features, axis=1)

        features = []
        for x in inputs:
            coeffs = pywt.wavedec(x, "db4", level=4)
            flat = np.concatenate([c for c in coeffs])
            features.append(flat)
        max_len = max(len(f) for f in features)
        padded = np.zeros((n, max_len))
        for i, f in enumerate(features):
            padded[i, :len(f)] = f
        return padded

    elif method == "moments":
        # Statistical moments per sample
        features = np.column_stack([
            np.mean(inputs, axis=1),
            np.std(inputs, axis=1),
            scipy_stats.skew(inputs, axis=1),
            scipy_stats.kurtosis(inputs, axis=1),
            np.median(inputs, axis=1),
            np.max(inputs, axis=1) - np.min(inputs, axis=1),  # range
        ])
        return features

    elif method == "peaks":
        from scipy.signal import find_peaks
        features = []
        for x in inputs:
            peaks, props = find_peaks(x, height=0)
            if len(peaks) > 0:
                feat = [
                    len(peaks),
                    np.mean(x[peaks]),
                    np.std(x[peaks]),
                    np.mean(np.diff(peaks)) if len(peaks) > 1 else 0,
                    np.max(x[peaks]),
                ]
            else:
                feat = [0, 0, 0, 0, 0]
            features.append(feat)
        return np.array(features)

    elif method == "centroid":
        # Spectral centroid and bandwidth
        freq_bins = np.arange(inputs.shape[1])
        magnitudes = np.abs(inputs) + 1e-12
        centroids = np.sum(freq_bins * magnitudes, axis=1) / np.sum(magnitudes, axis=1)
        bandwidths = np.sqrt(
            np.sum(((freq_bins - centroids[:, None]) ** 2) * magnitudes, axis=1)
            / np.sum(magnitudes, axis=1)
        )
        return np.column_stack([centroids, bandwidths])

    else:
        raise ValueError(f"Unknown method: {method}")


# ------------------------------------------------------------------ #
#  Metric Intercorrelation (Phase 6G)                                #
# ------------------------------------------------------------------ #

def compute_metric_intercorrelation(
    metrics_across_checkpoints: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Compute correlation between all metrics across checkpoints.

    Args:
        metrics_across_checkpoints: List of metric dicts, one per checkpoint

    Returns:
        Dict with summary statistics about metric redundancy
    """
    import pandas as pd

    if len(metrics_across_checkpoints) < 3:
        return {"metric_corr_error": "Need >= 3 checkpoints for correlation analysis"}

    df = pd.DataFrame(metrics_across_checkpoints)
    # Drop non-numeric and constant columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].dropna(axis=1, how="any")
    # Remove constant columns
    df_numeric = df_numeric.loc[:, df_numeric.std() > 1e-10]

    if df_numeric.shape[1] < 2:
        return {"metric_corr_error": "Not enough varying metrics"}

    corr_matrix = df_numeric.corr(method="spearman")

    # Summary stats
    triu = np.triu_indices_from(corr_matrix.values, k=1)
    corr_values = corr_matrix.values[triu]

    metrics = {
        "metric_corr_mean_abs": float(np.mean(np.abs(corr_values))),
        "metric_corr_median_abs": float(np.median(np.abs(corr_values))),
        "metric_corr_n_highly_correlated": int(np.sum(np.abs(corr_values) > 0.9)),
        "metric_corr_n_metrics": int(df_numeric.shape[1]),
    }

    return metrics


# ================================================================== #
#  Phase 5A: Scaling Law Analysis                                    #
#  Ref: Kaplan et al. (2020) "Scaling Laws for Neural Language Models"#
# ================================================================== #

def fit_power_law(
    sizes: List[int],
    values: List[float],
) -> Dict[str, float]:
    """
    Fit a power law: metric = a * size^b + c.

    Uses log-log linear regression as initialization, then refines
    with scipy curve_fit. Falls back to log-log fit if curve_fit fails.

    Args:
        sizes: dataset sizes
        values: metric values at each size

    Returns:
        Dict with power law parameters and goodness-of-fit
    """
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr

    sizes = np.array(sizes, dtype=float)
    values = np.array(values, dtype=float)

    # Filter out zero/negative values for log
    valid = (sizes > 0) & np.isfinite(values) & (values > 0)
    if valid.sum() < 2:
        return {"power_law_fit_error": "Not enough valid data points"}

    log_sizes = np.log10(sizes[valid])
    log_values = np.log10(values[valid])

    # Log-log linear regression: log(y) = b*log(x) + log(a)
    coeffs = np.polyfit(log_sizes, log_values, 1)
    b_init, log_a_init = coeffs
    a_init = 10 ** log_a_init

    # R² of log-log fit
    pred_log = np.polyval(coeffs, log_sizes)
    ss_res = np.sum((log_values - pred_log) ** 2)
    ss_tot = np.sum((log_values - np.mean(log_values)) ** 2)
    r2_loglog = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result = {
        "power_law_exponent": float(b_init),
        "power_law_coefficient": float(a_init),
        "power_law_r2_loglog": float(r2_loglog),
    }

    # Try nonlinear fit: y = a * x^b + c
    try:
        def power_func(x, a, b, c):
            return a * np.power(x, b) + c

        popt, _ = curve_fit(
            power_func, sizes[valid], values[valid],
            p0=[a_init, b_init, 0],
            maxfev=5000,
        )
        pred = power_func(sizes[valid], *popt)
        ss_res = np.sum((values[valid] - pred) ** 2)
        ss_tot = np.sum((values[valid] - np.mean(values[valid])) ** 2)
        r2_nonlinear = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        result["power_law_a"] = float(popt[0])
        result["power_law_b"] = float(popt[1])
        result["power_law_c"] = float(popt[2])
        result["power_law_r2_nonlinear"] = float(r2_nonlinear)
    except Exception:
        pass  # Fall back to log-log fit

    return result


def compute_scaling_law_metrics(
    dataset_sizes: List[int],
    metrics_per_size: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    Analyze scaling behavior across dataset sizes.

    For each metric, fits a power law and reports the exponent + R².

    Args:
        dataset_sizes: list of dataset sizes (e.g. [100, 1k, 10k, ...])
        metrics_per_size: dict of metric_name -> values at each size

    Returns:
        Dict with scaling law fit parameters per metric
    """
    results = {}
    results["scaling_n_sizes"] = len(dataset_sizes)
    results["scaling_n_metrics"] = len(metrics_per_size)

    for metric_name, values in metrics_per_size.items():
        if len(values) != len(dataset_sizes):
            continue
        fit = fit_power_law(dataset_sizes, values)
        for k, v in fit.items():
            results[f"scaling_{metric_name}_{k}"] = v

    # Find metrics with strongest scaling (highest absolute exponent)
    exponents = {
        k.replace("scaling_", "").replace("_power_law_exponent", ""): v
        for k, v in results.items()
        if k.endswith("_power_law_exponent")
    }
    if exponents:
        best = max(exponents, key=lambda x: abs(exponents[x]))
        results["scaling_strongest_metric"] = best
        results["scaling_strongest_exponent"] = exponents[best]

    return results


# ================================================================== #
#  Phase 5B: Cross-Dataset Transfer Metrics                          #
# ================================================================== #

def compute_transfer_metrics(
    embeddings_source: np.ndarray,
    labels_source: np.ndarray,
    embeddings_target: np.ndarray,
    labels_target: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate transfer by training a probe on source and testing on target.

    Args:
        embeddings_source: [N_s, D] embeddings from source dataset
        labels_source: [N_s,] labels from source dataset
        embeddings_target: [N_t, D] embeddings from target dataset
        labels_target: [N_t,] labels from target dataset

    Returns:
        Dict with transfer accuracy, relative drop from in-domain
    """
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score

    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings_source)
    X_test = scaler.transform(embeddings_target)

    # Train on source
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(X_train, labels_source)

    # In-domain performance (cross-val on source)
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(clf, X_train, labels_source, cv=min(5, len(set(labels_source))),
                                 scoring="accuracy")

    # Transfer performance (test on target)
    y_pred = clf.predict(X_test)
    transfer_acc = accuracy_score(labels_target, y_pred)

    # Only compute F1 for shared labels
    shared_labels = set(labels_source) & set(labels_target)
    if shared_labels:
        mask = np.isin(labels_target, list(shared_labels))
        if mask.sum() > 0:
            transfer_f1 = f1_score(
                labels_target[mask], y_pred[mask],
                average="weighted", zero_division=0,
            )
        else:
            transfer_f1 = 0.0
    else:
        transfer_f1 = 0.0

    in_domain_acc = float(np.mean(cv_scores))
    relative_drop = (in_domain_acc - transfer_acc) / max(in_domain_acc, 1e-8)

    return {
        "transfer_in_domain_acc": in_domain_acc,
        "transfer_acc": float(transfer_acc),
        "transfer_f1_weighted": float(transfer_f1),
        "transfer_relative_drop": float(relative_drop),
        "transfer_n_shared_labels": len(shared_labels),
        "transfer_n_source_labels": len(set(labels_source)),
        "transfer_n_target_labels": len(set(labels_target)),
    }


# ================================================================== #
#  Phase 4B: Extended Signal Completion                              #
#  Per-bin error, mask sweep, mask type comparison                   #
# ================================================================== #

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

    metrics = {}
    seq_len = len(mask_indices)
    masked_positions = np.where(mask_indices)[0]

    if len(masked_positions) == 0:
        return {"ext_completion_error": "No masked positions"}

    # Overall metrics
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


def compute_mask_sweep_metrics(
    per_mask_results: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Aggregate metrics across a mask probability sweep.

    Args:
        per_mask_results: {"mask_prob_0.3": {metrics}, "mask_prob_0.5": {...}, ...}

    Returns:
        Summary metrics comparing different mask probabilities
    """
    metrics = {}
    mask_probs = []
    cos_sims = []
    mses = []

    for key, vals in sorted(per_mask_results.items()):
        # Extract mask prob from key
        try:
            prob = float(key.split("_")[-1])
        except (ValueError, IndexError):
            continue

        cos_sim = vals.get("completion_cos_sim_mean")
        mse = vals.get("completion_mse_mean")

        if cos_sim is not None:
            mask_probs.append(prob)
            cos_sims.append(cos_sim)
            if mse is not None:
                mses.append(mse)

    if len(mask_probs) < 2:
        return {"mask_sweep_error": "Need >= 2 mask probabilities"}

    metrics["mask_sweep_n_probs"] = len(mask_probs)
    metrics["mask_sweep_best_prob"] = mask_probs[int(np.argmax(cos_sims))]
    metrics["mask_sweep_best_cos_sim"] = float(max(cos_sims))
    metrics["mask_sweep_worst_cos_sim"] = float(min(cos_sims))
    metrics["mask_sweep_cos_sim_range"] = float(max(cos_sims) - min(cos_sims))

    # Correlation between mask prob and performance
    from scipy.stats import spearmanr
    corr, p_val = spearmanr(mask_probs, cos_sims)
    metrics["mask_sweep_prob_perf_corr"] = float(corr)
    metrics["mask_sweep_prob_perf_pval"] = float(p_val)

    return metrics


# ================================================================== #
#  Phase 6E: Training Dynamics Analysis                              #
# ================================================================== #

def compute_training_dynamics_metrics(
    steps: List[int],
    metrics_over_time: List[Dict[str, float]],
    key_metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Analyze metric trajectories across training steps.

    Computes convergence rate, stability, and monotonicity for each
    tracked metric.

    Args:
        steps: list of training steps/updates
        metrics_over_time: list of metric dicts at each step
        key_metrics: specific metrics to analyze (None = all numeric)

    Returns:
        Dict with training dynamics analysis
    """
    import pandas as pd
    from scipy.stats import spearmanr

    if len(steps) < 2:
        return {"dynamics_error": "Need >= 2 checkpoints"}

    df = pd.DataFrame(metrics_over_time, index=steps)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if key_metrics:
        numeric_cols = [c for c in numeric_cols if c in key_metrics]

    results = {"dynamics_n_checkpoints": len(steps)}

    for col in numeric_cols:
        vals = df[col].dropna().values
        step_subset = df[col].dropna().index.values

        if len(vals) < 2:
            continue

        prefix = f"dynamics_{col}"

        # Final value
        results[f"{prefix}_final"] = float(vals[-1])

        # Relative improvement: (final - first) / |first|
        if abs(vals[0]) > 1e-10:
            results[f"{prefix}_relative_improvement"] = float(
                (vals[-1] - vals[0]) / abs(vals[0])
            )

        # Monotonicity: Spearman correlation with step index
        corr, _ = spearmanr(np.arange(len(vals)), vals)
        results[f"{prefix}_monotonicity"] = float(corr)

        # Stability: coefficient of variation over last 30% of training
        last_third = vals[max(1, int(len(vals) * 0.7)):]
        if len(last_third) > 1 and abs(np.mean(last_third)) > 1e-10:
            results[f"{prefix}_late_stability"] = float(
                np.std(last_third) / abs(np.mean(last_third))
            )

        # Convergence point: first step where metric reaches 95% of final value
        # (for metrics that increase; inverted for decreasing metrics)
        if corr > 0:
            threshold = vals[-1] * 0.95
            converged = np.where(vals >= threshold)[0]
        else:
            threshold = vals[-1] * 1.05
            converged = np.where(vals <= threshold)[0]

        if len(converged) > 0:
            convergence_step = step_subset[converged[0]]
            results[f"{prefix}_convergence_step"] = int(convergence_step)
            results[f"{prefix}_convergence_fraction"] = float(
                converged[0] / len(vals)
            )

    return results


# ================================================================== #
#  Phase 6D: Ablation Study Analysis                                 #
# ================================================================== #

def compute_ablation_analysis(
    ablation_results: Dict[str, List[Dict[str, float]]],
    target_metric: str = "comp_cluster_ari",
) -> Dict[str, float]:
    """
    Analyze ablation study results.

    For each ablation variable, identifies the optimal setting
    and the sensitivity (range of target metric).

    Args:
        ablation_results: {"mask_prob": [{"mask_prob": 0.3, ...metrics...}, ...], ...}
        target_metric: primary metric to optimize

    Returns:
        Summary dict of ablation findings
    """
    results = {}

    for var_name, configs in ablation_results.items():
        if not configs:
            continue

        # Extract variable values and target metric
        var_values = []
        metric_values = []
        for cfg in configs:
            v = cfg.get(var_name)
            m = cfg.get(target_metric)
            if v is not None and m is not None:
                var_values.append(v)
                metric_values.append(m)

        if len(var_values) < 2:
            continue

        metric_values = np.array(metric_values)
        best_idx = np.argmax(metric_values)

        prefix = f"ablation_{var_name}"
        results[f"{prefix}_best_value"] = var_values[best_idx]
        results[f"{prefix}_best_{target_metric}"] = float(metric_values[best_idx])
        results[f"{prefix}_worst_{target_metric}"] = float(np.min(metric_values))
        results[f"{prefix}_range"] = float(np.max(metric_values) - np.min(metric_values))
        results[f"{prefix}_n_configs"] = len(var_values)

        # Sensitivity: std of metric across configs
        results[f"{prefix}_sensitivity"] = float(np.std(metric_values))

    return results
