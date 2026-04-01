"""
Plot helper functions for SpectralFM evaluation.

All functions generate matplotlib figures and save to disk.
Separated from evaluation_runner.py for maintainability.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


# ------------------------------------------------------------------ #
#  Component Clustering Plots (Phase 4)                              #
# ------------------------------------------------------------------ #

def plot_component_clustering(
    embeddings: np.ndarray,
    component_ids: np.ndarray,
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Component Clustering",
) -> str:
    """
    Plot 2D projection of embeddings colored by component ID.

    Returns:
        Path to saved plot
    """
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA projection
    n_pca = min(50, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=n_pca)
    data_pca = pca.fit_transform(embeddings)
    data_2d = data_pca[:, :2]

    unique_comps = np.unique(component_ids)
    n_comps = len(unique_comps)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_comps, 2)))

    # Left: scatter by component
    for i, comp in enumerate(unique_comps):
        mask = component_ids == comp
        axes[0].scatter(
            data_2d[mask, 0], data_2d[mask, 1],
            c=[colors[i % len(colors)]], s=15, alpha=0.6,
            label=f"comp {comp}" if n_comps <= 20 else None,
        )
    axes[0].set_title(f"PCA projection (colored by component)\n"
                       f"ARI={metrics.get('comp_cluster_ari', 0):.3f}  "
                       f"NMI={metrics.get('comp_cluster_nmi', 0):.3f}")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    if n_comps <= 20:
        axes[0].legend(fontsize=7, ncol=2, loc="best")
    axes[0].grid(True, alpha=0.3)

    # Right: metrics bar chart
    metric_names = ["ARI", "NMI", "Silhouette", "V-Measure", "KNN Prec"]
    metric_keys = [
        "comp_cluster_ari", "comp_cluster_nmi", "comp_cluster_silhouette",
        "comp_cluster_vmeasure", "comp_cluster_knn_precision",
    ]
    values = [metrics.get(k, 0.0) for k in metric_keys]
    bar_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    bars = axes[1].bar(metric_names, values, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"Clustering Metrics (n_components={n_comps})")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Representation Geometry Plots (Phase 3)                           #
# ------------------------------------------------------------------ #

def plot_repr_geometry(
    metrics: Dict[str, float],
    singular_values: Optional[np.ndarray],
    save_path: str,
    title: str = "Representation Geometry",
) -> str:
    """
    Plot representation geometry metrics: singular value spectrum,
    effective rank, and summary metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: singular value spectrum
    if singular_values is not None:
        axes[0].semilogy(singular_values[:100], "b-", linewidth=1.5)
        axes[0].axhline(y=singular_values[0] * 0.01, color="r", linestyle="--",
                         alpha=0.5, label="1% of max")
        axes[0].set_title(f"Singular Value Spectrum\n"
                          f"Eff. Rank={metrics.get('repr_effective_rank', 0):.1f}")
        axes[0].set_xlabel("Component index")
        axes[0].set_ylabel("Singular value (log)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No SVD data", ha="center", va="center",
                     transform=axes[0].transAxes)

    # Middle: CKA scores
    cka_names = ["Linear CKA", "RBF CKA"]
    cka_values = [metrics.get("repr_cka_linear", 0), metrics.get("repr_cka_rbf", 0)]
    bars = axes[1].bar(cka_names, cka_values, color=["#1976D2", "#388E3C"],
                        edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, cka_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("CKA Score")
    axes[1].set_title("CKA (input ↔ embedding)\nKornblith et al. (ICML 2019)")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Right: other metrics
    other_names = ["Uniformity", "Alignment", "Vendi Score"]
    other_values = [
        metrics.get("repr_uniformity", 0),
        metrics.get("repr_alignment", 0),
        metrics.get("repr_vendi_score", 0),
    ]
    colors = ["#F57C00", "#7B1FA2", "#C62828"]
    bars = axes[2].bar(other_names, other_values, color=colors,
                        edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, other_values):
        y_pos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.15
        axes[2].text(bar.get_x() + bar.get_width() / 2, y_pos,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    axes[2].set_ylabel("Score")
    axes[2].set_title("Uniformity & Alignment\nWang & Isola (ICML 2020)")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Linear Probing Plots (Phase 1A)                                   #
# ------------------------------------------------------------------ #

def plot_probing_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "Linear Probing Comparison",
) -> str:
    """
    Bar chart comparing different probe types.

    Args:
        results: {"ridge": {"r2": 0.5, ...}, "knn": {"r2": 0.3}, ...}
        save_path: where to save
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    probe_names = list(results.keys())
    r2_values = [results[p].get("r2", 0) for p in probe_names]
    pearson_values = [results[p].get("pearson_r", 0) for p in probe_names]

    colors = plt.cm.Set2(np.linspace(0, 1, len(probe_names)))

    # R2
    bars = axes[0].bar(probe_names, r2_values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, r2_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", fontsize=9)
    axes[0].set_ylabel("R²")
    axes[0].set_title("Probing R² (parameter_0)")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Pearson
    bars = axes[1].bar(probe_names, pearson_values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, pearson_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", fontsize=9)
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Probing Pearson r (parameter_0)")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Few-Shot Plots (Phase 1C)                                         #
# ------------------------------------------------------------------ #

def plot_fewshot_curve(
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Few-Shot Learning Curve",
) -> str:
    """
    Plot metric vs label fraction.
    """
    import re

    fig, ax = plt.subplots(figsize=(8, 5))

    # Extract fewshot metrics
    pattern = re.compile(r"downstream_fewshot_(\d+)pct_(r2|acc)$")
    std_pattern = re.compile(r"downstream_fewshot_(\d+)pct_(r2|acc)_std$")

    fracs = []
    means = []
    stds = []
    for key, val in sorted(metrics.items()):
        m = pattern.match(key)
        if m:
            pct = int(m.group(1))
            fracs.append(pct)
            means.append(val)
            std_key = f"downstream_fewshot_{pct}pct_{m.group(2)}_std"
            stds.append(metrics.get(std_key, 0))

    if not fracs:
        plt.close(fig)
        return save_path

    fracs = np.array(fracs)
    means = np.array(means)
    stds = np.array(stds)

    ax.errorbar(fracs, means, yerr=stds, marker="o", linewidth=2,
                capsize=5, color="#1976D2", markersize=8)
    ax.fill_between(fracs, means - stds, means + stds, alpha=0.2, color="#1976D2")

    ax.set_xlabel("Label Fraction (%)")
    ax.set_ylabel("Score (R² or Accuracy)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Spectral-Domain Task Plots (Phase 2)                              #
# ------------------------------------------------------------------ #

def plot_parameter_verification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Parameter Verification (EER/ROC-AUC)",
) -> str:
    """
    ROC curve and score distributions for parameter verification.
    """
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Generate pairs and scores (same as in compute_parameter_verification)
    n = len(labels)
    rng = np.random.RandomState(42)
    n_pairs = min(10000, n * (n - 1) // 2)
    idx_a = rng.randint(0, n, n_pairs)
    idx_b = rng.randint(0, n, n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    label_dists = np.abs(labels[idx_a] - labels[idx_b])
    threshold = np.quantile(label_dists, 0.1)
    y_true = (label_dists <= threshold).astype(int)

    emb_a = embeddings[idx_a]
    emb_b = embeddings[idx_b]
    scores = np.sum(emb_a * emb_b, axis=1) / (
        np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1) + 1e-12
    )

    # Left: ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        axes[0].plot(fpr, tpr, "b-", linewidth=2,
                     label=f"AUC={metrics.get('spectral_param_verify_auc', 0):.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title("ROC Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    except Exception:
        axes[0].text(0.5, 0.5, "ROC unavailable", ha="center", va="center",
                     transform=axes[0].transAxes)

    # Right: score distributions
    axes[1].hist(scores[y_true == 1], bins=50, alpha=0.6, label="Same param", color="green")
    axes[1].hist(scores[y_true == 0], bins=50, alpha=0.6, label="Different param", color="red")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Score Distribution\nEER={metrics.get('spectral_param_verify_eer', 0):.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Attention Visualization Plots (Phase 6D)                          #
# ------------------------------------------------------------------ #

def plot_attention_maps(
    attention_maps: List[np.ndarray],
    save_path: str,
    title: str = "Attention Maps",
    max_layers: int = 6,
) -> str:
    """
    Plot attention maps from transformer layers.

    Args:
        attention_maps: list of [n_heads, L, L] arrays
        save_path: where to save
        max_layers: max number of layers to show
    """
    n_layers = min(len(attention_maps), max_layers)
    if n_layers == 0:
        return save_path

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for i in range(n_layers):
        attn = attention_maps[i]
        if attn.ndim == 4:  # [batch, heads, L, L]
            attn = attn[0]
        if attn.ndim == 3:  # [heads, L, L]
            attn = attn.mean(axis=0)  # average over heads

        im = axes[i].imshow(attn, aspect="auto", cmap="viridis")
        axes[i].set_title(f"Layer {i + 1}")
        axes[i].set_xlabel("Key position")
        if i == 0:
            axes[i].set_ylabel("Query position")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Failure Analysis Plots (Phase 6F)                                 #
# ------------------------------------------------------------------ #

def plot_failure_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    component_ids: Optional[np.ndarray],
    save_path: str,
    title: str = "Failure Analysis",
) -> str:
    """
    Plot error distributions and per-component breakdown.
    """
    errors = np.abs(y_true - y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: error histogram
    axes[0].hist(errors, bins=50, color="#1976D2", edgecolor="black", linewidth=0.5, alpha=0.8)
    axes[0].axvline(np.percentile(errors, 95), color="red", linestyle="--",
                     label=f"95th pct: {np.percentile(errors, 95):.3f}")
    axes[0].set_xlabel("Absolute Error")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Middle: Q-Q plot
    from scipy import stats as scipy_stats
    scipy_stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (errors vs normal)")
    axes[1].grid(True, alpha=0.3)

    # Right: per-component errors
    if component_ids is not None and len(np.unique(component_ids)) > 1:
        unique_comps = np.unique(component_ids)
        comp_means = [np.mean(errors[component_ids == c]) for c in unique_comps]
        axes[2].barh(
            [str(c) for c in unique_comps], comp_means,
            color="#FF9800", edgecolor="black", linewidth=0.5,
        )
        axes[2].set_xlabel("Mean Absolute Error")
        axes[2].set_ylabel("Component ID")
        axes[2].set_title("Per-Component Error")
        axes[2].grid(True, alpha=0.3, axis="x")
    else:
        axes[2].text(0.5, 0.5, "No component data", ha="center", va="center",
                     transform=axes[2].transAxes)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Scaling Law Plots (Phase 5A)                                      #
# ------------------------------------------------------------------ #

def plot_scaling_laws(
    dataset_sizes: List[int],
    metrics_per_size: Dict[str, List[float]],
    save_path: str,
    title: str = "Scaling Laws",
) -> str:
    """
    Log-linear scaling law plots.
    Kaplan et al. (2020) style.

    Args:
        dataset_sizes: [100, 1000, 10000, ...]
        metrics_per_size: {"metric_name": [val_at_100, val_at_1k, ...]}
    """
    n_metrics = len(metrics_per_size)
    if n_metrics == 0:
        return save_path

    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    colors = plt.cm.Set1(np.linspace(0, 1, n_metrics))

    for i, (metric_name, values) in enumerate(metrics_per_size.items()):
        ax = axes.flat[i]
        ax.semilogx(dataset_sizes[:len(values)], values, "o-",
                     color=colors[i], linewidth=2, markersize=8)
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_metrics, len(axes.flat)):
        axes.flat[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Metric Intercorrelation Heatmap (Phase 6G)                        #
# ------------------------------------------------------------------ #

def plot_metric_intercorrelation(
    metrics_across_checkpoints: List[Dict[str, float]],
    save_path: str,
    title: str = "Metric Intercorrelation",
) -> str:
    """
    Heatmap of metric-vs-metric Spearman correlation.
    """
    import pandas as pd

    df = pd.DataFrame(metrics_across_checkpoints)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].dropna(axis=1, how="any")
    df_numeric = df_numeric.loc[:, df_numeric.std() > 1e-10]

    if df_numeric.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Not enough metrics", ha="center", va="center")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    corr = df_numeric.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(max(10, corr.shape[0] * 0.4),
                                     max(8, corr.shape[0] * 0.35)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=corr.shape[0] <= 20, fmt=".2f",
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Training Dynamics Plot (Phase 6E)                                 #
# ------------------------------------------------------------------ #

def plot_training_dynamics(
    steps: List[int],
    metrics_over_time: List[Dict[str, float]],
    save_path: str,
    key_metrics: Optional[List[str]] = None,
    title: str = "Training Dynamics",
) -> str:
    """
    Plot metric trajectories over training steps.

    Args:
        steps: training step numbers
        metrics_over_time: list of metric dicts at each step
        key_metrics: specific metrics to plot (None = auto-select top 6)
    """
    import pandas as pd

    df = pd.DataFrame(metrics_over_time, index=steps)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if key_metrics:
        cols = [c for c in key_metrics if c in numeric_cols]
    else:
        # Auto-select: pick columns with highest variance (most interesting)
        variances = df[numeric_cols].var().dropna()
        # Normalize by mean to get coefficient of variation
        means = df[numeric_cols].mean().abs()
        cv = (variances / means.clip(lower=1e-10)).sort_values(ascending=False)
        cols = cv.head(6).index.tolist()

    if not cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No metrics to plot", ha="center", va="center")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    n_cols = min(3, len(cols))
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if len(cols) == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    colors = plt.cm.Set2(np.linspace(0, 1, len(cols)))

    for i, col in enumerate(cols):
        ax = axes.flat[i]
        vals = df[col].dropna()
        ax.plot(vals.index, vals.values, "o-", color=colors[i],
                linewidth=2, markersize=6)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(col)
        ax.set_title(col, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(vals) >= 3:
            z = np.polyfit(vals.index.astype(float), vals.values, 1)
            trend = np.poly1d(z)
            ax.plot(vals.index, trend(vals.index.astype(float)),
                     "--", color=colors[i], alpha=0.5, linewidth=1)

    for i in range(len(cols), len(axes.flat)):
        axes.flat[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Mask Sweep Plot (Phase 4B extension)                              #
# ------------------------------------------------------------------ #

def plot_mask_sweep(
    mask_probs: List[float],
    cos_sims: List[float],
    mses: Optional[List[float]],
    save_path: str,
    title: str = "Mask Probability Sweep",
) -> str:
    """
    Plot signal completion metrics across mask probabilities.
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = "#2196F3"
    color2 = "#F44336"

    ax1.plot(mask_probs, cos_sims, "o-", color=color1, linewidth=2,
             markersize=8, label="Cosine Similarity")
    ax1.set_xlabel("Mask Probability", fontsize=12)
    ax1.set_ylabel("Cosine Similarity", color=color1, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color1)

    if mses:
        ax2 = ax1.twinx()
        ax2.plot(mask_probs, mses, "s--", color=color2, linewidth=2,
                 markersize=8, label="MSE")
        ax2.set_ylabel("MSE", color=color2, fontsize=12)
        ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    if mses:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        ax1.legend(loc="best")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Transfer Evaluation Plot (Phase 5B)                               #
# ------------------------------------------------------------------ #

def plot_transfer_matrix(
    transfer_results: Dict[str, Dict[str, float]],
    save_path: str,
    metric_key: str = "transfer_acc",
    title: str = "Cross-Dataset Transfer Matrix",
) -> str:
    """
    Heatmap of transfer accuracy between datasets.
    """
    if not transfer_results:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No transfer results", ha="center", va="center")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    # Build bar chart of transfer pairs
    pairs = list(transfer_results.keys())
    values = [transfer_results[p].get(metric_key, 0) for p in pairs]

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 1.5), 5))
    colors = plt.cm.RdYlGn(np.array(values))

    bars = ax.bar(range(len(pairs)), values, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric_key)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Ablation Study Plot (Phase 6D)                                    #
# ------------------------------------------------------------------ #

def plot_ablation_study(
    ablation_results: Dict[str, List[Dict]],
    target_metric: str,
    save_path: str,
    title: str = "Ablation Studies",
) -> str:
    """
    Multi-panel plot showing how each ablation variable affects the target metric.
    """
    n_ablations = len(ablation_results)
    if n_ablations == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No ablation results", ha="center", va="center")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    n_cols = min(3, n_ablations)
    n_rows = (n_ablations + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_ablations == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    colors = plt.cm.tab10(np.linspace(0, 1, n_ablations))

    for i, (var_name, configs) in enumerate(ablation_results.items()):
        ax = axes.flat[i]

        var_values = []
        metric_values = []
        for cfg in configs:
            v = cfg.get(var_name)
            m = cfg.get(target_metric)
            if v is not None and m is not None:
                var_values.append(v)
                metric_values.append(m)

        if not var_values:
            ax.text(0.5, 0.5, f"No data for {var_name}", ha="center", va="center")
            continue

        # Try to convert to numeric for line plot
        try:
            var_numeric = [float(v) for v in var_values]
            sort_idx = np.argsort(var_numeric)
            var_sorted = [var_numeric[j] for j in sort_idx]
            met_sorted = [metric_values[j] for j in sort_idx]
            ax.plot(var_sorted, met_sorted, "o-", color=colors[i],
                     linewidth=2, markersize=8)
        except (ValueError, TypeError):
            # Categorical: use bar chart
            ax.bar(range(len(var_values)), metric_values, color=colors[i])
            ax.set_xticks(range(len(var_values)))
            ax.set_xticklabels([str(v) for v in var_values], rotation=30, ha="right")

        ax.set_xlabel(var_name)
        ax.set_ylabel(target_metric)
        ax.set_title(f"Ablation: {var_name}", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Mark the best
        if metric_values:
            best_idx = np.argmax(metric_values)
            ax.axhline(y=metric_values[best_idx], color=colors[i],
                       linestyle="--", alpha=0.3)

    for i in range(n_ablations, len(axes.flat)):
        axes.flat[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  Per-Bin Signal Completion Error (Phase 4B)                        #
# ------------------------------------------------------------------ #

def plot_per_bin_completion_error(
    bin_metrics: Dict[str, float],
    n_bins: int,
    save_path: str,
    title: str = "Per-Bin Signal Completion Error",
) -> str:
    """
    Bar chart showing completion error per positional bin.
    """
    bin_cos = []
    bin_mse = []
    bin_labels = []

    for b in range(n_bins):
        cos_key = f"ext_completion_bin{b}_cos_sim"
        mse_key = f"ext_completion_bin{b}_mse"
        if cos_key in bin_metrics:
            bin_cos.append(bin_metrics[cos_key])
            bin_mse.append(bin_metrics.get(mse_key, 0))
            bin_labels.append(f"Bin {b}")

    if not bin_cos:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No per-bin data", ha="center", va="center")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(bin_labels))
    ax1.bar(x, bin_cos, color="#2196F3", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Reconstruction Quality by Position")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x, bin_mse, color="#F44336", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax2.set_ylabel("MSE")
    ax2.set_title("Reconstruction Error by Position")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------ #
#  FE Decoder Reconstruction Plots (fe_dec_*)                        #
# ------------------------------------------------------------------ #

def plot_fe_decoder_reconstruction_samples(
    originals: np.ndarray,
    reconstructed: np.ndarray,
    per_cosine: np.ndarray,
    per_mse: np.ndarray,
    save_path: str,
    title: str = "FE Decoder — Reconstruction Samples",
    n_panels: int = 8,
    seed: int = 0,
) -> str:
    """
    Grid of N eval samples showing original vs reconstructed 245-d spectrogram.

    Each panel overlays:
      - blue solid:    original input
      - orange dashed: decoder reconstruction
      - annotation box: per-sample cosine similarity and MSE

    Args:
        originals:     float32 [N_eval, D]  ground-truth spectrogram
        reconstructed: float32 [N_eval, D]  decoder output
        per_cosine:    float [N_eval]       per-sample cosine similarity
        per_mse:       float [N_eval]       per-sample MSE
        save_path:     output PNG path
        title:         figure suptitle
        n_panels:      number of samples to show (default 8, arranged in 2 rows)
        seed:          for reproducible sample selection

    Returns:
        save_path
    """
    rng = np.random.default_rng(seed)
    N = len(originals)
    n_panels = min(n_panels, N)
    indices = sorted(rng.choice(N, n_panels, replace=False).tolist())

    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for panel_idx, sample_idx in enumerate(indices):
        row, col = divmod(panel_idx, n_cols)
        ax = axes[row, col]

        orig = originals[sample_idx]
        recon = reconstructed[sample_idx]
        cos = per_cosine[sample_idx]
        mse = per_mse[sample_idx]

        x = np.arange(len(orig))
        ax.plot(x, orig, color="#2196F3", linewidth=1.2, label="Original")
        ax.plot(x, recon, color="#FF9800", linewidth=1.2, linestyle="--",
                label="Reconstructed")
        ax.text(0.02, 0.98,
                f"cos={cos:.3f}\nMSE={mse:.4f}",
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.set_title(f"Sample {sample_idx}", fontsize=9)
        ax.set_xlabel("Position", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    for panel_idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(panel_idx, n_cols)
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_fe_decoder_score_distribution(
    per_cosine: np.ndarray,
    per_mse: np.ndarray,
    save_path: str,
    title: str = "FE Decoder — Score Distributions",
    run_name: str = "",
) -> str:
    """
    Two-panel distribution plot for the eval split.

    Left panel:  histogram of per-sample cosine similarity (higher is better)
    Right panel: histogram of per-sample MSE (lower is better)

    Both panels annotate mean ± std with vertical lines and a text box.

    Args:
        per_cosine: float [N_eval] per-sample cosine similarity
        per_mse:    float [N_eval] per-sample MSE
        save_path:  output PNG path
        title:      figure suptitle
        run_name:   checkpoint label shown in subtitle

    Returns:
        save_path
    """
    fig, (ax_cos, ax_mse) = plt.subplots(1, 2, figsize=(12, 5))

    def _hist_panel(ax, values, xlabel, color, higher_is_better=True):
        mean_v = np.mean(values)
        std_v = np.std(values)
        ax.hist(values, bins=40, color=color, alpha=0.7, edgecolor="white",
                density=True)
        ax.axvline(mean_v, color="black", linewidth=1.8, linestyle="-",
                   label=f"Mean {mean_v:.3f}")
        ax.axvline(mean_v - std_v, color="black", linewidth=1.0, linestyle="--",
                   alpha=0.6)
        ax.axvline(mean_v + std_v, color="black", linewidth=1.0, linestyle="--",
                   alpha=0.6, label=f"±Std {std_v:.3f}")
        ax.text(0.98, 0.97,
                f"Mean: {mean_v:.4f}\nStd:  {std_v:.4f}\nN={len(values)}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        direction = "(↑ better)" if higher_is_better else "(↓ better)"
        ax.set_title(f"{xlabel} {direction}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    _hist_panel(ax_cos, per_cosine, "Cosine Similarity", "#2196F3",
                higher_is_better=True)
    _hist_panel(ax_mse, per_mse, "MSE", "#F44336",
                higher_is_better=False)

    subtitle = f"Run: {run_name}" if run_name else ""
    full_title = f"{title}\n{subtitle}" if subtitle else title
    fig.suptitle(full_title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── FE vs Transformer comparison bar chart ───────────────────────────────────

def plot_fe_vs_transformer_comparison_bar_chart(
    fe_metrics_list: List[Dict],
    tr_metrics_list: List[Dict],
    run_names: List[str],
    output_path: str,
) -> None:
    """
    Grouped bar chart: for each checkpoint, two side-by-side bars
    (FE decoder = blue, Transformer decoder = orange) across 3 metric panels
    (cosine similarity, MSE, R²).
    """
    n = len(run_names)
    x = np.arange(n)
    w = 0.38

    metrics_cfg = [
        ("fe_dec_cosine_mean", "fe_dec_cosine_std",
         "Cosine Similarity (↑)", "Reconstruction Cosine (mean ± std)", True, ".3f"),
        ("fe_dec_mse", None,
         "MSE (↓)",               "Reconstruction MSE",                 False, ".4f"),
        ("fe_dec_r2",  None,
         "R² (↑)",                "Reconstruction R²",                  True,  ".3f"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("FE Decoder  vs  Transformer Decoder — Checkpoint Comparison",
                 fontsize=13, fontweight="bold")

    fe_color = "#3498DB"
    tr_color = "#E67E22"

    for ax, (key, std_key, ylabel, title, higher, fmt) in zip(axes, metrics_cfg):
        fe_vals = [m[key] for m in fe_metrics_list]
        tr_vals = [m[key] for m in tr_metrics_list]
        fe_errs = [m[std_key] for m in fe_metrics_list] if std_key else None
        tr_errs = [m[std_key] for m in tr_metrics_list] if std_key else None

        b_fe = ax.bar(x - w / 2, fe_vals, w,
                      yerr=fe_errs, capsize=4 if fe_errs else 0,
                      color=fe_color, alpha=0.85, label="FE decoder")
        b_tr = ax.bar(x + w / 2, tr_vals, w,
                      yerr=tr_errs, capsize=4 if tr_errs else 0,
                      color=tr_color, alpha=0.85, label="Transformer decoder")

        for bar, val in zip(b_fe, fe_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.003 if higher else 1e-5),
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=7.5,
                    color=fe_color, fontweight="bold")
        for bar, val in zip(b_tr, tr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.003 if higher else 1e-5),
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=7.5,
                    color=tr_color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=18, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved FE vs Transformer bar chart: {output_path}")


# ── Triple reconstruction comparison (original / FE recon / transformer recon) ─

def plot_reconstruction_triple(
    originals: np.ndarray,
    fe_reconstructed: np.ndarray,
    tr_reconstructed: np.ndarray,
    per_cosine_fe: np.ndarray,
    per_cosine_tr: np.ndarray,
    per_mse_fe: np.ndarray,
    per_mse_tr: np.ndarray,
    save_path: str,
    title: str = "FE vs Transformer Decoder — Reconstruction Comparison",
    n_panels: int = 8,
    seed: int = 0,
) -> str:
    """
    3-row grid:  Row 1 = original  |  Row 2 = FE recon  |  Row 3 = Transformer recon.

    All three rows share the same randomly selected eval samples so the
    reconstructions can be compared column by column.
    """
    rng = np.random.default_rng(seed)
    n_panels = min(n_panels, len(originals))
    indices = rng.choice(len(originals), size=n_panels, replace=False)

    fig, axes = plt.subplots(3, n_panels, figsize=(2.5 * n_panels, 8))
    if n_panels == 1:
        axes = axes[:, np.newaxis]

    row_labels = ["Original", "FE decoder", "Transformer\ndecoder"]
    row_colors = ["#555555", "#3498DB", "#E67E22"]

    for col, idx in enumerate(indices):
        orig = originals[idx]
        fe   = fe_reconstructed[idx]
        tr   = tr_reconstructed[idx]
        x    = np.arange(len(orig))

        for row, (signal, color) in enumerate([(orig, "#555555"), (fe, "#3498DB"), (tr, "#E67E22")]):
            ax = axes[row][col]
            ax.plot(x, orig,   color="#CCCCCC", linewidth=0.8, alpha=0.7)
            ax.plot(x, signal, color=color,     linewidth=1.0)
            ax.set_xlim(0, len(orig) - 1)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8, color=row_colors[row],
                              fontweight="bold")
            if row == 0:
                ax.set_title(f"Sample {idx}", fontsize=7)
            if row == 1:
                ax.set_xlabel(
                    f"cos={per_cosine_fe[idx]:.3f}  mse={per_mse_fe[idx]:.4f}",
                    fontsize=6,
                )
            if row == 2:
                ax.set_xlabel(
                    f"cos={per_cosine_tr[idx]:.3f}  mse={per_mse_tr[idx]:.4f}",
                    fontsize=6,
                )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── D1: Per-bin normalized RMSE heatmap ──────────────────────────────────────

def plot_per_bin_error_heatmap(
    originals_list: List[np.ndarray],
    reconstructed_list: List[np.ndarray],
    run_names: List[str],
    save_path: str,
) -> str:
    """
    D1 — Heatmap: rows = models, columns = frequency bins (245).
    Color encodes normalized RMSE = sqrt(mean((orig-recon)²)) / (std(orig) + 1e-8).
    Teal = low error, red = high error.
    """
    from matplotlib.colors import LinearSegmentedColormap

    n_models = len(run_names)
    n_bins = originals_list[0].shape[1]

    heatmap = np.zeros((n_models, n_bins))
    for i, (orig, recon) in enumerate(zip(originals_list, reconstructed_list)):
        rmse = np.sqrt(np.mean((orig - recon) ** 2, axis=0))
        heatmap[i] = rmse / (np.std(orig, axis=0) + 1e-8)

    cmap = LinearSegmentedColormap.from_list(
        "teal_red", ["#00897B", "#FFF9C4", "#C62828"]
    )

    fig_h = max(2.5, n_models * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    im = ax.imshow(heatmap, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Normalized RMSE  (lower = better)", shrink=0.8)

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(run_names, fontsize=10)
    ax.set_xlabel("Frequency Bin (0 – 244)", fontsize=11)
    ax.set_title(
        "D1 — Per-Bin Normalized RMSE\n"
        r"RMSE$_k$ / std($x_k$)  per frequency bin $k$",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── D2: PCA component R² ─────────────────────────────────────────────────────

def plot_pca_component_r2(
    originals: np.ndarray,
    reconstructed_list: List[np.ndarray],
    run_names: List[str],
    save_path: str,
    n_components: int = 50,
) -> str:
    """
    D2 — Line chart of R² per PCA component (1–50).

    PCA is fitted once on the original eval spectrograms; both originals and
    each model's reconstructions are projected into that space.  R² is computed
    per component across all eval samples.

    The x-coordinate where each model's curve first crosses R²=0.5 is the
    "effective reconstruction depth" — annotated on the chart.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    proj_orig = pca.fit_transform(originals)          # [N, n_components]

    colors = ["#3498DB", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12",
              "#1ABC9C", "#E67E22", "#95A5A6"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (recon, name) in enumerate(zip(reconstructed_list, run_names)):
        proj_recon = pca.transform(recon)

        r2_per_comp = []
        for k in range(n_components):
            ss_res = np.sum((proj_orig[:, k] - proj_recon[:, k]) ** 2)
            ss_tot = np.sum((proj_orig[:, k] - proj_orig[:, k].mean()) ** 2)
            r2_per_comp.append(float(1.0 - ss_res / (ss_tot + 1e-10)))

        r2_arr = np.array(r2_per_comp)
        color = colors[i % len(colors)]
        ax.plot(range(1, n_components + 1), r2_arr,
                color=color, linewidth=2.0, label=name)

        # Mark effective depth (first component that drops below 0.5)
        crossing = next((k + 1 for k, v in enumerate(r2_arr) if v < 0.5), None)
        if crossing:
            ax.axvline(crossing, color=color, linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(crossing + 0.3, 0.53, str(crossing),
                    color=color, fontsize=8, va="bottom")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0,
               label="R²=0.5 threshold")
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlim(1, n_components)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xlabel("PCA Component Index", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title(
        "D2 — PCA Component R²\n"
        "Number at dotted line = effective reconstruction depth (first component below R²=0.5)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── D3: Residual scatter ──────────────────────────────────────────────────────

def plot_residual_scatter(
    originals_list: List[np.ndarray],
    reconstructed_list: List[np.ndarray],
    run_names: List[str],
    save_path: str,
    n_points: int = 2000,
) -> str:
    """
    D3 — Residual scatter: x = original value, y = original − reconstructed.

    2000 random (sample, bin) pairs are plotted per panel.
    A dashed trend line exposes any slope — a non-zero slope indicates the
    decoder systematically under-predicts high-amplitude bins (positive slope)
    or over-predicts them (negative slope).  Ideal: flat band at y=0.
    """
    n_models = len(run_names)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols

    colors = ["#3498DB", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12",
              "#1ABC9C", "#E67E22", "#95A5A6"]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    rng = np.random.default_rng(42)

    for i, (orig, recon, name) in enumerate(
        zip(originals_list, reconstructed_list, run_names)
    ):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        residuals = (orig - recon).ravel()
        orig_flat = orig.ravel()

        total = len(residuals)
        idx = rng.choice(total, size=min(n_points, total), replace=False)
        x, y = orig_flat[idx], residuals[idx]

        ax.scatter(x, y, alpha=0.15, s=4, color=colors[i % len(colors)],
                   rasterized=True)
        ax.axhline(0, color="black", linewidth=1.5)

        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, "k--", linewidth=1.5,
                label=f"slope = {m:+.3f}")

        ax.set_xlabel("Original value", fontsize=10)
        ax.set_ylabel("Residual (orig − recon)", fontsize=10)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    for j in range(n_models, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(
        "D3 — Residual Scatter\n"
        "Flat band → unbiased  |  Sloped band → systematic under/over-prediction",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── D4: FE decoder vs transformer decoder R² per sample ──────────────────────

def plot_fe_vs_transformer_r2(
    fe_per_r2_list: List[np.ndarray],
    transformer_per_r2_list: List[np.ndarray],
    run_names: List[str],
    save_path: str,
) -> str:
    """
    D4 — Scatter: FE decoder R² (x) vs transformer decoder R² (y) per sample.

    Points well below the diagonal indicate samples where the transformer
    discarded information that the FE still retained.
    """
    n_models = len(run_names)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols

    colors = ["#3498DB", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12",
              "#1ABC9C", "#E67E22", "#95A5A6"]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for i, (fe_r2, tr_r2, name) in enumerate(
        zip(fe_per_r2_list, transformer_per_r2_list, run_names)
    ):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        ax.scatter(fe_r2, tr_r2, alpha=0.35, s=8,
                   color=colors[i % len(colors)], rasterized=True)

        lo = min(float(fe_r2.min()), float(tr_r2.min())) - 0.05
        hi = max(float(fe_r2.max()), float(tr_r2.max())) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="y = x")

        frac_below = float(np.mean(tr_r2 < fe_r2))
        ax.text(
            0.05, 0.95,
            f"{frac_below * 100:.1f}% below diagonal\n"
            f"(transformer < FE decoder)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        ax.set_xlabel("FE decoder R² per sample", fontsize=10)
        ax.set_ylabel("Transformer decoder R² per sample", fontsize=10)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    for j in range(n_models, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle(
        "D4 — FE Decoder vs Transformer Decoder R² per sample\n"
        "Points below diagonal = samples where transformer lost info the FE had",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
