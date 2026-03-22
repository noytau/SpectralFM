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
