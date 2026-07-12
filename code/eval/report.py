"""
Report Generator for SpectralFM Evaluation
--------------------------------------------
Generates a markdown report + self-contained HTML report (base64-embedded figures)
from eval results dict.
"""
from __future__ import annotations
import base64
import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

_ACCENT   = "#2c5f8a"
_DARK     = "#16213e"

# ── Figure helpers ─────────────────────────────────────────────────────────────

def _save_fig(fig, path: str) -> str:
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _fig_to_b64(path: str) -> str:
    """Read a saved PNG and return a data-URI string for HTML embedding."""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


# ── Per-eval plot functions ────────────────────────────────────────────────────

def _plot_cosine_similarity_maps(sim_dists: dict, output_dir: str, label: str = "") -> list:
    """
    Plot N×N cosine similarity heatmaps side-by-side (raw | softmax) for input and embedding space.
    Matches the original compute_stats.py visualisation (viridis heatmap, T=0.1 softmax).
    Samples are assumed sorted by stack_idx — diagonal blocks = same-observation spectra.
    """
    figures = []
    suffix = f"_{label}" if label else ""
    tag = f" [{label}]" if label else ""
    T = 0.1

    for key, space_label, fname_base in [
        ("sim_matrix_inp", "Input Space",     f"cosine_map_inp{suffix}"),
        ("sim_matrix_emb", "Embedding Space", f"cosine_map_emb{suffix}"),
    ]:
        mat = sim_dists.get(key)
        if mat is None or len(mat) == 0:
            continue

        mat = np.array(mat, dtype=np.float32)
        softmax_mat = np.exp(mat / T) / np.exp(mat / T).sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        im0 = axes[0].imshow(mat, cmap="viridis", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_title(f"Cosine Similarity — {space_label}{tag}", fontsize=10)
        axes[0].set_xlabel("Sample index")
        axes[0].set_ylabel("Sample index")

        im1 = axes[1].imshow(softmax_mat, cmap="viridis", aspect="auto", interpolation="nearest")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title(f"Cosine Similarity softmax (T={T}) — {space_label}{tag}", fontsize=10)
        axes[1].set_xlabel("Sample index")
        axes[1].set_ylabel("Sample index")

        fig.tight_layout()
        path = _save_fig(fig, os.path.join(output_dir, f"{fname_base}.png"))
        figures.append((f"Cosine similarity maps — {space_label}{tag}", path))

    return figures


def _plot_cosine_sim_distributions(sim_dists: dict, output_dir: str, label: str = "") -> list:
    """
    Plot cosine similarity distributions for intra-stack, random inter-stack,
    and preset-index inter-stack pairs — for both input and embedding spaces.
    """
    figures = []
    if not sim_dists:
        return figures

    suffix = f"_{label}" if label else ""

    for space, space_key, title_space in [
        ("Embedding space", "emb", "Embedding Space"),
        ("Input space",     "inp", "Input Space"),
    ]:
        intra   = sim_dists.get(f"intra_stack_{space_key}", np.array([]))
        rand    = sim_dists.get(f"random_inter_{space_key}", np.array([]))
        preset  = sim_dists.get(f"preset_inter_{space_key}", np.array([]))

        if not any(len(x) for x in [intra, rand, preset]):
            continue

        fig, ax = plt.subplots(figsize=(9, 4))
        bins = np.linspace(-1, 1, 60)

        def _mean_label(arr, name):
            m = np.mean(arr) if len(arr) else float("nan")
            return f"{name} (μ={m:.3f}, n={len(arr)})"

        if len(intra):
            ax.hist(intra,  bins=bins, alpha=0.55, color="steelblue",
                    label=_mean_label(intra,  "Intra-stack"))
        if len(rand):
            ax.hist(rand,   bins=bins, alpha=0.55, color="tomato",
                    label=_mean_label(rand,   "Random inter-stack"))
        if len(preset):
            ax.hist(preset, bins=bins, alpha=0.55, color="seagreen",
                    label=_mean_label(preset, "Same-index inter-stack"))

        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"Cosine Similarity Distributions — {title_space}{' [' + label + ']' if label else ''}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        fname = f"cosine_sim_{space_key}{suffix}.png"
        path = _save_fig(fig, os.path.join(output_dir, fname))
        figures.append((f"Cosine similarity distributions ({space}){' — ' + label if label else ''}", path))

    return figures


def _plot_embedding_similarity(results: dict, output_dir: str) -> list:
    figures = []
    rdf = results.get("results_df")

    sim_dists = results.get("similarity_distributions", {})
    figures += _plot_cosine_similarity_maps(sim_dists, output_dir)
    figures += _plot_cosine_sim_distributions(sim_dists, output_dir)

    if rdf is None:
        return figures

    fig, ax = plt.subplots(figsize=(9, 5))
    max_k = max(
        rdf["embedding_stack_matches"].apply(len).max(),
        rdf["input_stack_matches"].apply(len).max(),
    )
    bins = np.arange(-0.5, max_k + 1.5, 0.5)
    ax.hist(rdf["input_stack_matches"].apply(len), bins=bins, alpha=0.55, label="Input-space matches")
    ax.hist(rdf["embedding_stack_matches"].apply(len), bins=bins, alpha=0.55, label="Embedding-space matches")
    ax.set_xlabel("Same-stack neighbors in top-k")
    ax.set_ylabel("Count")
    ax.set_title("Embedding vs Input-Space: Same-Stack Neighbor Counts")
    ax.legend()
    ax.grid(alpha=0.25)
    path = _save_fig(fig, os.path.join(output_dir, "emb_similarity_histogram.png"))
    figures.append(("Embedding similarity histogram", path))

    fig, ax = plt.subplots(figsize=(8, 4))
    scores = rdf["match_score"].dropna()
    ax.hist(scores, bins=20, color=_ACCENT, alpha=0.85)
    ax.axvline(scores.mean(), color="red", linestyle="--", label=f"Mean = {scores.mean():.1f}")
    ax.set_xlabel("Match Score (0–100)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Match Scores")
    ax.legend()
    path = _save_fig(fig, os.path.join(output_dir, "emb_match_score_dist.png"))
    figures.append(("Match score distribution", path))

    return figures


def _plot_signal_reconstruction(results: dict, output_dir: str) -> list:
    """
    True signal reconstruction figures, styled after compare_fe_vs_trans_recon.py:
      1. Per-sample overlay panel — target (black) + FE recon (blue) / TR recon (red),
         one column per available pathway, y-axis fixed to target range per row.
      2. Per-sample MSE bars (log scale) comparing the pathways.
    """
    figures = []
    if results.get("skipped"):
        return figures

    panel = results.get("panel") or {}
    target  = panel.get("target")
    indices = panel.get("indices") or []
    names   = panel.get("names") or []
    rdf     = results.get("results_df")

    _PATHWAY_STYLE = [
        ("fe",   "FE recon",          "#1f77b4"),
        ("proj", "Projection recon",  "#2ca02c"),
        ("tr",   "Transformer recon", "#d62728"),
    ]
    pathways = [
        (label, panel[f"pred_{key}"], color, f"{key}_mse")
        for key, label, color in _PATHWAY_STYLE
        if panel.get(f"pred_{key}") is not None
    ]
    if target is None or not pathways:
        return figures

    n = len(indices)
    T = target.shape[1]
    fig, axes = plt.subplots(n, len(pathways), figsize=(6.5 * len(pathways), 1.9 * n),
                             squeeze=False)
    for r in range(n):
        tgt = target[r]
        pad = 0.1 * max(tgt.max() - tgt.min(), 0.1)
        ylo, yhi = tgt.min() - pad, tgt.max() + pad
        for c, (pw_label, pred, color, mse_col) in enumerate(pathways):
            ax = axes[r, c]
            ax.plot(tgt, color="black", lw=1.6, label="target", alpha=0.9)
            ax.plot(pred[r], color=color, lw=1.1, label=pw_label, alpha=0.9)
            ax.set_xlim(0, T - 1)
            ax.set_ylim(ylo, yhi)
            ax.tick_params(labelsize=6, labelleft=(c == 0))
            if c == 0:
                ax.set_ylabel(f"idx {indices[r]}", fontsize=9, rotation=0, ha="right", labelpad=22)
            mse_val = ""
            if rdf is not None and mse_col in rdf.columns:
                mse_val = f"    MSE = {rdf[rdf['index'] == indices[r]][mse_col].iloc[0]:.2e}"
            title = f"{names[r][-40:]}{mse_val}" if c == 0 else f"{pw_label}{mse_val}"
            ax.set_title(title, fontsize=7, loc="left")
            if r == 0:
                ax.legend(fontsize=7, loc="upper right")

    mean_bits = [
        f"{label} mean MSE = {results[f'recon_{key}_mse_mean']:.3e}"
        for key, label, _ in _PATHWAY_STYLE
        if f"recon_{key}_mse_mean" in results
    ]
    fig.suptitle(
        "Signal Reconstruction — per-pathway (FE / projection / transformer)\n"
        + "   |   ".join(mean_bits)
        + f"\nnormalize={results.get('normalize')}   "
        "(y-axis fixed to target range per row; predictions outside are clipped)",
        fontsize=9, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = _save_fig(fig, os.path.join(output_dir, "recon_overlay.png"))
    figures.append(("Reconstruction overlay — target vs per-pathway recon", path))

    # Per-sample MSE bars (log scale), one bar group per pathway present
    bar_cols = [(label, color, mse_col) for label, _, color, mse_col in pathways
                if rdf is not None and mse_col in rdf.columns]
    if bar_cols:
        sub = rdf[rdf["index"].isin(indices)]
        x = np.arange(len(sub))
        w = 0.8 / len(bar_cols)
        fig, ax = plt.subplots(figsize=(9, 4))
        for j, (label, color, mse_col) in enumerate(bar_cols):
            offset = (j - (len(bar_cols) - 1) / 2) * w
            ax.bar(x + offset, sub[mse_col], w, color=color,
                   label=f"{label}  (mean {rdf[mse_col].mean():.2e})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"idx {i}" for i in sub["index"]], fontsize=8)
        ax.set_ylabel("per-sample MSE", fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both", axis="y")
        ax.legend(fontsize=8)
        ax.set_title("Per-sample reconstruction MSE — per pathway", fontsize=10)
        fig.tight_layout()
        path = _save_fig(fig, os.path.join(output_dir, "recon_mse_bars.png"))
        figures.append(("Per-sample reconstruction MSE", path))

    return figures


def _plot_noise_robustness(results: dict, output_dir: str) -> list:
    figures = []
    summary = results.get("summary", {})
    if not summary:
        return figures

    fig, ax = plt.subplots(figsize=(9, 4))
    noise_types = list(summary.keys())
    values = [summary[k] for k in noise_types]
    bars = ax.barh(noise_types, values, color="teal", alpha=0.8)
    ax.set_xlabel("Mean Cosine Similarity (clean vs noisy embedding)")
    ax.set_title("Noise Robustness: Embedding Stability per Noise Type")
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")
    path = _save_fig(fig, os.path.join(output_dir, "noise_robustness.png"))
    figures.append(("Noise robustness bar chart", path))
    return figures


def _ss_annotate(ax, sim_matrix: np.ndarray) -> None:
    """Mean/Std box from upper-triangle values — matches eval_plots.py exactly."""
    triu = np.triu_indices_from(sim_matrix, k=1)
    vals = sim_matrix[triu]
    ax.text(
        0.02, 0.98,
        f"Mean: {vals.mean():.3f}\nStd:  {vals.std():.3f}",
        transform=ax.transAxes, fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


_DS_SHORT = {
    "single_channel_all": "SC All",
    "multi_channel":      "Multi Ch.",
    "sampled_data":       "Sampled",
    "labeled_data":       "Labeled",
}


def _ss_heatmap(ax, sim_matrix, title: str, ylabel: str = None,
                color: str = "#333333", vmin: float = 0.0, vmax: float = 1.0,
                groups: list = None) -> None:
    """Single structured-similarity heatmap cell with optional dataset block annotations."""
    import seaborn as sns
    if sim_matrix is not None and sim_matrix.shape[0] > 1:
        sns.heatmap(
            sim_matrix, ax=ax, cmap="viridis",
            xticklabels=False, yticklabels=False,
            vmin=vmin, vmax=vmax,
            cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
        )
        _ss_annotate(ax, sim_matrix)

        # Dataset block boundaries and labels
        if groups:
            boundaries, starts, labels = [], [0], []
            prev = groups[0]
            for i, g in enumerate(groups[1:], 1):
                if g != prev:
                    boundaries.append(i)
                    starts.append(i)
                    prev = g
            ends = boundaries + [len(groups)]
            midpoints = [(s + e) / 2 for s, e in zip(starts, ends)]
            tick_labels = [_DS_SHORT.get(groups[s], groups[s]) for s in starts]

            for b in boundaries:
                ax.axhline(b, color="white", linewidth=1.5, alpha=0.9)
                ax.axvline(b, color="white", linewidth=1.5, alpha=0.9)

            ax.set_xticks(midpoints)
            ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right")
            ax.set_yticks(midpoints)
            ax.set_yticklabels(tick_labels, fontsize=7, rotation=0)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12, color="#E74C3C")
        ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", color=color)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def _plot_structured_similarity_maps(results: dict, output_dir: str, label: str = "") -> list:
    """
    2×2 cosine similarity heatmaps for one checkpoint — all 4 representation stages:
      (0,0) Input space     (0,1) FE output  (raw conv, 512-dim)
      (1,0) Projection      (1,1) Embedding  (transformer output)
    All panels use vmin=-1, vmax=1 for consistent, unclipped comparison.
    """
    panels = [
        ("sim_matrix_inp",  "Input Space",        "Stage →"),
        ("sim_matrix_fe",   "FE Output (512d)",   None),
        ("sim_matrix_proj", "Projection (768d)",  "Stage →"),
        ("sim_matrix_emb",  "Embedding (768d)",   None),
    ]

    suffix   = f"_{label}" if label else ""
    run_name = label or "checkpoint"

    groups = results.get("groups") or []

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), squeeze=False)
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        mat = results.get(key)
        mat = np.array(mat, dtype=np.float32) if mat is not None else None
        _ss_heatmap(ax, mat, title, ylabel=ylabel, vmin=0.0, vmax=1.0, groups=groups)

    fig.suptitle(f"Cosine Similarity Maps — {run_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = _save_fig(fig, os.path.join(output_dir, f"struct_sim{suffix}.png"))
    return [(f"Cosine similarity maps — {run_name}", path)]


def _plot_label_regression_comparison(cdf: pd.DataFrame, output_dir: str) -> list:
    """
    Label regression bars across checkpoints.
    Exact recreation of eval_label_regression._plot_label_regression:
    Panel 1: input vs embedding R² grouped bars; Panel 2: ΔR² green/red bars.
    """
    needed = {"label_reg_input_r2", "label_reg_emb_r2", "label_reg_improvement_r2"}
    if not needed.issubset(cdf.columns):
        return []

    labels = cdf["checkpoint"].tolist()
    r2_in  = cdf["label_reg_input_r2"].fillna(0.0).tolist()
    r2_emb = cdf["label_reg_emb_r2"].fillna(0.0).tolist()
    delta  = cdf["label_reg_improvement_r2"].fillna(0.0).tolist()
    n = len(labels)
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 3), 5))
    fig.suptitle("Label Regression — Ridge Probe (parameter_0)", fontsize=13, fontweight="bold")

    # Panel 1: side-by-side R²
    ax = axes[0]
    b1 = ax.bar(x - w / 2, r2_in, w, color="#90CAF9", edgecolor="white", label="Input R²")
    b2 = ax.bar(x + w / 2, r2_emb, w, color="#1565C0", edgecolor="white", label="Embedding R²")
    for bar, v in zip(b1, r2_in):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8, color="#555")
    for bar, v in zip(b2, r2_emb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8, color="#1565C0", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("Input vs Embedding R²")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    # Panel 2: ΔR²
    ax2 = axes[1]
    colors = ["#2E7D32" if d > 0 else "#C62828" for d in delta]
    bars = ax2.bar(x, delta, 0.5, color=colors, edgecolor="white")
    for bar, v in zip(bars, delta):
        y_off = 0.001 if v >= 0 else -0.003
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off,
                 f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("ΔR² (embedding − input)")
    ax2.set_title("Improvement over Raw Input")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = _save_fig(fig, os.path.join(output_dir, "label_regression_comparison.png"))
    return [("Label regression — ridge probe (parameter_0)", path, "")]


def _shorten_label(label: str, max_len: int = 28) -> str:
    """Shorten a long checkpoint label for axis ticks (keep head + tail)."""
    if len(label) <= max_len:
        return label
    half = (max_len - 1) // 2
    return label[:half] + "…" + label[-half:]


def _plot_noise_robustness_comparison(cdf: pd.DataFrame, output_dir: str) -> list:
    """
    Noise robustness grouped bars across checkpoints.
    Recreation of evaluation_runner.plot_noise_robustness_comparison panel 1
    (embedding similarity per noise type). Legend sits outside the axes and
    checkpoint labels are shortened so nothing overlaps the title.
    """
    noise_cols = [c for c in cdf.columns if c.startswith("noise_")]
    if not noise_cols:
        return []

    run_labels = [_shorten_label(l) for l in cdf["checkpoint"].tolist()]
    x = np.arange(len(run_labels))
    width = 0.8 / len(noise_cols)

    fig, ax = plt.subplots(figsize=(max(8, len(run_labels) * 2.5 + 3), 4.5))
    for i, col in enumerate(noise_cols):
        ax.bar(x + i * width, cdf[col].fillna(0.0), width, label=col.replace("noise_", ""))

    ax.set_ylabel("Embedding Similarity (higher = more robust)", fontsize=9)
    ax.set_title("Noise Robustness: Embedding Similarity", fontsize=11)
    ax.set_xticks(x + width * (len(noise_cols) - 1) / 2)
    ax.set_xticklabels(run_labels, rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    path = _save_fig(fig, os.path.join(output_dir, "noise_robustness_comparison.png"))
    return [("Noise robustness — embedding similarity per noise type", path, "")]


def _plot_noise_example_grid(noise_results: dict, output_dir: str, label: str = "",
                             sample_idx: int | None = None) -> list:
    """
    Noise Example Grid — one cell per noise type showing the clean vs noisy input
    signal together with that noise type's embedding similarity.
    Style restored from compute_stats.plot_noisy_vs_clean_spectrogram (clean=black,
    noisy overlay, per-cell title), laid out as a 2×3 grid over all 6 noise types.
    """
    rdf   = noise_results.get("results_df")
    clean = noise_results.get("clean_data")
    noisy = noise_results.get("noisy_data") or {}
    if rdf is None or rdf.empty or clean is None or not noisy:
        return []

    noise_types = [c for c in rdf.columns if c in noisy]
    if not noise_types:
        return []

    # Representative sample: median embedding similarity on the first noise type
    if sample_idx is None:
        first = rdf[noise_types[0]]
        sample_idx = int((first - first.median()).abs().idxmin())

    suffix   = f"_{label}" if label else ""
    run_name = label or "checkpoint"
    n_cols = 3
    n_rows = (len(noise_types) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows), squeeze=False)
    fig.suptitle(f"Noise Example Grid — idx {sample_idx} — {run_name}",
                 fontsize=12, fontweight="bold")

    for i, nt in enumerate(noise_types):
        ax = axes[i // n_cols][i % n_cols]
        ax.plot(clean[sample_idx], label="Clean", color="black", linewidth=1.2)
        ax.plot(noisy[nt][sample_idx], label=nt.replace("_", " "), alpha=0.8)
        emb_sim = rdf.iloc[sample_idx][nt]
        ax.set_title(f"{nt.replace('_', ' ')} | emb sim = {emb_sim:.4f}", fontsize=9)
        ax.set_xlabel("Time / Frequency bins", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    for j in range(len(noise_types), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.tight_layout()
    path = _save_fig(fig, os.path.join(output_dir, f"noise_example_grid{suffix}.png"))
    return [(f"Noise example grid — {run_name}", path)]


def _plot_clustering_scatter(clust_results: dict, output_dir: str, label: str = "") -> list:
    """
    Per-checkpoint KMeans cluster scatter in t-SNE (and UMAP if installed) space.
    Exact recreation of compute_stats.cluster_vectors visualisation:
    PCA(50) → t-SNE/UMAP(2), seaborn scatterplot, hls palette, s=50, no legend.
    """
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    embeddings  = clust_results.get("embeddings")
    pred_labels = clust_results.get("pred_labels")
    if embeddings is None or pred_labels is None:
        return []

    suffix     = f"_{label}" if label else ""
    run_name   = label or "checkpoint"
    n_clusters = len(np.unique(pred_labels))
    figures    = []

    vectors_np = np.asarray(embeddings)
    # PCA to speed up t-SNE
    n_comp = min(50, vectors_np.shape[0] - 1, vectors_np.shape[1])
    pca_embeddings = PCA(n_components=n_comp).fit_transform(vectors_np)

    palette = sns.color_palette("hls", n_colors=n_clusters)

    projections = []
    tsne_emb = TSNE(n_components=2, random_state=42).fit_transform(pca_embeddings)
    projections.append(("tSNE", tsne_emb))
    try:
        import umap
        umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(pca_embeddings)
        projections.append(("UMAP", umap_emb))
    except ImportError:
        pass

    for proj_name, emb2d in projections:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=emb2d[:, 0], y=emb2d[:, 1], hue=pred_labels,
                        palette=palette, s=50, legend=False, ax=ax)
        ax.set_title(f"KMeans Clustering (on embeddings), Visualized in {proj_name} — n={n_clusters}\n{run_name}")
        ax.set_xlabel(f"{proj_name}-1")
        ax.set_ylabel(f"{proj_name}-2")
        ax.grid(True)
        plt.tight_layout()
        fname = f"kmeans_clustered_then_{proj_name.lower()}{suffix}.png"
        path = _save_fig(fig, os.path.join(output_dir, fname))
        figures.append((f"KMeans clusters in {proj_name} space — {run_name}", path))

    return figures


_NOISE_PLOT_TYPES = ["gaussian_std", "gaussian_mean", "gain_low", "gain_high"]


def _plot_noise_examples(noise_results: dict, output_dir: str, label: str = "", k: int = 3) -> list:
    """
    Best/worst noise robustness examples per noise type.
    Recreation of evaluation_runner.plot_noisy_vs_clean_spectrogram: clean (black) vs
    noisy overlay with emb-sim in the title. One figure per noise type,
    2 rows (best/worst) × k columns.
    """
    rdf   = noise_results.get("results_df")
    clean = noise_results.get("clean_data")
    noisy = noise_results.get("noisy_data") or {}
    if rdf is None or rdf.empty or clean is None:
        return []

    suffix   = f"_{label}" if label else ""
    run_name = label or "checkpoint"
    figures  = []

    for noise_type in _NOISE_PLOT_TYPES:
        if noise_type not in rdf.columns or noise_type not in noisy:
            continue

        sorted_df = rdf.sort_values(noise_type, ascending=False)
        best  = sorted_df.head(k)
        worst = sorted_df.tail(k).iloc[::-1]

        fig, axes = plt.subplots(2, k, figsize=(5.5 * k, 7), squeeze=False)
        fig.suptitle(f"Noise Robustness — {noise_type} — Best/Worst Examples — {run_name}",
                     fontsize=12, fontweight="bold")

        for row_i, (status, group) in enumerate([("BEST", best), ("WORST", worst)]):
            for col_i, (_, r) in enumerate(group.iterrows()):
                ax = axes[row_i][col_i]
                idx = int(r["index"])
                ax.plot(clean[idx], label="Clean", color="black", linewidth=1.2)
                ax.plot(noisy[noise_type][idx], label=f"Noisy ({noise_type})", alpha=0.7, linewidth=1)
                ax.set_title(f"{status} | idx={idx} | Emb Sim: {r[noise_type]:.4f}", fontsize=9)
                ax.grid(alpha=0.3)
                if row_i == 0 and col_i == 0:
                    ax.legend(fontsize=8)

        plt.tight_layout()
        path = _save_fig(fig, os.path.join(output_dir, f"noisy_vs_clean_{noise_type}{suffix}.png"))
        figures.append((f"Noisy vs clean examples ({noise_type}) — {run_name}", path))

    return figures


def _plot_label_reg_scatter(lr_results: dict, output_dir: str, label: str = "") -> list:
    """
    True vs predicted parameter_0 scatter for input and embedding probes.
    Recreation of label_reg_evaluation._scatter_panel styling (s=3 alpha=0.2 scatter,
    red dashed diagonal, R²/pearson/MAE + distribution stats in the title).
    """
    from scipy.stats import pearsonr

    y = lr_results.get("labels")
    panels = [
        ("Input probe (raw 245-d signal)",   lr_results.get("y_pred_input"), "#90CAF9"),
        ("Embedding probe (768-d)",          lr_results.get("y_pred_emb"),   "#1565C0"),
    ]
    if y is None or all(p[1] is None for p in panels):
        return []

    suffix   = f"_{label}" if label else ""
    run_name = label or "checkpoint"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Label Regression — True vs Predicted (parameter_0) — {run_name}",
                 fontsize=12, fontweight="bold")

    for ax, (title, y_pred, color) in zip(axes, panels):
        if y_pred is None:
            ax.axis("off")
            continue
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_val = 1.0 - ss_res / (ss_tot + 1e-12)
        pr, _  = pearsonr(y, y_pred)
        mae    = float(np.mean(np.abs(y - y_pred)))
        ax.scatter(y, y_pred, s=3, alpha=0.2, color=color, rasterized=True)
        lo = min(y.min(), y_pred.min()) - 0.05
        hi = max(y.max(), y_pred.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
        ax.set_title(
            f"{title}\n"
            f"R²={r2_val:.3f}  r={float(pr):.3f}  MAE={mae:.3f}\n"
            f"true μ={y.mean():.2f} σ={y.std():.2f}  |  pred μ={y_pred.mean():.2f} σ={y_pred.std():.2f}",
            fontsize=8.5,
        )
        ax.set_xlabel("True", fontsize=8)
        ax.set_ylabel("Pred", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    path = _save_fig(fig, os.path.join(output_dir, f"label_reg_true_vs_pred{suffix}.png"))
    return [(f"Label regression true vs predicted — {run_name}", path)]


def _plot_ksimilar_examples(emb_results: dict, output_dir: str, label: str = "",
                            k: int = 5, n_examples: int = 3) -> list:
    """
    k-similar neighbor grids for best/worst match-score queries.
    Exact recreation of evaluation_runner.plot_embedding_vs_input_similarity_comparison:
    3×(k+1) grid — query in column 0, top-k embedding neighbors (row 1),
    top-k input neighbors (row 2), row 3 reserved.
    """
    match_df = emb_results.get("results_df")
    inputs   = emb_results.get("inputs")
    if match_df is None or match_df.empty or inputs is None:
        return []

    suffix   = f"_{label}" if label else ""
    run_name = label or "checkpoint"
    figures  = []

    sorted_df = match_df.sort_values("match_score", ascending=False)
    best_indices  = sorted_df.head(n_examples)["index"].values
    worst_indices = sorted_df.tail(n_examples)["index"].values

    for status, indices in [("best", best_indices), ("worst", worst_indices)]:
        for query_idx in indices:
            row = match_df[match_df["index"] == query_idx].iloc[0]
            query_stack = row["stack_idx"]

            topk_emb_idx   = row["embedding_neighbors"][:k]
            topk_input_idx = row["input_neighbors"][:k]
            emb_sims       = row["embedding_similarities"][:k]
            input_sims     = row["input_similarities"][:k]

            fig, axes = plt.subplots(3, k + 1, figsize=(3.5 * (k + 1), 9))
            fig.suptitle(
                f"Run: {run_name}\n"
                f"{status.upper()} | Query idx={query_idx} | stack={query_stack} | "
                f"Match Score: {row['match_score']:.1f} | "
                f"Emb matches: {len(row['embedding_stack_matches'])} | "
                f"Input matches: {len(row['input_stack_matches'])}",
                fontsize=11,
            )

            row_titles = ["Embedding neighbors", "Input neighbors", "Query signal"]

            for row_idx in range(3):
                axes[row_idx, 0].plot(inputs[query_idx])
                axes[row_idx, 0].set_title(f"QUERY\nidx={query_idx}")
                axes[row_idx, 0].set_xticks([])
                axes[row_idx, 0].set_ylabel(row_titles[row_idx])

            for j, idx in enumerate(topk_emb_idx):
                axes[0, j + 1].plot(inputs[idx])
                axes[0, j + 1].set_title(f"idx={idx}\nsim={emb_sims[j]:.3f}")
                axes[0, j + 1].set_xticks([])

            for j, idx in enumerate(topk_input_idx):
                axes[1, j + 1].plot(inputs[idx])
                axes[1, j + 1].set_title(f"idx={idx}\nsim={input_sims[j]:.3f}")
                axes[1, j + 1].set_xticks([])

            for j in range(k):
                axes[2, j + 1].axis("off")

            plt.tight_layout()
            fname = f"similarity_comparison_{status}_query{query_idx}{suffix}.png"
            path = _save_fig(fig, os.path.join(output_dir, fname))
            figures.append((f"k-similar neighbors — {status} query {query_idx} — {run_name}", path))

    return figures


def _plot_struct_sim_all_models(per_cp: dict, output_dir: str, centered: bool = False) -> list:
    """
    All-models structured similarity: Input Space | ckpt1 emb | ckpt2 emb | ...
    (+ second row of FE outputs when available).
    Exact port of compare_checkpoints._plot_similarity_rows styling: viridis
    heatmap vmin=0/vmax=1, (mean=, std=) in each title, 'Sample Index (N=…)' labels.
    centered=True subtracts the mean vector per representation before cosine
    (anisotropy correction) and switches the scale to vmin=-1/vmax=1.
    """
    import seaborn as sns
    from sklearn.metrics.pairwise import cosine_similarity as _cosim

    entries = []   # (run_name, inputs, embeddings, fe_outputs)
    for cp_label, cp_res in per_cp.items():
        ss = cp_res.get("structured_similarity") or {}
        if ss.get("embeddings") is not None:
            entries.append((cp_label, ss.get("inputs"), ss["embeddings"], ss.get("fe_outputs")))
    if not entries:
        return []

    inputs = entries[0][1]

    def _prep(m):
        if m is None:
            return None
        m = np.asarray(m, dtype=np.float64)
        if centered:
            m = m - m.mean(axis=0, keepdims=True)
        return m

    vmin, vmax = (-1.0, 1.0) if centered else (0.0, 1.0)
    n_models = 1 + len(entries)
    has_fe = any(e[3] is not None for e in entries)
    n_rows = 2 if has_fe else 1

    fig, axes = plt.subplots(n_rows, n_models, figsize=(5 * n_models, 5 * n_rows), squeeze=False)

    def _panel(ax, vectors, title, ylabel=None):
        if vectors is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.axis("off")
            return
        sim = _cosim(vectors)
        triu = np.triu_indices_from(sim, k=1)
        sns.heatmap(sim, ax=ax, cmap="viridis",
                    xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n(mean={sim[triu].mean():.3f}, std={sim[triu].std():.3f})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Sample Index (N={len(vectors)})", fontsize=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10)

    # Input column: never centered — raw inputs have no common-mode problem
    _panel(axes[0][0], np.asarray(inputs, dtype=np.float64) if inputs is not None else None,
           "Input Space", ylabel="Embeddings\nSample Index")
    for col, (name, _, emb, _fe) in enumerate(entries, start=1):
        _panel(axes[0][col], _prep(emb), name)

    if has_fe:
        _panel(axes[1][0], np.asarray(inputs, dtype=np.float64) if inputs is not None else None,
               "Input Space", ylabel="FE Output\nSample Index")
        for col, (name, _, _emb, fe) in enumerate(entries, start=1):
            _panel(axes[1][col], _prep(fe), f"{name}\n(FE Output)")

    tag = "centered cosine (mean vector removed)" if centered else "raw cosine"
    fig.suptitle(f"Input Space vs All Models — structured_similarity ({tag})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = "struct_sim_all_models_centered.png" if centered else "struct_sim_all_models.png"
    path = _save_fig(fig, os.path.join(output_dir, fname))
    return [(f"Structured similarity — all models ({tag})", path, "")]


def _plot_checkpoint_comparison(results: dict, output_dir: str) -> list:
    """
    Returns a list of (caption, path, checkpoint_label) triples.
    checkpoint_label="" for summary figures, checkpoint name for per-checkpoint figures.
    Callers use the label to group figures under per-checkpoint sections in the report.
    """
    figures = []   # (caption, path, cp_label)
    cdf     = results.get("comparison_df")
    per_cp  = results.get("per_checkpoint", {})

    # ── Summary: scalar metrics across all checkpoints ────────────────────────
    if cdf is not None and not cdf.empty:
        # Only numeric columns are plottable (excludes 'checkpoint', 'date', errors)
        metric_cols = cdf.select_dtypes(include="number").columns.tolist()
        if metric_cols:
            n = len(metric_cols)
            n_cols = min(n, 4)
            n_rows = (n + n_cols - 1) // n_cols
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
            )
            for i, col in enumerate(metric_cols):
                ax = axes[i // n_cols][i % n_cols]
                ax.plot(range(len(cdf)), cdf[col], marker="o", color=_ACCENT)
                ax.set_xticks(range(len(cdf)))
                ax.set_xticklabels(cdf["checkpoint"], rotation=35, ha="right", fontsize=8)
                ax.set_title(col.replace("_", " ").title(), fontsize=9)
                ax.grid(alpha=0.25)
            for j in range(len(metric_cols), n_rows * n_cols):
                axes[j // n_cols][j % n_cols].set_visible(False)
            fig.suptitle("Checkpoint Comparison — Scalar Metrics", fontsize=12, y=1.02)
            fig.tight_layout()
            path = _save_fig(fig, os.path.join(output_dir, "checkpoint_comparison.png"))
            figures.append(("Scalar metrics across checkpoints", path, ""))

        # ── Summary: per-method comparison figures ────────────────────────────
        figures += _plot_noise_robustness_comparison(cdf, output_dir)
        figures += _plot_label_regression_comparison(cdf, output_dir)

    # ── Summary: all-models structured similarity (raw + centered) ───────────
    if per_cp:
        figures += _plot_struct_sim_all_models(per_cp, output_dir, centered=False)
        figures += _plot_struct_sim_all_models(per_cp, output_dir, centered=True)

    # ── Per-checkpoint figures ────────────────────────────────────────────────
    if per_cp:
        for cp_label, cp_res in per_cp.items():
            ss = cp_res.get("structured_similarity", {})
            if ss:
                ss_figs = _plot_structured_similarity_maps(ss, output_dir, label=cp_label)
                figures += [(cap, path, cp_label) for cap, path in ss_figs]

            clust = cp_res.get("clustering", {})
            if clust:
                cl_figs = _plot_clustering_scatter(clust, output_dir, label=cp_label)
                figures += [(cap, path, cp_label) for cap, path in cl_figs]

            noise = cp_res.get("noise_robustness", {})
            if noise:
                nz_figs = _plot_noise_example_grid(noise, output_dir, label=cp_label)
                nz_figs += _plot_noise_examples(noise, output_dir, label=cp_label)
                figures += [(cap, path, cp_label) for cap, path in nz_figs]

            lr = cp_res.get("label_regression", {})
            if lr:
                lr_figs = _plot_label_reg_scatter(lr, output_dir, label=cp_label)
                figures += [(cap, path, cp_label) for cap, path in lr_figs]

            emb = cp_res.get("embedding_similarity", {})
            if emb:
                ks_figs = _plot_ksimilar_examples(emb, output_dir, label=cp_label)
                figures += [(cap, path, cp_label) for cap, path in ks_figs]

    return figures


# ── HTML helpers ───────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
       background: #f0f2f5; color: #1a1a2e; line-height: 1.5; }
header { background: __DARK__; color: white; padding: 2rem 3rem; }
header h1 { font-size: 1.9rem; font-weight: 700; letter-spacing: -.5px; }
header .meta { opacity: .75; margin-top: .4rem; font-size: .88rem; }
main { max-width: 1150px; margin: 2rem auto; padding: 0 1.5rem; }
section { background: white; border-radius: 10px; padding: 2rem 2.5rem;
          margin-bottom: 1.8rem; box-shadow: 0 1px 6px rgba(0,0,0,.07); }
h2 { font-size: 1.15rem; color: __DARK__; border-bottom: 2.5px solid __ACCENT__;
     padding-bottom: .5rem; margin-bottom: 1.2rem; text-transform: uppercase;
     letter-spacing: .5px; }
table { width: 100%; border-collapse: collapse; font-size: .88rem; }
thead th { background: __ACCENT__; color: white; padding: .55rem 1rem;
           text-align: left; white-space: nowrap; }
tbody td { padding: .45rem 1rem; border-bottom: 1px solid #e8eaf0; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:nth-child(even) td { background: #f7f8fa; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.good { color: #1a7a3c; font-weight: 600; }
td.warn { color: #856404; font-weight: 600; }
td.bad  { color: #c0392b; font-weight: 600; }
.kv-table td:first-child { color: #555; width: 38%; }
.kv-table td:last-child { font-family: monospace; font-size: .83rem; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
               gap: 1rem; margin-bottom: 1.5rem; }
.metric-card { background: #f7f8fa; border-left: 4px solid __ACCENT__;
               border-radius: 6px; padding: .8rem 1.1rem; }
.metric-card .label { font-size: .78rem; color: #666; text-transform: uppercase;
                      letter-spacing: .4px; }
.metric-card .value { font-size: 1.4rem; font-weight: 700; color: __DARK__; margin-top: .2rem; }
figure { margin: 1.5rem 0; text-align: center; }
figure img { max-width: 100%; border-radius: 6px; border: 1px solid #e0e3ea; }
figcaption { margin-top: .5rem; font-size: .82rem; color: #777; font-style: italic; }
.cp-section { border-top: 2px dashed #d0d4e0; margin-top: 2rem; padding-top: 1.2rem; }
.cp-heading { font-size: 1rem; color: __ACCENT__; font-family: monospace;
              margin-bottom: 1rem; font-weight: 600; }
.cp-date { font-size: .8rem; color: #888; font-weight: 400; font-family: sans-serif;
           margin-left: .8rem; }
.panel-legend { font-size: .82rem; color: #555; background: #f7f8fa;
                border-left: 3px solid __ACCENT__; padding: .6rem 1rem;
                margin: 1.2rem 0; border-radius: 4px; }
""".replace("__DARK__", _DARK).replace("__ACCENT__", _ACCENT)


def _html_table_from_df(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Render a DataFrame as an HTML table with right-aligned numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    rows = ["<table>", "<thead><tr>"]
    for col in df.columns:
        rows.append(f"<th>{col}</th>")
    rows.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        rows.append("<tr>")
        for col in df.columns:
            val = row[col]
            if col in numeric_cols:
                formatted = f"{val:{float_fmt}}"
                # Colour-code noise/match columns
                cls = "num"
                if "match_score" in col:
                    cls = "num good" if val >= 50 else "num bad"
                elif "noise_" in col or "match_rate" in col:
                    cls = "num good" if val >= 0.9 else ("num warn" if val >= 0.7 else "num bad")
                rows.append(f'<td class="{cls}">{formatted}</td>')
            else:
                rows.append(f"<td>{val}</td>")
        rows.append("</tr>")

    rows.append("</tbody></table>")
    return "\n".join(rows)


def _html_kv(mapping: dict) -> str:
    rows = ['<table class="kv-table">']
    for k, v in mapping.items():
        rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def _html_metric_cards(metrics: dict) -> str:
    cards = ['<div class="metric-grid">']
    for label, val in metrics.items():
        val_str = f"{val:.3f}" if isinstance(val, float) else str(val)
        cards.append(
            f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{val_str}</div>'
            f"</div>"
        )
    cards.append("</div>")
    return "\n".join(cards)


def _html_figure(caption: str, img_path: str) -> str:
    src = _fig_to_b64(img_path)
    return (
        f"<figure>"
        f'<img src="{src}" alt="{caption}">'
        f"<figcaption>{caption}</figcaption>"
        f"</figure>"
    )


# ── HTML section builders ──────────────────────────────────────────────────────

def _html_section_config(config) -> str:
    if config is None:
        return ""
    kv = {k: str(v) for k, v in vars(config).items()}
    return f"<section>\n<h2>Configuration</h2>\n{_html_kv(kv)}\n</section>"


def _html_section_embedding(results: dict, figures: list) -> str:
    r = results.get("embedding_similarity", {})
    cards = _html_metric_cards({
        "Embedding match rate": r.get("embedding_stack_match_rate", "N/A"),
        "Input baseline rate": r.get("input_stack_match_rate", "N/A"),
        "Match score (0–100)": r.get("match_score_avg", "N/A"),
    })
    fig_html = "\n".join(_html_figure(cap, path) for cap, path in figures)
    return f"<section>\n<h2>Embedding Similarity</h2>\n{cards}\n{fig_html}\n</section>"


def _html_section_signal_reconstruction(results: dict, figures: list) -> str:
    r = results.get("signal_reconstruction", {})
    if r.get("skipped"):
        body = f"<p><em>Skipped — {r.get('error', 'no reconstruction checkpoints given')}.</em></p>"
    else:
        cards = {}
        for key, label in (("fe", "FE"), ("proj", "Projection"), ("tr", "Transformer")):
            if f"recon_{key}_mse_mean" in r:
                cards[f"{label} recon mean MSE"] = f"{r[f'recon_{key}_mse_mean']:.3e}"
        cards["normalize"] = str(r.get("normalize"))
        body = _html_metric_cards(cards)
        body += "\n" + "\n".join(_html_figure(cap, path) for cap, path in figures)
    return f"<section>\n<h2>Signal Reconstruction</h2>\n{body}\n</section>"


def _html_section_generic(title: str, cards: dict, figures: list) -> str:
    """Section with optional metric cards + figures, for standalone evals."""
    body = _html_metric_cards(cards) if cards else ""
    body += "\n" + "\n".join(_html_figure(cap, path) for cap, path in figures)
    return f"<section>\n<h2>{title}</h2>\n{body}\n</section>"


def _html_section_noise(results: dict, figures: list) -> str:
    r = results.get("noise_robustness", {})
    summary = r.get("summary", {})
    cards = _html_metric_cards(summary) if summary else ""
    fig_html = "\n".join(_html_figure(cap, path) for cap, path in figures)
    return f"<section>\n<h2>Noise Robustness</h2>\n{cards}\n{fig_html}\n</section>"


def _html_section_comparison(results: dict, figures: list) -> str:
    """
    figures is a list of (caption, path, cp_label).
    Summary figures (cp_label="") come first, then one subsection per checkpoint.
    """
    r   = results.get("checkpoint_comparison", {})
    cdf = r.get("comparison_df")

    table_html   = _html_table_from_df(cdf) if cdf is not None else ""
    summary_figs = [f for f in figures if f[2] == ""]
    summary_html = "\n".join(_html_figure(cap, path) for cap, path, _ in summary_figs)

    # Per-checkpoint subsections
    cp_labels = []
    seen = set()
    for _, _, lbl in figures:
        if lbl and lbl not in seen:
            cp_labels.append(lbl)
            seen.add(lbl)

    per_cp = r.get("per_checkpoint", {})
    panel_legend = (
        '<p class="panel-legend">'
        '<strong>Panel composition (100 samples, seed=42):</strong> '
        '0–29 single_channel_all (3 stacks × 10) &nbsp;|&nbsp; '
        '30–59 multi_channel (3 comps × 10) &nbsp;|&nbsp; '
        '60–79 sampled_data (2 comps × 10) &nbsp;|&nbsp; '
        '80–99 labeled_data (2 comps × 10)'
        '</p>'
    )

    cp_sections = []
    for lbl in cp_labels:
        cp_figs = [f for f in figures if f[2] == lbl]
        fig_html  = "\n".join(_html_figure(cap, path) for cap, path, _ in cp_figs)
        date_str  = per_cp.get(lbl, {}).get("_date", "")
        date_tag  = f'<span class="cp-date">{date_str}</span>' if date_str and date_str != "N/A" else ""
        cp_sections.append(
            f'<div class="cp-section">'
            f'<h3 class="cp-heading">{lbl} {date_tag}</h3>'
            f'{fig_html}'
            f'</div>'
        )

    cp_html = "\n".join(cp_sections)
    return (
        f"<section>\n<h2>Checkpoint Comparison</h2>\n"
        f"{table_html}\n{summary_html}\n{panel_legend}\n{cp_html}\n</section>"
    )


def _build_html(results: dict, config, ts: str, sections_html: list) -> str:
    config_meta = ""
    if config is not None:
        ckpt = getattr(config, "checkpoint_path", "")
        data = getattr(config, "data_source", "")
        config_meta = f" &nbsp;|&nbsp; data: <code>{os.path.basename(data)}</code> &nbsp;|&nbsp; ckpt: <code>{os.path.basename(str(ckpt))}</code>"

    body = "\n".join(s for s in sections_html if s)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpectralFM Eval Report — {ts}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>SpectralFM Evaluation Report</h1>
  <div class="meta">{ts}{config_meta}</div>
</header>
<main>
{body}
</main>
</body>
</html>"""


# ── Main entry point ───────────────────────────────────────────────────────────

def _write_run_info(run_dir: str, ts: str, results: dict, config) -> str:
    """Write a human-readable run_info.md describing what was analyzed."""
    lines = [
        "# SpectralFM Eval Run Info",
        "",
        f"**Timestamp:** {ts}",
        "",
        "## What was analyzed",
        "",
    ]

    if config is not None:
        data_src = getattr(config, "data_source", "N/A")
        ckpt_mode = getattr(config, "checkpoint_mode", "N/A")
        ckpt_path = getattr(config, "checkpoint_path", "N/A")
        nova_dir  = getattr(config, "nova_data_dir", None)
        evals     = getattr(config, "evals", [])
        device    = getattr(config, "device", "N/A")
        n_holdout = getattr(config, "n_holdout_stacks", "N/A")

        lines += [
            f"- **Data source:** `{data_src}`",
            f"- **Holdout stacks:** {n_holdout}",
            f"- **Checkpoint mode:** {ckpt_mode}",
            f"- **Checkpoint path:** `{ckpt_path}`",
            f"- **Evals run:** {', '.join(evals)}",
            f"- **Device:** {device}",
        ]
        if nova_dir:
            lines.append(f"- **Structured similarity panel:** `{nova_dir}`")
        labeled_dir = getattr(config, "labeled_data_dir", None)
        if labeled_dir:
            lines.append(f"- **Label regression data:** `{labeled_dir}`")

    # Checkpoints found in results
    cp_results = results.get("checkpoint_comparison", {})
    cdf = cp_results.get("comparison_df")
    if cdf is not None and not cdf.empty and "checkpoint" in cdf.columns:
        lines += ["", "## Checkpoints analyzed", ""]
        for name in cdf["checkpoint"].tolist():
            lines.append(f"- `{name}`")

    lines += [
        "", "## Files in this directory", "",
        "| File | Description |",
        "|------|-------------|",
        "| `eval_report.html` | Self-contained HTML report with all figures embedded |",
        "| `eval_report.md` | Markdown version of the report |",
        "| `run_info.md` | This file |",
        "| `*.png` | Individual figure PNGs |",
        "| `*.csv` | Exported metric tables |",
    ]

    path = os.path.join(run_dir, "run_info.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def generate_report(results: dict, output_dir: str, config=None) -> tuple[str, str]:
    """
    Generate markdown + self-contained HTML report from eval results.
    Each run gets its own timestamped subdirectory inside output_dir.
    Returns (md_path, html_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # All files for this run go into a dedicated subdirectory
    run_dir = os.path.join(output_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    # ── Collect figures per eval ──────────────────────────────────────────────
    figures_by_eval: dict[str, list] = {}

    if "embedding_similarity" in results:
        figures_by_eval["embedding_similarity"] = (
            _plot_embedding_similarity(results["embedding_similarity"], run_dir)
            + _plot_ksimilar_examples(results["embedding_similarity"], run_dir)
        )
    if "signal_reconstruction" in results:
        figures_by_eval["signal_reconstruction"] = _plot_signal_reconstruction(
            results["signal_reconstruction"], run_dir
        )
    if "noise_robustness" in results:
        figures_by_eval["noise_robustness"] = (
            _plot_noise_robustness(results["noise_robustness"], run_dir)
            + _plot_noise_example_grid(results["noise_robustness"], run_dir)
            + _plot_noise_examples(results["noise_robustness"], run_dir)
        )
    if "clustering" in results:
        figures_by_eval["clustering"] = _plot_clustering_scatter(
            results["clustering"], run_dir
        )
    if "label_regression" in results:
        figures_by_eval["label_regression"] = _plot_label_reg_scatter(
            results["label_regression"], run_dir
        )
    if "structured_similarity" in results:
        figures_by_eval["structured_similarity"] = _plot_structured_similarity_maps(
            results["structured_similarity"], run_dir
        )
    if "checkpoint_comparison" in results:
        figures_by_eval["checkpoint_comparison"] = _plot_checkpoint_comparison(
            results["checkpoint_comparison"], run_dir
        )

    # checkpoint_comparison returns (cap, path, cp_label) triples; others return (cap, path) pairs
    all_figures = []
    for eval_name, figs in figures_by_eval.items():
        for fig in figs:
            all_figures.append(fig[:2])  # just (caption, path) for counting

    # ── Run info ──────────────────────────────────────────────────────────────
    _write_run_info(run_dir, ts, results, config)

    # ── Markdown ──────────────────────────────────────────────────────────────
    md_path = os.path.join(run_dir, "eval_report.md")
    lines = [
        "# SpectralFM Evaluation Report",
        "",
        f"**Date:** {ts}",
    ]

    if config is not None:
        lines += ["", "## Configuration", "", "| Parameter | Value |", "|-----------|-------|"]
        for k, v in vars(config).items():
            lines.append(f"| `{k}` | `{v}` |")

    if "embedding_similarity" in results:
        r = results["embedding_similarity"]
        lines += [
            "", "## Embedding Similarity", "",
            "| Metric | Value |", "|--------|-------|",
            f"| Embedding stack match rate | {r.get('embedding_stack_match_rate', 'N/A'):.3f} |",
            f"| Input stack match rate     | {r.get('input_stack_match_rate', 'N/A'):.3f} |",
            f"| Average match score (0–100)| {r.get('match_score_avg', 'N/A'):.1f} |",
        ]
        for cap, path in figures_by_eval.get("embedding_similarity", []):
            lines += ["", f"![{cap}]({os.path.basename(path)})"]

    if "signal_reconstruction" in results:
        r = results["signal_reconstruction"]
        lines += ["", "## Signal Reconstruction"]
        if r.get("skipped"):
            lines.append(f"*Skipped — {r.get('error', 'no reconstruction checkpoints given')}.*")
        else:
            lines += ["", "| Metric | Value |", "|--------|-------|"]
            for key, label in (("fe", "FE"), ("proj", "Projection"), ("tr", "Transformer")):
                if f"recon_{key}_mse_mean" in r:
                    lines.append(f"| {label} recon mean MSE | {r[f'recon_{key}_mse_mean']:.6e} |")
            lines.append(f"| normalize | {r.get('normalize')} |")
            for cap, path in figures_by_eval.get("signal_reconstruction", []):
                lines += ["", f"![{cap}]({os.path.basename(path)})"]

    if "noise_robustness" in results:
        r = results["noise_robustness"]
        lines += ["", "## Noise Robustness", ""]
        summary = r.get("summary", {})
        if summary:
            lines += ["| Noise Type | Mean Cosine Sim |", "|------------|-----------------|"]
            for k, v in summary.items():
                lines.append(f"| {k} | {v:.4f} |")
        for cap, path in figures_by_eval.get("noise_robustness", []):
            lines += ["", f"![{cap}]({os.path.basename(path)})"]

    if "clustering" in results:
        r = results["clustering"]
        lines += ["", "## Clustering", ""]
        if "comp_cluster_error" in r:
            lines.append(f"*Skipped — {r['comp_cluster_error']}*")
        else:
            lines += ["| Metric | Value |", "|--------|-------|"]
            for key in ("comp_cluster_ari", "comp_cluster_nmi", "comp_cluster_vmeasure",
                        "comp_cluster_silhouette", "comp_cluster_knn_precision",
                        "comp_cluster_retrieval_map"):
                if key in r:
                    lines.append(f"| {key.replace('comp_cluster_', '')} | {r[key]:.4f} |")
        for cap, path in figures_by_eval.get("clustering", []):
            lines += ["", f"![{cap}]({os.path.basename(path)})"]

    if "label_regression" in results:
        r = results["label_regression"]
        lines += ["", "## Label Regression (parameter_0)", ""]
        lines += ["| Metric | Value |", "|--------|-------|"]
        for key in ("label_reg_input_r2", "label_reg_emb_r2", "label_reg_improvement_r2"):
            if key in r:
                lines.append(f"| {key} | {r[key]:.4f} |")
        for cap, path in figures_by_eval.get("label_regression", []):
            lines += ["", f"![{cap}]({os.path.basename(path)})"]

    if "structured_similarity" in results:
        lines += ["", "## Structured Similarity (canonical 100-sample panel)"]
        for cap, path in figures_by_eval.get("structured_similarity", []):
            lines += ["", f"![{cap}]({os.path.basename(path)})"]

    if "checkpoint_comparison" in results:
        r = results["checkpoint_comparison"]
        lines += ["", "## Checkpoint Comparison", ""]
        cdf = r.get("comparison_df")
        if cdf is not None:
            try:
                lines.append(cdf.to_markdown(index=False))
            except Exception:
                lines.append(cdf.to_string(index=False))
        for fig in figures_by_eval.get("checkpoint_comparison", []):
            cap, path = fig[0], fig[1]
            lines += ["", f"![{cap}]({os.path.basename(path)})"]

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    # ── HTML ──────────────────────────────────────────────────────────────────
    html_path = os.path.join(run_dir, "eval_report.html")
    sections = [_html_section_config(config)]

    if "embedding_similarity" in results:
        sections.append(_html_section_embedding(
            results, figures_by_eval.get("embedding_similarity", [])
        ))
    if "signal_reconstruction" in results:
        sections.append(_html_section_signal_reconstruction(
            results, figures_by_eval.get("signal_reconstruction", [])
        ))
    if "noise_robustness" in results:
        sections.append(_html_section_noise(
            results, figures_by_eval.get("noise_robustness", [])
        ))
    if "clustering" in results:
        r = results["clustering"]
        cards = {k.replace("comp_cluster_", ""): f"{v:.4f}"
                 for k, v in r.items()
                 if k.startswith("comp_cluster_") and isinstance(v, (int, float))}
        sections.append(_html_section_generic(
            "Clustering", cards, figures_by_eval.get("clustering", [])
        ))
    if "label_regression" in results:
        r = results["label_regression"]
        cards = {k: f"{v:.4f}" for k, v in r.items()
                 if k.startswith("label_reg_") and isinstance(v, (int, float))}
        sections.append(_html_section_generic(
            "Label Regression (parameter_0)", cards,
            figures_by_eval.get("label_regression", [])
        ))
    if "structured_similarity" in results:
        sections.append(_html_section_generic(
            "Structured Similarity (canonical panel)", {},
            figures_by_eval.get("structured_similarity", [])
        ))
    if "checkpoint_comparison" in results:
        sections.append(_html_section_comparison(
            results, figures_by_eval.get("checkpoint_comparison", [])
        ))

    with open(html_path, "w") as f:
        f.write(_build_html(results, config, ts, sections))

    # ── CSV exports ───────────────────────────────────────────────────────────
    # Nested-DataFrame export names mirror the old evaluation_runner conventions
    # (match_df_<run>.csv for the per-query same-stack results, noise_df_<run>.csv, ...)
    _CSV_NAMES = {
        "embedding_similarity": "match_df",
        "noise_robustness":     "noise_df",
        "signal_reconstruction": "recon_df",
        "label_regression":     "label_reg_df",
        "clustering":           "clustering_df",
    }

    for eval_name, r in results.items():
        if not isinstance(r, dict):
            continue
        # Top-level DataFrames (single-checkpoint evals + comparison_df)
        for key, val in r.items():
            if isinstance(val, pd.DataFrame):
                base = _CSV_NAMES.get(eval_name, eval_name)
                name = base if key == "results_df" else f"{eval_name}_{key}"
                val.to_csv(os.path.join(run_dir, f"{name}.csv"), index=False)
        # Per-checkpoint DataFrames inside checkpoint_comparison
        for cp_label, cp_res in (r.get("per_checkpoint") or {}).items():
            if not isinstance(cp_res, dict):
                continue
            for sub_name, sub_res in cp_res.items():
                if not isinstance(sub_res, dict):
                    continue
                rdf = sub_res.get("results_df")
                if isinstance(rdf, pd.DataFrame):
                    base = _CSV_NAMES.get(sub_name, sub_name)
                    rdf.to_csv(os.path.join(run_dir, f"{base}_{cp_label}.csv"), index=False)

    total_figs = len(all_figures)
    print(f"[Report] Run dir  : {run_dir}")
    print(f"[Report] Markdown : {md_path}")
    print(f"[Report] HTML     : {html_path}")
    print(f"[Report] Figures  : {total_figs}")
    return md_path, html_path
