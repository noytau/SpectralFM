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

    # Unstructured cosine similarity maps removed per E4 decision;
    # the distribution histograms remain.
    sim_dists = results.get("similarity_distributions", {})
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


def _recon_fig_caption(path: str) -> str:
    """Short caption for a figure, from the shared recon_plots registry."""
    from . import recon_plots
    d = recon_plots.doc_for_figure(path)
    return d["caption"] if d else os.path.basename(path)


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

    # Head colours, line styles and the e-ink switch all live in recon_plots, so this
    # figure follows them too rather than keeping a private palette that would stay in
    # colour when everything else went greyscale.
    from . import recon_plots
    eink = recon_plots.is_eink()

    pathways = [
        (label, panel[f"pred_{key}"], recon_plots._HEAD_COLOR[key],
         recon_plots._HEAD_LS[key], f"{key}_mse")
        for key, label in (("fe", "FE recon"), ("proj", "Projection recon"),
                           ("tr", "Transformer recon"))
        if panel.get(f"pred_{key}") is not None
    ]
    if target is None or not pathways:
        return figures

    # Eighteen panels across a 6-inch page is nothing but ink. Fewer, taller rows on a
    # page; the full six stay in the HTML, where there is room to scroll.
    rows = list(range(len(indices)))
    if eink and len(rows) > 3:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, 3).astype(int)]

    n = len(rows)
    T = target.shape[1]
    cell_w, cell_h = (2.3, 1.75) if eink else (6.5, 1.9)
    fs_tick, fs_label, fs_title = (8, 10, 8) if eink else (6, 9, 7)
    fig, axes = plt.subplots(n, len(pathways),
                             figsize=(cell_w * len(pathways), cell_h * n),
                             squeeze=False)
    for r, ri in enumerate(rows):
        tgt = target[ri]
        pad = 0.1 * max(tgt.max() - tgt.min(), 0.1)
        ylo, yhi = tgt.min() - pad, tgt.max() + pad
        for c, (pw_label, pred, color, ls, mse_col) in enumerate(pathways):
            ax = axes[r, c]
            ax.plot(tgt, color="black", lw=1.6, label="target", alpha=0.9)
            ax.plot(pred[ri], color=color, lw=1.3 if eink else 1.1, ls=ls,
                    label=pw_label, alpha=0.9)
            ax.set_xlim(0, T - 1)
            ax.set_ylim(ylo, yhi)
            ax.tick_params(labelsize=fs_tick, labelleft=(c == 0))
            if c == 0:
                ax.set_ylabel(f"idx {indices[ri]}", fontsize=fs_label, rotation=0,
                              ha="right", labelpad=22)
            mse_val = ""
            if rdf is not None and mse_col in rdf.columns:
                mse_val = f"  MSE = {rdf[rdf['index'] == indices[ri]][mse_col].iloc[0]:.2e}"
            title = f"{names[ri][-40:]}{mse_val}" if c == 0 else f"{pw_label}{mse_val}"
            ax.set_title(title, fontsize=fs_title, loc="left")
            if r == 0:
                ax.legend(fontsize=fs_title, loc="upper right")

    mean_bits = [
        f"{label} mean MSE = {results[f'recon_{key}_mse_mean']:.3e}"
        + (f" (MAE = {results[f'recon_{key}_mae_mean']:.3e})"
           if f"recon_{key}_mae_mean" in results else "")
        for key, label in (("fe", "FE recon"), ("proj", "Projection recon"),
                           ("tr", "Transformer recon"))
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
    figures.append((_recon_fig_caption(path), path))

    # Per-sample MSE bars (log scale), one bar group per pathway present
    bar_cols = [(label, color, mse_col.split("_")[0], mse_col)
                for label, _, color, _, mse_col in pathways
                if rdf is not None and mse_col in rdf.columns]
    if bar_cols:
        sub = rdf[rdf["index"].isin([indices[i] for i in rows])]
        x = np.arange(len(sub))
        w = 0.8 / len(bar_cols)
        fig, ax = plt.subplots(figsize=(6.4, 4.2) if eink else (9, 4))
        for j, (label, color, head, mse_col) in enumerate(bar_cols):
            offset = (j - (len(bar_cols) - 1) / 2) * w
            ax.bar(x + offset, sub[mse_col], w, color=color,
                   hatch=recon_plots._HEAD_HATCH.get(head, ""),
                   edgecolor="white", linewidth=0.6,
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
        figures.append((_recon_fig_caption(path), path))

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


def _struct_sim_panel(ax, sim, title: str, vmin: float = 0.0, vmax: float = 1.0,
                      ylabel: str = None) -> None:
    """
    Shared panel styling for ALL structured-similarity figures (single-model and
    all-models): viridis heatmap, (mean=, std=) appended to the title, and
    'Sample Index (N=…)' axis labels — matching compare_checkpoints._plot_similarity_rows.
    """
    import seaborn as sns

    if sim is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.axis("off")
        return
    sim = np.asarray(sim, dtype=np.float64)
    triu = np.triu_indices_from(sim, k=1)
    sns.heatmap(sim, ax=ax, cmap="viridis",
                xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\n(mean={sim[triu].mean():.3f}, std={sim[triu].std():.3f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Sample Index (N={len(sim)})", fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def _plot_structured_similarity_maps(results: dict, output_dir: str, label: str = "") -> list:
    """
    Single-checkpoint structured similarity — one row across the 4 pipeline stages:
      Input Space | FE Output | Projection | Embeddings
    Styled identically to the all-models comparison figure (same panel helper).
    """
    panels = [
        ("sim_matrix_inp",  "Input Space (Raw Signals)"),
        ("sim_matrix_fe",   "FE Output (512d)"),
        ("sim_matrix_proj", "Projection (768d)"),
        ("sim_matrix_emb",  "Embeddings (768d)"),
    ]

    suffix   = f"_{label}" if label else ""
    run_name = label or "checkpoint"

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5), squeeze=False)
    for ax, (key, title) in zip(axes[0], panels):
        _struct_sim_panel(ax, results.get(key), title,
                          ylabel="Sample Index" if key == "sim_matrix_inp" else None)

    fig.suptitle(f"Structured Similarity — {run_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = _save_fig(fig, os.path.join(output_dir, f"struct_sim{suffix}.png"))
    return [(f"Structured similarity (4 pipeline stages) — {run_name}", path)]


def _plot_label_regression_comparison(cdf: pd.DataFrame, output_dir: str) -> list:
    """
    Label regression bars across checkpoints, one COLUMN per multi-channel config
    (1/2/3 components). Per config: top panel input vs embedding R² grouped bars
    (eval_label_regression._plot_label_regression styling), bottom panel ΔR² bars.
    """
    configs = [(sfx, cfg_label) for sfx, cfg_label in _LR_CONFIGS
               if f"label_reg_input_r2{sfx}" in cdf.columns
               and f"label_reg_emb_r2{sfx}" in cdf.columns]
    if not configs:
        return []

    labels = cdf["checkpoint"].tolist()
    n = len(labels)
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(2, len(configs),
                             figsize=(max(5, n * 2.5) * len(configs), 9), squeeze=False)
    fig.suptitle("Label Regression — Ridge Probe (parameter_0) — 1/2/3-component configs",
                 fontsize=13, fontweight="bold")

    for col, (sfx, cfg_label) in enumerate(configs):
        r2_in  = cdf[f"label_reg_input_r2{sfx}"].fillna(0.0).tolist()
        r2_emb = cdf[f"label_reg_emb_r2{sfx}"].fillna(0.0).tolist()
        delta  = cdf[f"label_reg_improvement_r2{sfx}"].fillna(0.0).tolist() \
            if f"label_reg_improvement_r2{sfx}" in cdf.columns \
            else [e - i for e, i in zip(r2_emb, r2_in)]

        # Top: side-by-side R²
        ax = axes[0][col]
        b1 = ax.bar(x - w / 2, r2_in, w, color="#90CAF9", edgecolor="white", label="Input R²")
        b2 = ax.bar(x + w / 2, r2_emb, w, color="#1565C0", edgecolor="white", label="Embedding R²")
        for bar, v in zip(b1, r2_in):
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#555")
        for bar, v in zip(b2, r2_emb):
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                    color="#1565C0", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([_shorten_label(l) for l in labels],
                           rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("R²")
        ax.set_title(cfg_label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

        # Bottom: ΔR²
        ax2 = axes[1][col]
        colors = ["#2E7D32" if d > 0 else "#C62828" for d in delta]
        bars = ax2.bar(x, delta, 0.5, color=colors, edgecolor="white")
        for bar, v in zip(bars, delta):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.001 if v >= 0 else -0.003),
                     f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                     fontsize=8, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels([_shorten_label(l) for l in labels],
                            rotation=25, ha="right", fontsize=8)
        ax2.set_ylabel("ΔR² (embedding − input)")
        ax2.set_title("Improvement over Raw Input", fontsize=10)
        ax2.axhline(0, color="black", linewidth=1)
        ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = _save_fig(fig, os.path.join(output_dir, "label_regression_comparison.png"))
    return [("Label regression — ridge probe (parameter_0), 1/2/3-comp", path, "")]


def _shorten_label(label: str, max_len: int = 28) -> str:
    """Shorten a long checkpoint label for axis ticks (keep head + tail)."""
    if len(label) <= max_len:
        return label
    half = (max_len - 1) // 2
    return label[:half] + "…" + label[-half:]


def _plot_noise_robustness_comparison(cdf: pd.DataFrame, output_dir: str,
                                      alias: str = "") -> list:
    """
    Noise robustness grouped bars across checkpoints.
    Recreation of evaluation_runner.plot_noise_robustness_comparison panel 1
    (embedding similarity per noise type). Legend sits outside the axes and
    checkpoint labels are shortened so nothing overlaps the title.
    alias filters the E4 dataset-suffixed columns (noise_<type>_<alias>).
    """
    noise_cols = [c for c in cdf.columns if c.startswith("noise_")]
    if alias:
        noise_cols = [c for c in noise_cols if c.endswith(f"_{alias}")]
    if not noise_cols:
        return []

    run_labels = [_shorten_label(l) for l in cdf["checkpoint"].tolist()]
    x = np.arange(len(run_labels))
    width = 0.8 / len(noise_cols)

    fig, ax = plt.subplots(figsize=(max(8, len(run_labels) * 2.5 + 3), 4.5))
    for i, col in enumerate(noise_cols):
        lbl = col.replace("noise_", "")
        if alias and lbl.endswith(f"_{alias}"):
            lbl = lbl[: -len(alias) - 1]
        ax.bar(x + i * width, cdf[col].fillna(0.0), width, label=lbl)

    ax.set_ylabel("Embedding Similarity (higher = more robust)", fontsize=9)
    ax.set_title(f"Noise Robustness: Embedding Similarity"
                 + (f" — {alias}" if alias else ""), fontsize=11)
    ax.set_xticks(x + width * (len(noise_cols) - 1) / 2)
    ax.set_xticklabels(run_labels, rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    sfx = f"_{alias}" if alias else ""
    path = _save_fig(fig, os.path.join(output_dir, f"noise_robustness_comparison{sfx}.png"))
    return [(f"Noise robustness — embedding similarity per noise type"
             + (f" ({alias})" if alias else ""), path, "")]


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


# Multi-channel label regression configs: (suffix, display label)
_LR_CONFIGS = [("", "1-comp (C0) — raw 245 / emb 768"),
               ("_2c", "2-comp (C0,C1) — raw 490 / emb 1536"),
               ("_3c", "3-comp (C0,C1,C2) — raw 735 / emb 2304")]


def _plot_label_reg_scatter(lr_results: dict, output_dir: str, label: str = "") -> list:
    """
    True vs predicted parameter_0 scatter for input and embedding probes —
    one row per multi-channel config (1/2/3 components), two columns (input | emb).
    label_reg_evaluation._scatter_panel styling (s=3 alpha=0.2 scatter,
    red dashed diagonal, R²/pearson/MAE + distribution stats in the title).
    """
    from scipy.stats import pearsonr

    y = lr_results.get("labels")
    configs = [(sfx, cfg_label) for sfx, cfg_label in _LR_CONFIGS
               if lr_results.get(f"y_pred_input{sfx}") is not None
               or lr_results.get(f"y_pred_emb{sfx}") is not None]
    if y is None or not configs:
        return []

    suffix   = f"_{label}" if label else ""
    run_name = label or "checkpoint"

    fig, axes = plt.subplots(len(configs), 2, figsize=(11, 4.6 * len(configs)),
                             squeeze=False)
    fig.suptitle(f"Label Regression — True vs Predicted (parameter_0) — {run_name}",
                 fontsize=12, fontweight="bold")

    def _panel(ax, title, y_pred, color):
        if y_pred is None:
            ax.axis("off")
            return
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

    for row, (sfx, cfg_label) in enumerate(configs):
        _panel(axes[row][0], f"Input probe — {cfg_label}",
               lr_results.get(f"y_pred_input{sfx}"), "#90CAF9")
        _panel(axes[row][1], f"Embedding probe — {cfg_label}",
               lr_results.get(f"y_pred_emb{sfx}"), "#1565C0")

    plt.tight_layout()
    path = _save_fig(fig, os.path.join(output_dir, f"label_reg_true_vs_pred{suffix}.png"))
    return [(f"Label regression true vs predicted (1/2/3-comp) — {run_name}", path)]


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
        sim = _cosim(vectors) if vectors is not None else None
        _struct_sim_panel(ax, sim, title, vmin=vmin, vmax=vmax, ylabel=ylabel)

    # Input column: never centered — raw inputs have no common-mode problem
    inp_vecs = np.asarray(inputs, dtype=np.float64) if inputs is not None else None
    _panel(axes[0][0], inp_vecs, "Input Space (Raw Signals)", ylabel="Embeddings\nSample Index")
    for col, (name, _, emb, _fe) in enumerate(entries, start=1):
        _panel(axes[0][col], _prep(emb), f"{name}\n(Embeddings)")

    if has_fe:
        _panel(axes[1][0], inp_vecs, "Input Space (Raw Signals)", ylabel="FE Output\nSample Index")
        for col, (name, _, _emb, fe) in enumerate(entries, start=1):
            _panel(axes[1][col], _prep(fe), f"{name}\n(FE Output)")

    tag = "centered cosine (mean vector removed)" if centered else "raw cosine"
    fig.suptitle(f"Input Space vs All Models — structured_similarity ({tag})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = "struct_sim_all_models_centered.png" if centered else "struct_sim_all_models.png"
    path = _save_fig(fig, os.path.join(output_dir, fname))
    return [(f"Structured similarity — all models ({tag})", path, "")]


def _plot_reconstruction_all(r: dict, out_dir: str) -> list:
    """
    Every figure for one reconstruction dataset: sample-level first, dataset-level
    second. The report's narrative splits on that order, so it is fixed here.
    """
    from . import recon_plots
    return _plot_signal_reconstruction(r, out_dir) + recon_plots.plot_recon_dataset_level(
        r, out_dir)


# ── Reconstruction report block (sample → dataset → across datasets) ──────────
#
# Reconstruction is rendered by a dedicated block rather than the generic per-eval
# loop, because it is the only eval whose figures form a narrative: a reader has to
# see what one reconstruction looks like before a distribution over thousands of them
# means anything. The generic loop sorts keys alphabetically and would interleave
# datasets and granularities.

_RECON_INTRO = [
    "A **3AE checkpoint** holds one shared data2vec backbone and up to three decoder "
    "heads. Each head reads the backbone at a different depth and each reconstructs the "
    "same 245-point input signal, so comparing them asks *how much of the signal "
    "survives to that depth*:",
    "",
    "| Head | Reads | Shape | Decoder |",
    "|---|---|---|---|",
    "| **FE decoder** | post-LayerNorm conv feature-extractor output | 47 x 512 | "
    "`MirrorDecoder`, ~3.7M params |",
    "| **Projection decoder** | `post_extract_proj`, before the transformer | 47 x 768 | "
    "`TransformerMirrorDecoder`, ~4.1M params |",
    "| **Transformer decoder** | transformer encoder output | 47 x 768 | "
    "`TransformerMirrorDecoder`, ~4.1M params |",
    "",
    "> **Comparing heads is confounded.** The FE decoder is a different architecture on a "
    "narrower input than the other two, so a gap between them mixes encoder information "
    "content with decoder capacity (TASKS.md T7). Comparing *one head across datasets* is "
    "clean; comparing heads to each other is not.",
    "",
    "**Two kinds of data, never pooled.** `single_channel_*` names one wav per sample. "
    "`multi_channel`, `sampled_data` and `labeled_data` name one wav per *component*, and "
    "the physical sample is every wav sharing a `spec` index — so those datasets also get "
    "a per-spectrum view. In `multi_channel` and `labeled_data` (not `sampled_data`) "
    "component 20 is a byte-identical copy of component 14 and 21 of 15; both are dropped "
    "before anything is aggregated, so the sample count for those datasets is lower than "
    "the number drawn.",
    "",
    "The three parts below go from one sample, to one dataset, to all of them. Every "
    "figure is followed by what it shows, how to read it, what a good result looks like "
    "and its caveats; the same text is also drawn onto each PNG, so a figure opened on "
    "its own still explains itself.",
    "",
    "The single number worth checking first is **median R²**. It compares the model "
    "against the trivial predictor that outputs each sample's own mean value as a flat "
    "line: R² = 0 means the decoder is worth exactly nothing, however small its MSE "
    "looks. Medians lead throughout, because a handful of outliers pull the mean far "
    "above the typical sample on the multi-component sets.",
]

_RECON_SUMMARY_COLUMNS = [
    ("head_label",       "Head",                  None),
    ("n",                "n",                     "{:.0f}"),
    ("mse_median",       "MSE median ↓",          "{:.4g}"),
    ("mse_mean",         "MSE mean ↓",            "{:.4g}"),
    ("mse_p90",          "MSE p90 ↓",             "{:.4g}"),
    ("mae_median",       "MAE median ↓",          "{:.4g}"),
    ("r2_median",        "R² median ↑",           "{:.3f}"),
    ("frac_r2_positive", "beats baseline ↑",      "{:.0%}"),
    ("pearson_median",   "Pearson r median ↑",    "{:.3f}"),
    ("amp_ratio_median", "amplitude ratio (1 = ideal)", "{:.3f}"),
]


def _recon_keys(results: dict) -> list:
    """Reconstruction result keys, single-component datasets first."""
    keys = [k for k in results
            if _split_eval_key(k)[0] == "signal_reconstruction"
            and isinstance(results[k], dict)]
    return sorted(keys, key=lambda k: (results[k].get("component_group") == "multi", k))


def _is_dataset_level_fig(path: str) -> bool:
    from . import recon_plots
    base = os.path.basename(path)
    return any(base.startswith(key) for key in recon_plots.DATASET_LEVEL_FIGURE_KEYS)


def _recon_summary_table(r: dict) -> list:
    """Per-head summary as a markdown table — richer than the old key/value cards."""
    sdf = r.get("summary_df")
    if not isinstance(sdf, pd.DataFrame) or sdf.empty:
        return []
    cols = [(c, lbl, fmt) for c, lbl, fmt in _RECON_SUMMARY_COLUMNS if c in sdf.columns]
    lines = ["| " + " | ".join(lbl for _, lbl, _ in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in sdf.iterrows():
        cells = []
        for c, _, fmt in cols:
            v = row[c]
            if fmt is None:
                cells.append(str(v))
            elif pd.isna(v):
                cells.append("–")
            else:
                cells.append(fmt.format(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _recon_dataset_heading(r: dict, alias: str) -> str:
    group = r.get("component_group", "single")
    subset = r.get("dataset_subset") or ""
    n = r.get("n_samples", 0)
    bits = [alias or "dataset"]
    if subset:
        bits.append("`%s`" % os.path.basename(str(subset)))
    return "%s — %s-component, n = %d" % (" / ".join(bits), group, n)


# The four explanation fields, in reading order, with the labels used in the report.
_EXPLAIN_FIELDS = [("what", "What this shows"), ("read", "How to read it"),
                   ("good", "A good result"), ("caveats", "Caveats")]


def _explain_lines(path: str) -> tuple:
    """
    The figure's explanation as (title, [(label, text), ...]), or ("", []) if the figure
    is not registered.

    The same text is drawn onto the PNG so a figure read on its own still makes sense,
    but it is far easier to read here, so the report carries it too.
    """
    from . import recon_plots
    from .evaluations.signal_reconstruction import CROSS_HEAD_CAVEAT

    d = recon_plots.doc_for_figure(path)
    if not d:
        return "", []
    out = []
    for key, label in _EXPLAIN_FIELDS:
        text = (d.get(key) or "").strip()
        # Every figure repeats the cross-head confound verbatim. That is right on a
        # standalone PNG and eighteen-fold repetition in a report that already states it
        # up front, so here it becomes a pointer.
        if CROSS_HEAD_CAVEAT.strip() in text:
            text = text.replace(CROSS_HEAD_CAVEAT.strip(),
                                "Comparing heads to each other is confounded — see the "
                                "note at the top of this section.").strip()
        if not text:
            continue
        if key == "read":                     # stored lowercase, mid-sentence
            text = text[0].upper() + text[1:]
        if not text.endswith("."):
            text += "."
        out.append((label, text))
    return d.get("title", ""), out


def _recon_md_figure(caption: str, path: str, run_dir: str) -> list:
    """One image, its one-line caption, then the full explanation."""
    title, explain = _explain_lines(path)
    # Alt text is the short title, not the caption - the caption is printed right below
    # the image, and repeating a full sentence twice reads badly in raw markdown.
    lines = ["", "![%s](%s)" % (title or caption, os.path.relpath(path, run_dir)),
             "*" + caption + "*", ""]
    for label, text in explain:
        lines.append("- **%s.** %s" % (label, text))
    return lines


def _recon_html_figure(caption: str, path: str) -> str:
    title, explain = _explain_lines(path)
    body = "".join("<li><strong>%s.</strong> %s</li>" % (label, _md_inline_to_html(text))
                   for label, text in explain)
    if body:
        body = '<ul class="figexplain">%s</ul>' % body
    return ("<figure>"
            '<img src="%s" alt="%s">'
            "<figcaption>%s</figcaption>%s"
            "</figure>" % (_fig_to_b64(path), title or caption,
                           _md_inline_to_html(caption), body))


# A short legend for the summary tables: readers otherwise have to guess what a dash
# means and which direction is good.
_RECON_TABLE_LEGEND = (
    "Arrows mark the good direction. **amplitude ratio** is "
    "`std(prediction) / std(target)`: 1 keeps the target's dynamic range, below 1 is "
    "flatter than the target. **beats baseline** is the share of samples the model "
    "reconstructs better than a flat line at that sample's own mean. A dash means the "
    "statistic is undefined — Pearson r has no value when a head emits a constant signal, "
    "and R² has none when the target itself is constant."
)


def _recon_md_block(results: dict, figures_by_eval: dict, summary_figs: list,
                    run_dir: str) -> list:
    keys = _recon_keys(results)
    if not keys:
        return []

    first = results[keys[0]]
    lines = ["", "---", "", "## Signal Reconstruction (3AE)", ""]
    model = first.get("_model_name") or ""
    if model:
        lines += ["**Checkpoint:** `%s`" % model, ""]
    lines += ["**Target convention:** `normalize=%s` — %s, matching training."
              % (first.get("normalize"),
                 "each target is layer-normed to zero mean and unit variance"
                 if first.get("normalize") else "targets are the raw signal"), ""]
    lines += _RECON_INTRO

    # ── 1. Sample level ──────────────────────────────────────────────────────
    lines += ["", "### 1. One sample at a time", "",
              "Individual reconstructions, target in black against each head's output. "
              "These say what the model is doing; they cannot say how often it does it — "
              "that is section 2.", ""]
    for key in keys:
        r = results[key]
        alias = _split_eval_key(key)[1]
        lines += ["#### %s" % _recon_dataset_heading(r, alias), ""]
        if r.get("skipped"):
            lines += ["*Skipped — %s.*" % r.get("error", "n/a"), ""]
            continue
        figs = [f for f in figures_by_eval.get(key, []) if not _is_dataset_level_fig(f[1])]
        if not figs:
            lines += ["*No sample-level figures produced.*", ""]
        for f in figs:
            lines += _recon_md_figure(f[0], f[1], run_dir) + [""]

    # ── 2. Dataset level ─────────────────────────────────────────────────────
    lines += ["---", "", "### 2. The whole dataset", "",
              "The same reconstructions aggregated over every sample drawn: the headline "
              "numbers per head, then which kinds of spectra fail.", "",
              _RECON_TABLE_LEGEND, ""]
    for key in keys:
        r = results[key]
        alias = _split_eval_key(key)[1]
        if r.get("skipped"):
            continue
        lines += ["#### %s" % _recon_dataset_heading(r, alias), ""]
        table = _recon_summary_table(r)
        if table:
            lines += table + [""]
        spec = r.get("spectrum_df")
        if isinstance(spec, pd.DataFrame) and not spec.empty:
            lines += ["Per-spectrum view available: **%d spectra**, aggregating the "
                      "components of each `(dataset, spec)` key. See `spectrum_df.csv`."
                      % len(spec), ""]
        for f in [f for f in figures_by_eval.get(key, []) if _is_dataset_level_fig(f[1])]:
            lines += _recon_md_figure(f[0], f[1], run_dir) + [""]

    # ── 3. Across datasets ───────────────────────────────────────────────────
    if summary_figs:
        lines += ["---", "", "### 3. Across datasets", "",
                  "Every dataset in one view, with single-component and multi-component "
                  "blocks kept visually separate.", ""]
        combined = _recon_combined_table(results)
        if combined:
            lines += ["#### Summary table — all datasets", ""] + combined + [""]
        for f in summary_figs:
            lines += _recon_md_figure(f[0], f[1], run_dir) + [""]

    # Collapse runs of blank lines. The block is assembled from many small pieces that
    # each pad themselves, and doubled blanks render as ragged extra space.
    out = []
    for ln in lines:
        if ln == "" and out and out[-1] == "":
            continue
        out.append(ln)
    return out


def _recon_combined_table(results: dict) -> list:
    """One table over every dataset and head, component group as an explicit column."""
    from . import recon_plots
    by_alias = {_split_eval_key(k)[1] or "dataset": results[k] for k in _recon_keys(results)}
    frame = recon_plots.summary_frame(by_alias)
    if frame.empty:
        return []
    cols = ([("dataset", "Dataset", None), ("component_group", "Group", None)]
            + [(c, lbl, fmt) for c, lbl, fmt in _RECON_SUMMARY_COLUMNS
               if c in frame.columns and c != "head_label"]
            )
    cols.insert(2, ("head_label", "Head", None))
    lines = ["| " + " | ".join(lbl for _, lbl, _ in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for c, _, fmt in cols:
            v = row[c]
            if fmt is None:
                cells.append(str(v))
            elif pd.isna(v):
                cells.append("–")
            else:
                cells.append(fmt.format(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _md_inline_to_html(text: str) -> str:
    """
    Convert the inline markdown the reconstruction block is written in - `code` and
    **bold** - into HTML. The same strings feed both the markdown and HTML reports, and
    previously the HTML path just stripped the markers, so backticks rendered literally.
    """
    import html as _html
    import re as _re
    out = _html.escape(text, quote=False)
    out = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = _re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def _md_to_html_table(lines: list) -> str:
    """Render the markdown tables built above as HTML, reusing the report's own CSS."""
    rows = [ln for ln in lines if ln.strip().startswith("|")]
    if len(rows) < 2:
        return ""
    def cells(ln):
        return [c.strip() for c in ln.strip().strip("|").split("|")]
    head = cells(rows[0])
    body = [cells(ln) for ln in rows[2:]]
    out = (["<table>", "<thead><tr>"]
           + ["<th>%s</th>" % _md_inline_to_html(h) for h in head]
           + ["</tr></thead>", "<tbody>"])
    for row in body:
        out.append("<tr>" + "".join("<td>%s</td>" % _md_inline_to_html(c)
                                    for c in row) + "</tr>")
    out += ["</tbody>", "</table>"]
    return "\n".join(out)


def _recon_html_block(results: dict, figures_by_eval: dict, summary_figs: list) -> str:
    keys = _recon_keys(results)
    if not keys:
        return ""
    first = results[keys[0]]

    def para(text):
        return "<p>%s</p>" % text

    intro_md = [ln for ln in _RECON_INTRO]
    intro_html = [_md_to_html_table(intro_md)]
    for ln in intro_md:
        s = ln.strip()
        if not s or s.startswith("|"):
            continue
        if s.startswith(">"):
            intro_html.append("<blockquote><p>%s</p></blockquote>"
                              % _md_inline_to_html(s.lstrip("> ")))
        else:
            intro_html.append(para(_md_inline_to_html(s)))

    parts = ["<section>", "<h2>Signal Reconstruction (3AE)</h2>"]
    model = first.get("_model_name") or ""
    if model:
        parts.append(para("<strong>Checkpoint:</strong> <code>%s</code>" % model))
    parts.append(para("<strong>Target convention:</strong> <code>normalize=%s</code> — %s, "
                      "matching training."
                      % (first.get("normalize"),
                         "each target is layer-normed to zero mean and unit variance"
                         if first.get("normalize") else "targets are the raw signal")))
    parts += intro_html

    parts += ["<h3>1. One sample at a time</h3>",
              para("Individual reconstructions, target in black against each head's "
                   "output. These say what the model is doing; they cannot say how often "
                   "it does it — that is section 2.")]
    for key in keys:
        r = results[key]
        alias = _split_eval_key(key)[1]
        parts.append("<h4>%s</h4>"
                     % _md_inline_to_html(_recon_dataset_heading(r, alias)))
        if r.get("skipped"):
            parts.append(para("<em>Skipped — %s.</em>" % r.get("error", "n/a")))
            continue
        for f in figures_by_eval.get(key, []):
            if not _is_dataset_level_fig(f[1]):
                parts.append(_recon_html_figure(f[0], f[1]))

    parts += ["<h3>2. The whole dataset</h3>",
              para("The same reconstructions aggregated over every sample drawn: the "
                   "headline numbers per head, then which kinds of spectra fail."),
              para(_md_inline_to_html(_RECON_TABLE_LEGEND))]
    for key in keys:
        r = results[key]
        alias = _split_eval_key(key)[1]
        if r.get("skipped"):
            continue
        parts.append("<h4>%s</h4>"
                     % _md_inline_to_html(_recon_dataset_heading(r, alias)))
        parts.append(_md_to_html_table(_recon_summary_table(r)))
        spec = r.get("spectrum_df")
        if isinstance(spec, pd.DataFrame) and not spec.empty:
            parts.append(para("Per-spectrum view: <strong>%d spectra</strong>, aggregating "
                              "the components of each <code>(dataset, spec)</code> key."
                              % len(spec)))
        for f in figures_by_eval.get(key, []):
            if _is_dataset_level_fig(f[1]):
                parts.append(_recon_html_figure(f[0], f[1]))

    if summary_figs:
        parts += ["<h3>3. Across datasets</h3>",
                  para("Every dataset in one view, with single-component and "
                       "multi-component blocks kept visually separate.")]
        combined = _recon_combined_table(results)
        if combined:
            parts += ["<h4>Summary table — all datasets</h4>", _md_to_html_table(combined)]
        for f in summary_figs:
            parts.append(_recon_html_figure(f[0], f[1]))

    parts.append("</section>")
    return "\n".join(p for p in parts if p)


# Eval name → output subdirectory name (E3 restructure)
_METHOD_DIRS = {
    "embedding_similarity":  "similarity",
    "noise_robustness":      "noise_robustness",
    "clustering":            "clustering",
    "label_regression":      "label_regression",
    "structured_similarity": "structured_similarity",
    "signal_reconstruction": "reconstruction",
}


def _split_eval_key(key: str) -> tuple:
    """
    Split a results key into (base_eval, dataset_alias) — E4 keys carry the
    dataset alias as a suffix, e.g. 'noise_robustness_sanity' → ('noise_robustness',
    'sanity'); unsuffixed keys return alias ''.
    """
    for base in sorted(_METHOD_DIRS, key=len, reverse=True):
        if key == base:
            return base, ""
        if key.startswith(base + "_"):
            return base, key[len(base) + 1:]
    return key, ""


def _method_dirname(base: str, alias: str) -> str:
    d = _METHOD_DIRS.get(base, base)
    return f"{d}_{alias}" if alias else d


def _relocate(figs: list, target_dir: str, strip_label: str = "") -> list:
    """
    Move generated figure files into a per-checkpoint / per-method subdirectory,
    stripping the checkpoint-name suffix from filenames (redundant once the file
    lives inside the checkpoint's own directory). Returns figs with updated paths;
    tuple tails (captions, extra fields) are preserved.
    """
    os.makedirs(target_dir, exist_ok=True)
    out = []
    for fig in figs:
        cap, path, rest = fig[0], fig[1], fig[2:]
        fname = os.path.basename(path)
        if strip_label:
            fname = fname.replace(f"_{strip_label}", "")
        new_path = os.path.join(target_dir, fname)
        if os.path.abspath(path) != os.path.abspath(new_path):
            os.replace(path, new_path)
        out.append((cap, new_path, *rest))
    return out


def _plot_checkpoint_comparison(results: dict, output_dir: str) -> list:
    """
    Returns a list of (caption, path, checkpoint_label) triples.
    checkpoint_label="" for summary figures, checkpoint name for per-checkpoint figures.
    Figures are organized on disk as:
      <run_dir>/comparison/…                      cross-checkpoint figures
      <run_dir>/<checkpoint>/<method>/…           per-checkpoint figures (short names)
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
        # Per-dataset noise comparisons → comparison/<alias>/; label regression
        # (labeled set) → comparison/labeled/
        # NB: aliases contain underscores (in_dist, multi_ch) — match against the
        # known alias set instead of splitting on the last underscore.
        _KNOWN_ALIASES = ("sanity", "in_dist", "multi_ch", "samples", "labeled", "data")
        noise_aliases = sorted({a for c in cdf.columns if c.startswith("noise_")
                                for a in _KNOWN_ALIASES if c.endswith("_" + a)})
        for alias in noise_aliases:
            figs = _plot_noise_robustness_comparison(cdf, output_dir, alias=alias)
            figures += _relocate(figs, os.path.join(output_dir, "comparison", alias))
        lr_figs = _plot_label_regression_comparison(cdf, output_dir)
        figures += _relocate(lr_figs, os.path.join(output_dir, "comparison", "labeled"))

    # ── Summary: all-models structured similarity (raw + centered) ───────────
    if per_cp:
        figures += _plot_struct_sim_all_models(per_cp, output_dir, centered=False)
        figures += _plot_struct_sim_all_models(per_cp, output_dir, centered=True)

    # Remaining cross-checkpoint figures (scalar grid, all-models panels) → comparison/
    figures = [f for f in figures if os.sep + "comparison" + os.sep in f[1]] + _relocate(
        [f for f in figures if os.sep + "comparison" + os.sep not in f[1]],
        os.path.join(output_dir, "comparison"))

    # ── Per-checkpoint figures: <run_dir>/<checkpoint>/<method>_<dataset>/ ────
    _CP_FIG_BUILDERS = {
        "structured_similarity": lambda r, d, lbl: _plot_structured_similarity_maps(r, d, label=lbl),
        "clustering":            lambda r, d, lbl: _plot_clustering_scatter(r, d, label=lbl),
        "noise_robustness":      lambda r, d, lbl: (_plot_noise_example_grid(r, d, label=lbl)
                                                    + _plot_noise_examples(r, d, label=lbl)),
        "label_regression":      lambda r, d, lbl: _plot_label_reg_scatter(r, d, label=lbl),
        "embedding_similarity":  lambda r, d, lbl: _plot_ksimilar_examples(r, d, label=lbl),
    }

    if per_cp:
        for cp_label, cp_res in per_cp.items():
            cp_dir = os.path.join(output_dir, cp_label)
            for key, res in cp_res.items():
                if key.startswith("_") or not isinstance(res, dict):
                    continue
                base, alias = _split_eval_key(key)
                builder = _CP_FIG_BUILDERS.get(base)
                if builder is None:
                    continue
                figs = builder(res, output_dir, cp_label)
                target = os.path.join(cp_dir, _method_dirname(base, alias))
                for cap, path in _relocate(figs, target, strip_label=cp_label):
                    figures.append((cap, path, cp_label))

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
/* The same explanation is drawn onto the PNG, but it is much easier to read here. */
ul.figexplain { margin: .7rem 0 .2rem; padding: .8rem 1rem .8rem 2rem;
                background: #f7f8fa; border-left: 3px solid __ACCENT__;
                border-radius: 3px; font-size: .82rem; color: #444; line-height: 1.55; }
ul.figexplain li { margin-bottom: .45rem; }
ul.figexplain li:last-child { margin-bottom: 0; }
ul.figexplain strong { color: __DARK__; }
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
    return (f"<section>\n<h2>Appendix — configuration</h2>\n"
            f"<p>Every parameter this run was launched with.</p>\n"
            f"{_html_kv(kv)}\n</section>")


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
            if f"recon_{key}_mae_mean" in r:
                cards[f"{label} recon mean MAE"] = f"{r[f'recon_{key}_mae_mean']:.3e}"
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
        ckpt = getattr(config, "checkpoint_path", "") or ""
        data = getattr(config, "data_source", "") or ""
        if not data and getattr(config, "multi_dataset", False):
            data = "multi-dataset (E4)"
        config_meta = (f" &nbsp;|&nbsp; data: <code>{os.path.basename(data) or data}</code>"
                       f" &nbsp;|&nbsp; ckpt: <code>{os.path.basename(str(ckpt))}</code>")

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

    # E4 dataset matrix + sizes
    ds_info = results.get("_dataset_info") or {}
    if ds_info:
        lines += ["", "## Datasets evaluated (E4 multi-dataset mode)", "",
                  "| Alias | Samples |", "|-------|---------|"]
        for alias, n in (ds_info.get("sizes") or {}).items():
            lines.append(f"| {alias} | {n} |")
        lines += ["", "## Dataset x eval matrix (excluded combinations were skipped)", "",
                  "| Eval | Datasets |", "|------|----------|"]
        all_aliases = set((ds_info.get("sizes") or {}))
        for ev, aliases in (ds_info.get("matrix") or {}).items():
            run_on = [a for a in aliases if a in all_aliases]
            skipped = sorted(all_aliases - set(aliases))
            note = f" (skipped: {', '.join(skipped)})" if skipped else ""
            lines.append(f"| {ev} | {', '.join(run_on)}{note} |")
        lines.append("| label_regression | labeled (only) |")
        lines.append("| structured_similarity | run-level canonical panel |")

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
        "| `comparison/` | Cross-checkpoint figures + comparison CSVs |",
        "| `<checkpoint>/<method>/` | Per-checkpoint figures + CSVs (one dir per eval method) |",
        "| `<method>/` | Standalone-eval figures + CSVs (single-model runs) |",
        "| `<model>/reconstruction_<dataset>/` | Per-dataset reconstruction figures + "
        "`recon_df.csv` (per sample), `summary_df.csv` (per head), `strat_df.csv` "
        "(error by signal property), `spectrum_df.csv` (per spectrum, multi-component "
        "datasets only) |",
        "| `<model>/reconstruction_summary/` | Cross-dataset reconstruction figures, "
        "`recon_summary_all_datasets.csv`, and `FIGURES.md` explaining every "
        "reconstruction figure |",
    ]

    path = os.path.join(run_dir, "run_info.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def generate_report(results: dict, output_dir: str, config=None,
                    pdf: bool = False) -> tuple[str, str]:
    """
    Generate markdown + self-contained HTML report from eval results.
    Each run gets its own timestamped subdirectory inside output_dir.
    Returns (md_path, html_path).

    pdf=True additionally builds eval_report_eink.pdf — the same content, re-rendered
    for an e-ink reader: greyscale figures, one per page, with the explanations set as
    real type. Needs pdflatex; skipped with a message if absent.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # All files for this run go into a dedicated subdirectory
    run_dir = os.path.join(output_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    # ── Collect figures per eval ──────────────────────────────────────────────
    # Standalone results may carry an E4 dataset-alias suffix (e.g.
    # noise_robustness_sanity); figures land in <run_dir>/<method>_<alias>/.
    _FIG_BUILDERS = {
        "embedding_similarity": lambda r, d: (_plot_embedding_similarity(r, d)
                                              + _plot_ksimilar_examples(r, d)),
        "signal_reconstruction": _plot_reconstruction_all,
        "noise_robustness": lambda r, d: (_plot_noise_robustness(r, d)
                                          + _plot_noise_example_grid(r, d)
                                          + _plot_noise_examples(r, d)),
        "clustering": _plot_clustering_scatter,
        "label_regression": _plot_label_reg_scatter,
        "structured_similarity": _plot_structured_similarity_maps,
    }

    figures_by_eval: dict[str, list] = {}
    for key, r in results.items():
        if key.startswith("_") or key == "checkpoint_comparison" or not isinstance(r, dict):
            continue
        base, alias = _split_eval_key(key)
        builder = _FIG_BUILDERS.get(base)
        if builder is None:
            continue
        figs = builder(r, run_dir)
        if base == "signal_reconstruction" and r.get("_model_name"):
            # File under the recon model's own directory — coincides with the
            # compared checkpoint's dir when the model is one of them.
            target = os.path.join(run_dir, r["_model_name"],
                                  _method_dirname(base, alias))
        else:
            target = os.path.join(run_dir, _method_dirname(base, alias))
        figures_by_eval[key] = _relocate(figs, target)

    # Cross-dataset reconstruction figures need every dataset at once, so they get a
    # second pass — structurally the same as checkpoint_comparison's.
    recon_summary_figs = []
    recon_keys = _recon_keys(results)
    if recon_keys:
        from . import recon_plots
        by_alias = {(_split_eval_key(k)[1] or "dataset"): results[k] for k in recon_keys}
        model_name = results[recon_keys[0]].get("_model_name") or ""
        summary_dir = os.path.join(*[p for p in (run_dir, model_name,
                                                 "reconstruction_summary") if p])
        try:
            recon_summary_figs = recon_plots.plot_recon_across_datasets(
                by_alias, summary_dir)
            combined = recon_plots.summary_frame(by_alias)
            if not combined.empty:
                combined.to_csv(os.path.join(summary_dir, "recon_summary_all_datasets.csv"),
                                index=False)
        except Exception as e:
            print(f"[Report] cross-dataset reconstruction figures failed: "
                  f"{type(e).__name__}: {e}")

    if "checkpoint_comparison" in results:
        figures_by_eval["checkpoint_comparison"] = _plot_checkpoint_comparison(
            results["checkpoint_comparison"], run_dir
        )

    # checkpoint_comparison returns (cap, path, cp_label) triples; others return (cap, path) pairs
    all_figures = []
    for eval_name, figs in figures_by_eval.items():
        for fig in figs:
            all_figures.append(fig[:2])  # just (caption, path) for counting
    all_figures += [f[:2] for f in recon_summary_figs]

    # ── Run info ──────────────────────────────────────────────────────────────
    _write_run_info(run_dir, ts, results, config)

    # ── Markdown ──────────────────────────────────────────────────────────────
    md_path = os.path.join(run_dir, "eval_report.md")
    lines = [
        "# SpectralFM Evaluation Report",
        "",
        f"**Date:** {ts}",
    ]

    # Configuration is reference material, not the story - it is appended at the end.
    config_lines = []
    if config is not None:
        config_lines = ["", "---", "", "## Appendix — configuration", "",
                        "Every parameter this run was launched with.", "",
                        "| Parameter | Value |", "|-----------|-------|"]
        for k, v in vars(config).items():
            config_lines.append(f"| `{k}` | `{v}` |")

    _MD_TITLES = {
        "embedding_similarity": "Embedding Similarity (stack query)",
        "signal_reconstruction": "Signal Reconstruction",
        "noise_robustness": "Noise Robustness",
        "clustering": "Clustering",
        "label_regression": "Label Regression (parameter_0)",
        "structured_similarity": "Structured Similarity (canonical panel)",
    }

    def _md_metric_lines(base, r):
        out = ["| Metric | Value |", "|--------|-------|"]
        if base == "embedding_similarity":
            out.append(f"| Embedding stack match rate | {r.get('embedding_stack_match_rate', 0):.3f} |")
            out.append(f"| Input stack match rate     | {r.get('input_stack_match_rate', 0):.3f} |")
            out.append(f"| Average match score (0-100)| {r.get('match_score_avg', 0):.1f} |")
        elif base == "signal_reconstruction":
            for key, label in (("fe", "FE"), ("proj", "Projection"), ("tr", "Transformer")):
                if f"recon_{key}_mse_mean" in r:
                    out.append(f"| {label} recon mean MSE | {r[f'recon_{key}_mse_mean']:.6e} |")
                if f"recon_{key}_mae_mean" in r:
                    out.append(f"| {label} recon mean MAE | {r[f'recon_{key}_mae_mean']:.6e} |")
            out.append(f"| normalize | {r.get('normalize')} |")
        elif base == "noise_robustness":
            for k, v in (r.get("summary") or {}).items():
                out.append(f"| {k} | {v:.4f} |")
        elif base == "clustering":
            for key in ("comp_cluster_ari", "comp_cluster_nmi", "comp_cluster_vmeasure",
                        "comp_cluster_silhouette", "comp_cluster_knn_precision",
                        "comp_cluster_retrieval_map"):
                if key in r:
                    out.append(f"| {key.replace('comp_cluster_', '')} | {r[key]:.4f} |")
        elif base == "label_regression":
            for sfx in ("", "_2c", "_3c"):
                for key in ("label_reg_input_r2", "label_reg_emb_r2", "label_reg_improvement_r2"):
                    if f"{key}{sfx}" in r:
                        out.append(f"| {key}{sfx} | {r[f'{key}{sfx}']:.4f} |")
        return out if len(out) > 2 else []

    for key in sorted(figures_by_eval):
        # Reconstruction is rendered by _recon_md_block below, as one narrative running
        # from a single sample out to the whole dataset.
        if key == "checkpoint_comparison" or _split_eval_key(key)[0] == "signal_reconstruction":
            continue
        r = results.get(key, {})
        base, alias = _split_eval_key(key)
        title = _MD_TITLES.get(base, base)
        if alias:
            title += f" — dataset: {alias}"
        lines += ["", f"## {title}", ""]
        if r.get("skipped"):
            lines.append(f"*Skipped — {r.get('error', 'n/a')}.*")
            continue
        lines += _md_metric_lines(base, r)
        for cap, path in figures_by_eval.get(key, []):
            lines += ["", f"![{cap}]({os.path.relpath(path, run_dir)})"]

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
            lines += ["", f"![{cap}]({os.path.relpath(path, run_dir)})"]

    lines += _recon_md_block(results, figures_by_eval, recon_summary_figs, run_dir)
    lines += config_lines

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    # ── HTML ──────────────────────────────────────────────────────────────────
    html_path = os.path.join(run_dir, "eval_report.html")
    sections = []

    def _html_cards(base, r):
        if base == "embedding_similarity":
            return {"Emb stack match rate": f"{r.get('embedding_stack_match_rate', 0):.3f}",
                    "Input stack match rate": f"{r.get('input_stack_match_rate', 0):.3f}",
                    "Match score (0-100)": f"{r.get('match_score_avg', 0):.1f}"}
        if base == "signal_reconstruction":
            cards = {}
            for k2, lbl in (("fe", "FE"), ("proj", "Projection"), ("tr", "Transformer")):
                if f"recon_{k2}_mse_mean" in r:
                    cards[f"{lbl} recon mean MSE"] = f"{r[f'recon_{k2}_mse_mean']:.3e}"
            cards["normalize"] = str(r.get("normalize"))
            return cards
        if base == "noise_robustness":
            return {k: f"{v:.4f}" for k, v in (r.get("summary") or {}).items()}
        if base == "clustering":
            return {k.replace("comp_cluster_", ""): f"{v:.4f}" for k, v in r.items()
                    if k.startswith("comp_cluster_") and isinstance(v, (int, float))}
        if base == "label_regression":
            return {k: f"{v:.4f}" for k, v in r.items()
                    if k.startswith("label_reg_") and isinstance(v, (int, float))}
        return {}

    for key in sorted(figures_by_eval):
        # Reconstruction is rendered by _recon_html_block below.
        if key == "checkpoint_comparison" or _split_eval_key(key)[0] == "signal_reconstruction":
            continue
        r = results.get(key, {})
        base, alias = _split_eval_key(key)
        title = _MD_TITLES.get(base, base)
        if alias:
            title += f" — dataset: {alias}"
        if r.get("skipped"):
            sections.append(f"<section>\n<h2>{title}</h2>\n"
                            f"<p><em>Skipped — {r.get('error', 'n/a')}.</em></p>\n</section>")
            continue
        sections.append(_html_section_generic(
            title, _html_cards(base, r), figures_by_eval.get(key, [])))

    recon_html = _recon_html_block(results, figures_by_eval, recon_summary_figs)
    if recon_html:
        sections.append(recon_html)

    if "checkpoint_comparison" in results:
        sections.append(_html_section_comparison(
            results, figures_by_eval.get("checkpoint_comparison", [])
        ))

    # Reference material last.
    sections.append(_html_section_config(config))

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

    def _csv_dir(*parts):
        d = os.path.join(run_dir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    for eval_name, r in results.items():
        if eval_name.startswith("_") or not isinstance(r, dict):
            continue
        base_eval, alias = _split_eval_key(eval_name)
        # Top-level DataFrames: comparison_df → comparison/, standalone evals → <method>_<alias>/
        for key, val in r.items():
            if isinstance(val, pd.DataFrame):
                base = _CSV_NAMES.get(base_eval, base_eval)
                name = base if key == "results_df" else key
                if eval_name == "checkpoint_comparison":
                    parts = ("comparison",)
                elif base_eval == "signal_reconstruction" and r.get("_model_name"):
                    parts = (r["_model_name"], _method_dirname(base_eval, alias))
                else:
                    parts = (_method_dirname(base_eval, alias),)
                val.to_csv(os.path.join(_csv_dir(*parts), f"{name}.csv"), index=False)
        # Per-checkpoint DataFrames → <checkpoint>/<method>_<alias>/<base>.csv
        for cp_label, cp_res in (r.get("per_checkpoint") or {}).items():
            if not isinstance(cp_res, dict):
                continue
            for sub_name, sub_res in cp_res.items():
                if sub_name.startswith("_") or not isinstance(sub_res, dict):
                    continue
                rdf = sub_res.get("results_df")
                if isinstance(rdf, pd.DataFrame):
                    sub_base, sub_alias = _split_eval_key(sub_name)
                    base = _CSV_NAMES.get(sub_base, sub_base)
                    rdf.to_csv(os.path.join(
                        _csv_dir(cp_label, _method_dirname(sub_base, sub_alias)),
                        f"{base}.csv"), index=False)

    total_figs = len(all_figures)
    if pdf:
        try:
            from . import report_pdf
            report_pdf.build_pdf(results, figures_by_eval, recon_summary_figs,
                                 run_dir, config)
        except Exception as e:
            print(f"[Report] PDF build failed: {type(e).__name__}: {e}")

    print(f"[Report] Run dir  : {run_dir}")
    print(f"[Report] Markdown : {md_path}")
    print(f"[Report] HTML     : {html_path}")
    print(f"[Report] Figures  : {total_figs}")
    return md_path, html_path
