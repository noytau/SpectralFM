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


def _plot_signal_completion(results: dict, output_dir: str) -> list:
    figures = []
    if results.get("skipped"):
        return figures
    rdf = results.get("results_df")
    if rdf is None:
        return figures

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rdf["mse"], bins=30, color="coral", alpha=0.85)
    ax.axvline(rdf["mse"].mean(), color="darkred", linestyle="--",
               label=f"Mean = {rdf['mse'].mean():.4f}")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Count")
    ax.set_title("Signal Completion: MSE Distribution")
    ax.legend()
    path = _save_fig(fig, os.path.join(output_dir, "signal_completion_mse.png"))
    figures.append(("Signal completion MSE distribution", path))

    for label, sub in [("best", rdf.nsmallest(3, "mse")), ("worst", rdf.nlargest(3, "mse"))]:
        fig, axes = plt.subplots(1, len(sub), figsize=(5 * len(sub), 3))
        if len(sub) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, sub.iterrows()):
            ax.plot(row["inputs"],    label="Original",  linestyle="--", alpha=0.7)
            ax.plot(row["masked"],    label="Masked",    linestyle=":",  alpha=0.7)
            ax.plot(row["predicted"], label="Predicted", linewidth=1.5)
            ax.set_title(f"MSE = {row['mse']:.4f}")
            ax.legend(fontsize=7)
        fig.suptitle(f"Signal Completion: {label.upper()} samples")
        path = _save_fig(fig, os.path.join(output_dir, f"signal_completion_{label}.png"))
        figures.append((f"Signal completion {label} samples", path))

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

    # ── Per-checkpoint: structured similarity 4-panel only ───────────────────
    if per_cp:
        for cp_label, cp_res in per_cp.items():
            ss = cp_res.get("structured_similarity", {})
            if ss:
                ss_figs = _plot_structured_similarity_maps(ss, output_dir, label=cp_label)
                figures += [(cap, path, cp_label) for cap, path in ss_figs]

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


def _html_section_signal_completion(results: dict, figures: list) -> str:
    r = results.get("signal_completion", {})
    if r.get("skipped"):
        body = "<p><em>Skipped — model has no completion_head.</em></p>"
    else:
        body = _html_metric_cards({"Average MSE": r.get("avg_mse", "N/A")})
        body += "\n" + "\n".join(_html_figure(cap, path) for cap, path in figures)
    return f"<section>\n<h2>Signal Completion</h2>\n{body}\n</section>"


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
        figures_by_eval["embedding_similarity"] = _plot_embedding_similarity(
            results["embedding_similarity"], run_dir
        )
    if "signal_completion" in results:
        figures_by_eval["signal_completion"] = _plot_signal_completion(
            results["signal_completion"], run_dir
        )
    if "noise_robustness" in results:
        figures_by_eval["noise_robustness"] = _plot_noise_robustness(
            results["noise_robustness"], run_dir
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

    if "signal_completion" in results:
        r = results["signal_completion"]
        lines += ["", "## Signal Completion"]
        if r.get("skipped"):
            lines.append("*Skipped — model has no completion_head.*")
        else:
            lines += [
                "", "| Metric | Value |", "|--------|-------|",
                f"| Average MSE | {r.get('avg_mse', 'N/A'):.6f} |",
            ]
            for cap, path in figures_by_eval.get("signal_completion", []):
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
    if "signal_completion" in results:
        sections.append(_html_section_signal_completion(
            results, figures_by_eval.get("signal_completion", [])
        ))
    if "noise_robustness" in results:
        sections.append(_html_section_noise(
            results, figures_by_eval.get("noise_robustness", [])
        ))
    if "checkpoint_comparison" in results:
        sections.append(_html_section_comparison(
            results, figures_by_eval.get("checkpoint_comparison", [])
        ))

    with open(html_path, "w") as f:
        f.write(_build_html(results, config, ts, sections))

    # ── CSV exports ───────────────────────────────────────────────────────────
    for eval_name, r in results.items():
        if not isinstance(r, dict):
            continue
        for key, val in r.items():
            if isinstance(val, pd.DataFrame):
                val.to_csv(os.path.join(run_dir, f"{eval_name}_{key}.csv"), index=False)

    total_figs = len(all_figures)
    print(f"[Report] Run dir  : {run_dir}")
    print(f"[Report] Markdown : {md_path}")
    print(f"[Report] HTML     : {html_path}")
    print(f"[Report] Figures  : {total_figs}")
    return md_path, html_path
