"""
Figures for the search side of the label probe (from `label_probe_results.
json`, drawn by `study._write_figures`): the per-block depth profile, the
recipe-search bars, the probe-choice grid, and the axis-fixed
true-vs-predicted grid. The per-n_train honest crossover lives in
panel_plots.py instead, over `recipe_panel.json` -- see that module's
docstring for why it's a separate concern.

Every figure renders on a light surface (the HTML report frames them in a
card), so the light column of the palette is the one in use. `_style_axes`,
`_r2_of`, `_sd_of`, `_n_keys`, `_crossing_n` and the palette constants are
shared with panel_plots.py, which imports them from here.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical slots 1-3 (blue / orange / aqua), light-surface steps.
# Fixed order, never cycled: slot 1 is always the primary comparison series.
# These three clear the all-pairs colour-vision gates as a set, which the
# small-multiple and scatter forms below need; every figure that uses them
# also ships a legend AND direct labels, and the surrounding report carries
# the same numbers as tables, so identity is never colour-alone.
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

_INK = "#0b0b0b"
_INK_2 = "#52514e"
_INK_MUTED = "#8a8985"
_GRID = "#e4e4e1"
_SURFACE = "#ffffff"


def _style_axes(ax, *, grid_axis="both"):
    """Recessive grid and axes: the data is the ink, the frame is not."""
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK_2, labelsize=8.5, length=3, width=0.8)
    ax.grid(True, axis=grid_axis, color=_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _r2_of(score) -> float:
    """Search tables hold {r2, repeat_sd, bootstrap_sd} dicts."""
    return score["r2"] if isinstance(score, dict) else float(score)


def _sd_of(score, which="repeat_sd") -> float:
    return float(score.get(which, 0.0)) if isinstance(score, dict) else 0.0


def _n_keys(by_n: dict) -> list:
    """JSON round-trips the n_train keys to strings; sort them numerically
    and hand back the original keys so callers can index straight back in."""
    return sorted(by_n, key=lambda k: int(k))


def _scatter_cell(ax, y_true, y_pred, color, axis_limits, is_best, r2):
    lo, hi = axis_limits
    ax.scatter(y_true, y_pred, s=3, alpha=0.18, color=color, rasterized=True,
               linewidths=0)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1, color=_INK_2,
            zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    _style_axes(ax)
    ax.tick_params(labelsize=7)
    ax.annotate(f"R² = {r2:.3f}", xy=(0.045, 0.955), xycoords="axes fraction",
                fontsize=9, color=_INK, ha="left", va="top",
                fontweight="700" if is_best else "600",
                bbox=dict(boxstyle="round,pad=0.28", facecolor=_SURFACE,
                          edgecolor=_GRID, linewidth=0.7))
    if is_best:
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_PALETTE[0])
            ax.spines[side].set_linewidth(1.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(_PALETTE[0])
            ax.spines[side].set_linewidth(1.8)


def plot_true_vs_pred_grid(cells: dict, output_path: str,
                            axis_limits=(-2.0, 2.0), n_samples: int = None) -> str:
    """
    cells: {(row_label, col_label): {"y_true": arr, "y_pred": arr}}.

    One column per benchmark, one row per component count. Every panel shares
    the SAME axis limits (default [-2, 2]) and an equal aspect, so a
    badly-scaled tight cloud cannot pass for a good fit; points outside the
    limits are clipped from view rather than silently rescaling the frame.
    The best cell WITHIN EACH ROW is outlined -- comparing across rows would
    be comparing different component counts, which is not a like-for-like
    contest.
    """
    import textwrap

    row_labels = sorted({r for r, _ in cells})
    col_labels = list(dict.fromkeys(c for _, c in cells))
    n_rows, n_cols = len(row_labels), len(col_labels)

    r2s = {k: 1.0 - np.sum((v["y_true"] - v["y_pred"]) ** 2) /
           (np.sum((v["y_true"] - v["y_true"].mean()) ** 2) + 1e-12)
           for k, v in cells.items()}
    best_in_row = {}
    for r in row_labels:
        here = {k: v for k, v in r2s.items() if k[0] == r}
        if here:
            best_in_row[r] = max(here, key=here.get)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.95 * n_cols, 3.05 * n_rows + 1.5),
                             squeeze=False)
    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            ax = axes[i][j]
            key = (row, col)
            if key not in cells:
                ax.axis("off")
                continue
            _scatter_cell(ax, cells[key]["y_true"], cells[key]["y_pred"],
                          _PALETTE[j % len(_PALETTE)], axis_limits,
                          is_best=(best_in_row.get(row) == key), r2=r2s[key])
            # axis labels only on the outer edge -- repeating them on 15 panels
            # is clutter, and every panel shares the same scale anyway
            ax.set_xlabel("true" if i == n_rows - 1 else "", fontsize=8.5,
                          color=_INK_2)
            ax.set_ylabel("predicted" if j == 0 else "", fontsize=8.5,
                          color=_INK_2)
            if i < n_rows - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

    fig.tight_layout(rect=[0.035, 0.035, 1, 0.88])

    # Column headers go ABOVE the grid, wrapped, so long recipe names cannot
    # collide with their neighbours the way per-axes titles did.
    for j, col in enumerate(col_labels):
        box = axes[0][j].get_position()
        fig.text(box.x0 + box.width / 2, 0.895,
                 textwrap.fill(col.replace("embedding: ", "embedding\n"), 30),
                 ha="center", va="bottom", fontsize=8.5, color=_INK,
                 fontweight="600", linespacing=1.35)

    # Row labels once, down the left-hand side.
    for i, row in enumerate(row_labels):
        box = axes[i][0].get_position()
        fig.text(0.012, box.y0 + box.height / 2, row, rotation=90,
                 ha="left", va="center", fontsize=10, color=_INK,
                 fontweight="600")

    pool = f"  ·  n={n_samples:,}" if n_samples else ""
    fig.suptitle("Label regression — true vs. predicted" + pool,
                 fontsize=13, color=_INK, fontweight="600", x=0.012, ha="left",
                 y=0.985)
    fig.text(0.012, 0.008,
             f"every panel fixed to [{axis_limits[0]:.0f}, {axis_limits[1]:.0f}] "
             "on both axes, equal aspect · dashed line is y=x · points outside "
             "the limits are clipped, not rescaled · outlined panel is the best "
             "benchmark within that row",
             fontsize=8, color=_INK_MUTED, ha="left")
    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return output_path


# ── labels shared by the comparison figures ───────────────────────────────

def _crossing_n(ns, gaps):
    """Log-linear interpolation of the first upward zero crossing of the
    gap curve. Returns None when the gap never reaches zero -- which is the
    answer for most of these curves, not a failure."""
    for i in range(len(ns) - 1):
        g0, g1 = gaps[i], gaps[i + 1]
        if g0 < 0 <= g1:
            f = -g0 / (g1 - g0)
            return float(10 ** (np.log10(ns[i]) + f * (np.log10(ns[i + 1]) - np.log10(ns[i]))))
    return None


def plot_depth_profile(stage_scores: dict, output_path: str,
                        raw_reference: float = None,
                        raw_reference_sd: float = None,
                        raw_reference_label: str = "whitened",
                        display_name=None, n_comp: int = 1,
                        n_samples: int = None, n_repeats: int = None) -> str:
    """
    The block scoreboard as a shape: label signal against pipeline depth,
    FE through the final Transformer layer, one line per normalizer.
    Error bars are the split-assignment SD across repeated shuffled 5-fold
    splits -- the uncertainty that decides whether one block really beats
    the next one on this data.

    `raw_reference` is the raw-input dashed line: the caller picks it as the
    BEST of the whitened/z-scored full-pool scores (not always whitened --
    which normalizer wins depends on the dataset), and `raw_reference_sd`
    is that same recipe's bootstrap SD, shown as a shaded band and in the
    line's own label so the comparison against each block carries its
    uncertainty too, not just a bare number.
    """
    n_txt = f", n={n_samples:,}" if n_samples else ""
    rep_txt = f"{n_repeats} repeated 5-fold splits; " if n_repeats else ""
    # kept short: a long y-label runs off the left edge of the canvas
    y_label = (f"R²   ({n_comp} component{'s' if n_comp != 1 else ''}"
               f"{n_txt})")
    foot = (f"mean-pooled, RidgeCV, full pool; {rep_txt}"
            f"error bars: ±1 SD across repeated splits")
    # Depth order from whatever stages this backbone actually has -- never a
    # hardcoded layer count, which a different Transformer won't match.
    all_stages = {k.split("|")[0] for k in stage_scores}
    layer_stages = sorted((s for s in all_stages if s.startswith("layer") and s[5:].isdigit()),
                          key=lambda s: int(s[5:]))
    fe_stages = [s for s in ("fe", "extract_features") if s in all_stages]
    order = fe_stages + layer_stages
    present = [s for s in order if any(k.startswith(s + "|") for k in stage_scores)]

    def score(stage, norm):
        return stage_scores.get(f"{stage}|mean|{norm}")

    def val(stage, norm):
        sc = score(stage, norm)
        return _r2_of(sc) if sc is not None else None

    def err(stage, norm):
        sc = score(stage, norm)
        return _sd_of(sc) if sc is not None else 0.0

    xs = list(range(len(present)))
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_axes(ax, grid_axis="y")

    for i, norm in enumerate(("standardize", "whiten")):
        ys = [val(s, norm) for s in present]
        es = [err(s, norm) for s in present]
        ax.errorbar(xs, ys, yerr=es, marker="o", markersize=5, linewidth=1.9,
                    color=_PALETTE[i], label=norm, zorder=3,
                    elinewidth=1.2, capsize=3, ecolor=_PALETTE[i])

    if raw_reference is not None:
        if raw_reference_sd:
            ax.axhspan(raw_reference - raw_reference_sd, raw_reference + raw_reference_sd,
                       color=_INK_2, alpha=0.08, zorder=1, linewidth=0)
        ax.axhline(raw_reference, color=_INK_2, linestyle="--", linewidth=1.2, zorder=2)
        sd_txt = f" ±{raw_reference_sd:.3f}" if raw_reference_sd else ""
        ax.annotate(f"{raw_reference_label} raw input · {raw_reference:.3f}{sd_txt}",
                    xy=(xs[-1], raw_reference), xytext=(0, 5),
                    textcoords="offset points", fontsize=8.5, color=_INK_2,
                    ha="right", va="bottom")

    std = [val(s, "standardize") for s in present]
    best_i = int(np.argmax([v if v is not None else -9 for v in std]))
    ax.annotate(f"best · {std[best_i]:.3f}", xy=(xs[best_i], std[best_i]),
                xytext=(0, 10), textcoords="offset points", fontsize=9,
                color=_INK, ha="center", fontweight="600")
    if layer_stages and layer_stages[-1] in present:
        j = present.index(layer_stages[-1])  # final Transformer block, whatever its index
        ax.annotate(f"conventional tap · {std[j]:.3f}", xy=(xs[j], std[j]),
                    xytext=(-6, -14), textcoords="offset points", fontsize=8.5,
                    color=_INK, ha="right", va="top")

    names = [display_name(s) if display_name else s for s in present]
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=9.5, color=_INK_2)
    ax.set_title("Where the label signal lives",
                 fontsize=12.5, color=_INK, fontweight="600", loc="left", pad=12)
    ax.annotate(foot, xy=(0.0, -0.34), xycoords="axes fraction", fontsize=8,
                color=_INK_MUTED, ha="left", va="top")
    leg = ax.legend(fontsize=8.5, frameon=False, title="normalizer")
    leg.get_title().set_fontsize(8.5)
    leg.get_title().set_color(_INK_MUTED)
    for t in leg.get_texts():
        t.set_color(_INK_2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return output_path


def plot_search_bars(embedding_search: dict, output_path: str,
                      n_comp: int = 1, n_samples: int = None) -> str:
    """
    What the recipe search actually bought, and what it ruled out: pooling
    schemes on the winning block, and probes on the winning readout.
    """
    pooling = {k: v for k, v in embedding_search.get("pooling_scores", {}).items()
               if k.endswith("standardize")}
    probes = embedding_search.get("probe_scores", {})

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6),
                              gridspec_kw={"width_ratios": [1, 1.15]})

    def bars(ax, scores, title, relabel):
        items = sorted(scores.items(), key=lambda kv: _r2_of(kv[1]))
        names = [relabel(k) for k, _ in items]
        vals = [_r2_of(v) for _, v in items]
        errs = [_sd_of(v) for _, v in items]
        best = max(range(len(vals)), key=lambda i: vals[i])
        colors = [_PALETTE[0] if i == best else "#c9c9c5" for i in range(len(vals))]
        ax.barh(range(len(vals)), vals, color=colors, height=0.62, zorder=3,
                xerr=errs, error_kw=dict(elinewidth=1.1, capsize=2.5,
                                          ecolor=_INK_2, zorder=4))
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=8.5)
        _style_axes(ax, grid_axis="x")
        ax.set_xlim(min(0, min(vals)) - 0.02, max(vals) * 1.30)
        for i, v in enumerate(vals):
            lbl = f"{v:.3f} ±{errs[i]:.3f}" if errs[i] else f"{v:.3f}"
            ax.annotate(lbl, xy=(v + errs[i], i), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=_INK if i == best else _INK_2,
                        fontweight="600" if i == best else "normal")
        ax.set_title(title, fontsize=10.5, color=_INK, fontweight="600",
                     loc="left", pad=8)

    bars(axes[0], pooling, "Pooling scheme  (winning block, standardized)",
         lambda k: k.split("|")[1])
    bars(axes[1], probes, "Probe  (winning block + pooling)", lambda k: k)
    n_txt = f", n={n_samples:,}" if n_samples else ""
    for ax in axes:
        ax.set_xlabel(f"R²  ({n_comp} component{'s' if n_comp != 1 else ''}, "
                      f"full pool{n_txt}; ±1 SD across repeated splits)",
                      fontsize=9, color=_INK_2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return output_path


_PROBE_LABEL = {"ridgecv": "RidgeCV", "ols": "OLS"}
_PROBE_ORDER = ("ridgecv", "ols")


def plot_probe_comparison(probe_grid: dict, output_path: str, n_comp: int = 1,
                           n_samples: int = None) -> str:
    """
    RidgeCV vs OLS, at every normalizer each was actually run with, on raw
    input and on every embedding ARM in `probe_grid` -- plain single-layer
    readouts, never a pooling-squeezed recipe, so this figure never
    conflates "which embedding" with "which pooling scheme". One panel per
    arm, arm identity always in its title. A negative OLS score is omitted
    rather than plotted or clipped -- it carries no information beyond
    "unusable here", and would otherwise force every other bar's axis down
    to accommodate it.
    """
    arms = list(dict.fromkeys(v["arm"] for v in probe_grid.values()))
    any_omitted = False

    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 4.6), squeeze=False)
    axes = axes[0]
    for ax, arm in zip(axes, arms):
        rows = {}
        for v in probe_grid.values():
            if v["arm"] != arm:
                continue
            rows.setdefault(v["normalizer"], {})[v["probe"]] = v["r2_mean"]
        norms = sorted(rows, key=lambda nm: -max(rows[nm].values()))
        probes = sorted({p for r in rows.values() for p in r},
                        key=lambda p: _PROBE_ORDER.index(p) if p in _PROBE_ORDER else 9)
        x = np.arange(len(norms))
        width = 0.8 / max(1, len(probes))
        ymax = 0.05
        for i, probe in enumerate(probes):
            vals = [rows[nm].get(probe) for nm in norms]
            xs = x + (i - (len(probes) - 1) / 2) * width
            present = [(xx, v) for xx, v in zip(xs, vals) if v is not None and v >= 0]
            any_omitted = any_omitted or any(v is not None and v < 0 for v in vals)
            if not present:
                continue
            ymax = max(ymax, max(v for _, v in present))
            ax.bar([p[0] for p in present], [v for _, v in present], width=width * 0.92,
                   color=_PALETTE[i % len(_PALETTE)],
                   label=_PROBE_LABEL.get(probe, probe), zorder=3)
            for xx, v in present:
                ax.annotate(f"{v:.2f}", xy=(xx, v), xytext=(0, 3),
                           textcoords="offset points", ha="center", va="bottom",
                           fontsize=7.6, color=_INK)
        _style_axes(ax, grid_axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels(norms, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, ymax * 1.15)
        ax.set_title(arm, fontsize=9.5, color=_INK, fontweight="600", loc="left",
                    wrap=True)
        ax.set_ylabel("R²", fontsize=9, color=_INK_2)
        leg = ax.legend(fontsize=8.2, frameon=False)
        for t in leg.get_texts():
            t.set_color(_INK_2)

    n_txt = f", n={n_samples:,}" if n_samples else ""
    fig.suptitle(f"Probe choice by normalizer and embedding · {n_comp} component"
                 f"{'s' if n_comp != 1 else ''}, full pool{n_txt}",
                 fontsize=12, color=_INK, fontweight="600", x=0.01, ha="left")
    if any_omitted:
        fig.text(0.01, -0.02, "bars with R² < 0 omitted", fontsize=7.6,
                 color=_INK_MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return output_path
