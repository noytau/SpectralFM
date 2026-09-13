"""
Figures for the label probe: label-efficiency ladders, the embedding-vs-raw
crossover, the per-block depth profile, the recipe-search bars, and the
axis-fixed true-vs-predicted grid.

Every figure renders on a light surface (the HTML report frames them in a
card), so the light column of the palette is the one in use.
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


def plot_label_efficiency(ladder_results: dict, output_path: str) -> str:
    """
    ladder_results: {readout_label: {n_train: {r2_median, r2_p25, r2_p75, ...}}}
    x = n_train (log), y = median held-out R2 with an IQR band, one line per
    readout.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (label, by_n) in enumerate(ladder_results.items()):
        n_trains = sorted(by_n)
        med = [by_n[n]["r2_median"] for n in n_trains]
        lo = [by_n[n]["r2_p25"] for n in n_trains]
        hi = [by_n[n]["r2_p75"] for n in n_trains]
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(n_trains, med, marker="o", label=label, color=color)
        ax.fill_between(n_trains, lo, hi, color=color, alpha=0.15)
    ax.axhline(0, color="black", linestyle=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("labeled training samples (n_train)")
    ax.set_ylabel("held-out R² (median, IQR band)")
    ax.set_title("Label efficiency: raw input vs. embedding readouts")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def _scatter_cell(ax, y_true, y_pred, color, title, axis_limits, is_best):
    r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-12)
    ax.scatter(y_true, y_pred, s=3, alpha=0.2, color=color, rasterized=True)
    lo, hi = axis_limits
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    best_tag = "★ BEST  " if is_best else ""
    ax.set_title(f"{best_tag}{title}\nR²={r2:.3f}", fontsize=8.5,
                 fontweight="bold" if is_best else "normal",
                 color="darkgreen" if is_best else "black")
    if is_best:
        for spine in ax.spines.values():
            spine.set_edgecolor("gold")
            spine.set_linewidth(3)
        ax.patch.set_facecolor("#fffff0")
    ax.set_xlabel("True", fontsize=7)
    ax.set_ylabel("Pred", fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=6)


def plot_true_vs_pred_grid(cells: dict, output_path: str,
                            axis_limits=(-2.0, 2.0)) -> str:
    """
    cells: {(row_label, col_label): {"y_true": arr, "y_pred": arr}}.

    Every panel shares the SAME axis limits (default [-2, 2]) so panels are
    directly, visually comparable — a badly-scaled tight cloud cannot look
    as good as a properly-scaled fit here. Points outside `axis_limits` are
    clipped from view rather than silently rescaling the frame.
    """
    row_labels = sorted({r for r, _ in cells})
    col_labels = list(dict.fromkeys(c for _, c in cells))  # preserve insertion order

    n_rows, n_cols = len(row_labels), len(col_labels)
    r2s = {k: 1.0 - np.sum((v["y_true"] - v["y_pred"]) ** 2) /
           (np.sum((v["y_true"] - v["y_true"].mean()) ** 2) + 1e-12)
           for k, v in cells.items()}
    best_key = max(r2s, key=r2s.get)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.6 * n_rows),
                              squeeze=False)
    fig.suptitle("Label regression — true vs. predicted (axes fixed across all panels)",
                 fontsize=13, fontweight="bold")
    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            key = (row, col)
            ax = axes[i][j]
            if key not in cells:
                ax.axis("off")
                continue
            color = _PALETTE[j % len(_PALETTE)]
            title = f"{col}\n{row}" if i == 0 else row
            _scatter_cell(ax, cells[key]["y_true"], cells[key]["y_pred"], color,
                          title, axis_limits, is_best=(key == best_key))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


# ── labels shared by the comparison figures ───────────────────────────────

RAW_WHITENED = "raw input (whitened), RidgeCV"


def _find(labels, *needles, exclude=()):
    """Ladder/full-pool dicts are keyed by the human-readable recipe label,
    which embeds the winning block name and so is not knowable up front.
    Match on substrings instead of hard-coding the whole string."""
    for lab in labels:
        low = lab.lower()
        if all(n.lower() in low for n in needles) and not any(
                x.lower() in low for x in exclude):
            return lab
    return None


def _series_for(by_n_comp: dict, n_comp_key: str):
    """(raw, embedding-best, embedding-conventional) labels for one n_comp."""
    ladder = by_n_comp[n_comp_key]["ladder"]
    raw = _find(ladder, "raw input (whitened)") or RAW_WHITENED
    emb_conv = _find(ladder, "embedding", "conventional")
    emb_best = _find(ladder, "embedding", exclude=("conventional",))
    return raw, emb_best, emb_conv


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


def _paired_gaps(ladder, emb_label, raw_label, key):
    """Per-draw embedding-minus-raw differences at one rung. Both arms are
    scored on the SAME draws (same seed, same pool), so differencing draw by
    draw cancels the draw-to-draw variance that dominates at small n -- far
    tighter, and the right uncertainty for 'is one ahead of the other'."""
    a = ladder[emb_label][key].get("r2_draws")
    b = ladder[raw_label][key].get("r2_draws")
    if a and b and len(a) == len(b):
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if pairs:
            return np.array([x - y for x, y in pairs], dtype=float)
    # older runs kept only summary stats: fall back to the median difference,
    # which has no spread to report
    return np.array([ladder[emb_label][key]["r2_median"]
                     - ladder[raw_label][key]["r2_median"]], dtype=float)


def _crossing_ci(ns, gap_draws, n_boot=2000, seed=0):
    """Bootstrap CI for the crossing point: resample draws at every rung,
    recompute the median-gap curve, re-solve the crossing. Rungs whose draws
    are a single value (the full pool) resample to themselves."""
    rng = np.random.default_rng(seed)
    crossings = []
    for _ in range(n_boot):
        med = []
        for g in gap_draws:
            idx = rng.integers(0, len(g), len(g))
            med.append(float(np.median(g[idx])))
        c = _crossing_n(ns, med)
        if c is not None:
            crossings.append(c)
    if len(crossings) < 0.5 * n_boot:
        # the curve fails to cross in most resamples -- no stable crossing
        return None, None, len(crossings) / n_boot
    lo, hi = np.percentile(crossings, [2.5, 97.5])
    return float(lo), float(hi), len(crossings) / n_boot


def plot_crossover(by_n_comp: dict, output_path: str, n_samples: int = None) -> str:
    """
    The question the ladder exists to answer, stated directly: how far is the
    best embedding recipe from whitened raw input, and does that gap ever
    reach zero? One line per component count, with the interquartile band of
    the PAIRED per-draw differences; above zero the embedding is ahead.
    """
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    _style_axes(ax)

    xmin, xmax = 1e9, 0
    annotated = []
    summary = []
    for i, n_comp_key in enumerate(sorted(by_n_comp, key=lambda k: int(k))):
        ladder = by_n_comp[n_comp_key]["ladder"]
        raw, emb_best, _ = _series_for(by_n_comp, n_comp_key)
        if emb_best is None:
            continue
        keys = _n_keys(ladder[raw])
        ns = [int(k) for k in keys]
        gap_draws = [_paired_gaps(ladder, emb_best, raw, k) for k in keys]
        med = [float(np.median(g)) for g in gap_draws]
        lo = [float(np.percentile(g, 25)) if len(g) > 1 else float(np.median(g))
              for g in gap_draws]
        hi = [float(np.percentile(g, 75)) if len(g) > 1 else float(np.median(g))
              for g in gap_draws]

        color = _PALETTE[i % len(_PALETTE)]
        label = f"{n_comp_key} component" + ("s" if int(n_comp_key) != 1 else "")
        ax.plot(ns, med, marker="o", markersize=4.8, linewidth=1.9, color=color,
                label=label, zorder=3)
        ax.fill_between(ns, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=1)
        xmin, xmax = min(xmin, min(ns)), max(xmax, max(ns))
        annotated.append((ns[-1], med[-1], n_comp_key))

        cross = _crossing_n(ns, med)
        if cross is not None:
            c_lo, c_hi, frac = _crossing_ci(ns, gap_draws)
            ax.axvline(cross, color=color, linestyle="--", linewidth=1.1,
                       alpha=0.55, zorder=1)
            if c_lo is not None:
                ax.axvspan(c_lo, c_hi, color=color, alpha=0.10, zorder=0)
                txt = f"crossover\nn ≈ {cross:,.0f}\n[{c_lo:,.0f}–{c_hi:,.0f}]"
            else:
                txt = f"crossover\nn ≈ {cross:,.0f}"
            ax.annotate(txt, xy=(cross, 0), xytext=(cross * 1.08, 0.045),
                        fontsize=8.5, color=_INK, ha="left", va="bottom",
                        linespacing=1.3)
            summary.append((n_comp_key, cross, c_lo, c_hi))
        else:
            summary.append((n_comp_key, None, None, None))

    ax.axhline(0, color=_INK_2, linewidth=1.4, zorder=2)
    lo_y, hi_y = ax.get_ylim()
    pad = 0.02 * (hi_y - lo_y)
    ax.axhspan(0, hi_y, color=_PALETTE[0], alpha=0.05, zorder=0)
    ax.axhspan(lo_y, 0, color=_INK_MUTED, alpha=0.07, zorder=0)
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlim(xmin, xmax * 1.45)

    ax.text(xmin * 1.08, hi_y - pad, "embedding ahead", fontsize=8.5,
            color=_INK_2, va="top", ha="left")
    ax.text(xmin * 1.08, lo_y + pad, "raw input ahead", fontsize=8.5,
            color=_INK_2, va="bottom", ha="left")

    for x, y, key in annotated:
        ax.annotate(f"{key}-comp", xy=(x, y), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=_INK_2,
                    va="center", ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("labeled training samples (n_train)", fontsize=9.5, color=_INK_2)
    ax.set_ylabel("Δ R²   (best embedding recipe − whitened raw input)",
                  fontsize=9.5, color=_INK_2)
    pool = f"  ·  pool n={n_samples:,}" if n_samples else ""
    ax.set_title("Does the embedding ever overtake raw input?",
                 fontsize=12.5, color=_INK, fontweight="600", loc="left", pad=12)
    ax.annotate("median of paired per-draw differences; band = IQR across draws; "
                "bracket = 95% bootstrap CI on the crossing" + pool,
                xy=(0.0, -0.135), xycoords="axes fraction", fontsize=8,
                color=_INK_MUTED, ha="left", va="top")
    leg = ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    for t in leg.get_texts():
        t.set_color(_INK_2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    for key, c, cl, ch in summary:
        if c is None:
            print(f"[plots] crossover {key}-comp: never crosses", flush=True)
        else:
            ci = f" [95% CI {cl:,.0f}-{ch:,.0f}]" if cl is not None else ""
            print(f"[plots] crossover {key}-comp: n≈{c:,.0f}{ci}", flush=True)
    return output_path


def plot_depth_profile(stage_scores: dict, output_path: str,
                        raw_reference: float = None,
                        display_name=None, n_comp: int = 1,
                        n_samples: int = None, n_repeats: int = None) -> str:
    """
    The block scoreboard as a shape: label signal against pipeline depth,
    FE through the final Transformer layer, one line per normalizer.
    Error bars are the split-assignment SD across repeated shuffled 5-fold
    splits -- the uncertainty that decides whether one block really beats
    the next one on this data.
    """
    n_txt = f", n={n_samples:,}" if n_samples else ""
    rep_txt = f"{n_repeats} repeated 5-fold splits; " if n_repeats else ""
    # kept short: a long y-label runs off the left edge of the canvas
    y_label = (f"R²   ({n_comp} component{'s' if n_comp != 1 else ''}"
               f"{n_txt})")
    foot = (f"mean-pooled, RidgeCV, full pool; {rep_txt}"
            f"error bars: ±1 SD across repeated splits")
    order = ["fe", "extract_features", "layer0"] + [f"layer{i}" for i in range(1, 13)]
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
        ax.axhline(raw_reference, color=_INK_2, linestyle="--", linewidth=1.2, zorder=2)
        ax.annotate(f"whitened raw input · {raw_reference:.3f}",
                    xy=(xs[-1], raw_reference), xytext=(0, 5),
                    textcoords="offset points", fontsize=8.5, color=_INK_2,
                    ha="right", va="bottom")

    std = [val(s, "standardize") for s in present]
    best_i = int(np.argmax([v if v is not None else -9 for v in std]))
    ax.annotate(f"best · {std[best_i]:.3f}", xy=(xs[best_i], std[best_i]),
                xytext=(0, 10), textcoords="offset points", fontsize=9,
                color=_INK, ha="center", fontweight="600")
    if "layer12" in present:
        j = present.index("layer12")
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


def plot_ladder_panels(by_n_comp: dict, output_path: str,
                        n_samples: int = None) -> str:
    """
    Small multiples, one column per component count: held-out R² on top,
    and underneath the number that actually decides a few-shot deployment --
    how often a draw beats predicting the mean at all.
    """
    comps = sorted(by_n_comp, key=lambda k: int(k))
    fig, axes = plt.subplots(2, len(comps), figsize=(4.1 * len(comps), 7.2),
                              sharex="col", sharey="row")
    if len(comps) == 1:
        axes = axes.reshape(2, 1)

    series_names = ("whitened raw input", "best embedding recipe",
                    "embedding, conventional tap")
    handles = None
    for c, n_comp_key in enumerate(comps):
        ladder = by_n_comp[n_comp_key]["ladder"]
        raw, emb_best, emb_conv = _series_for(by_n_comp, n_comp_key)
        top, bot = axes[0][c], axes[1][c]
        _style_axes(top)
        _style_axes(bot)

        for i, lab in enumerate([raw, emb_best, emb_conv]):
            if lab is None:
                continue
            color = _PALETTE[i]
            keys = _n_keys(ladder[lab])
            ns = [int(k) for k in keys]
            med = [ladder[lab][k]["r2_median"] for k in keys]
            # IQR is only meaningful where more than one draw was taken; the
            # full-pool rung is a single draw, so its band would be a lie.
            multi = [j for j, k in enumerate(keys) if ladder[lab][k]["n_draws"] > 1]
            top.plot(ns, med, marker="o", markersize=4.5, linewidth=1.8,
                     color=color, label=series_names[i], zorder=3)
            if multi:
                top.fill_between([ns[j] for j in multi],
                                  [ladder[lab][keys[j]]["r2_p25"] for j in multi],
                                  [ladder[lab][keys[j]]["r2_p75"] for j in multi],
                                  color=color, alpha=0.14, linewidth=0, zorder=1)
            # The full-pool rung has one draw (there is only one way to take
            # every row), so an IQR there would be fiction -- show the
            # bootstrap SD, which is the uncertainty that does apply.
            boot = ladder[lab][keys[-1]].get("r2_bootstrap_sd")
            top.errorbar(ns[-1], med[-1], yerr=boot, marker="o", markersize=6.5,
                         markerfacecolor=_SURFACE, markeredgecolor=color,
                         markeredgewidth=1.6, ecolor=color, elinewidth=1.2,
                         capsize=3, zorder=4, linestyle="none")

            fpos_x = [int(k) for k in keys if ladder[lab][k]["n_draws"] > 1]
            fpos_y = [ladder[lab][k]["frac_positive_r2"] for k in keys
                      if ladder[lab][k]["n_draws"] > 1]
            bot.plot(fpos_x, fpos_y, marker="o", markersize=4.5, linewidth=1.8,
                     color=color, zorder=3)

        top.axhline(0, color=_INK_2, linewidth=1.1, zorder=2)
        bot.axhline(0.5, color=_INK_2, linestyle="--", linewidth=1.1, zorder=2)
        top.set_xscale("log")
        bot.set_xscale("log")
        bot.set_ylim(-0.03, 1.05)
        label = f"{n_comp_key} component" + ("s" if int(n_comp_key) != 1 else "")
        top.set_title(label, fontsize=10.5, color=_INK, fontweight="600", pad=8)
        bot.set_xlabel("n_train  (labeled training samples)", fontsize=9.5,
                        color=_INK_2)
        if c == 0:
            top.set_ylabel("held-out R²  (median, IQR band over draws)",
                            fontsize=9.5, color=_INK_2)
            bot.set_ylabel("fraction of draws beating the mean",
                            fontsize=9.5, color=_INK_2)
            handles, _lbls = top.get_legend_handles_labels()

    if handles:
        leg = fig.legend(handles, series_names, fontsize=9, frameon=False,
                         loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.028))
        for t in leg.get_texts():
            t.set_color(_INK_2)
    pool = f"  ·  pool n={n_samples:,}" if n_samples else ""
    fig.suptitle("Label efficiency, by component count" + pool,
                 fontsize=12.5, color=_INK, fontweight="600", x=0.01, ha="left")
    fig.text(0.5, 0.004,
             "bands = IQR across repeated draws  ·  hollow final marker = full pool "
             "(single draw; error bar is the bootstrap SD)  ·  dashed line = 50% of draws",
             fontsize=8, color=_INK_MUTED, ha="center")
    fig.tight_layout(rect=[0, 0.075, 1, 0.97])
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
