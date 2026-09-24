"""
The honest crossover figure (absolute R², not just the gap), drawn from
`panel.py`'s per-rung recipe panel (`recipe_panel.json`).

Kept separate from plots.py, which draws from the single-fixed-recipe study
(`label_probe_results.json`) -- these two files answer different questions
and must not be merged silently.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np

from .plots import (_PALETTE, _INK, _INK_2, _INK_MUTED, _SURFACE,
                    _style_axes, _n_keys, _crossing_n)

_NONE_COLOR = "#9a9a96"  # the naive baseline: deliberately neutral, not a competing series


def _selected_series(sel: dict):
    keys = _n_keys(sel)
    ns = [int(k) for k in keys]
    med = [sel[k]["r2_median"] for k in keys]
    lo = [sel[k]["r2_p25"] for k in keys]
    hi = [sel[k]["r2_p75"] for k in keys]
    draws = [len(sel[k].get("r2_draws") or []) for k in keys]
    return ns, med, lo, hi, draws


def _layer_envelope_draws(layer_curves: dict, k: str):
    """Per-draw max across every named layer curve at rung k -- the best
    available layer on that specific draw, not just by median. Layers share
    draw indices with the raw curve (same seed, same pool per rung), so
    pairing this against raw's own draws is valid."""
    per_layer = [c[k].get("r2_draws") or [c[k]["r2_median"]] for c in layer_curves.values()]
    n = max(len(d) for d in per_layer)
    out = []
    for i in range(n):
        vals = [d[i] for d in per_layer if i < len(d) and d[i] is not None]
        out.append(max(vals) if vals else None)
    return out


def plot_crossover_panel(by_n_comp: dict, output_path: str,
                          n_eval_report: int = None, n_pool: int = None,
                          raw_full_pool: dict = None) -> str:
    """
    Two rows per component count: absolute R² on top (so a "crossing" can be
    read against how good either side actually is, not just which is ahead),
    the paired gap with its crossing below. Raw is the HONEST per-rung best
    recipe (selected on a split disjoint from the one scored); each embedding
    layer is one fixed recipe (see panel.LAYER_RECIPE) -- there is no
    per-layer recipe search, and no pooling-squeezed recipe here at all (see
    panel.py's module docstring). The gap curve pairs raw against whichever
    named layer wins on each individual draw.

    `raw_full_pool`, when given, is {n_comp: (r2_mean, r2_bootstrap_sd,
    normalizer_label)} -- the best-of-(whitened, z-scored) raw score from
    the full-pool CV diagnostic (label_probe_results.json), drawn as a
    second, dotted reference line. The panel's own raw curve is already
    honest (it searches every normalizer, not a fixed one), but at small n
    its selection+report split can be a handful of rows, so its own
    endpoint is noisier than the properly cross-validated full-pool number
    -- this anchors "how good can raw really get" against that split noise,
    so a small gap here never reads as "the embedding is competitive" when
    a much larger, reliably-estimated one is sitting right next to it.
    """
    comps = sorted(by_n_comp, key=lambda k: int(k))
    fig, axes = plt.subplots(2, len(comps), figsize=(4.3 * len(comps), 6.6),
                             sharex="col")
    if len(comps) == 1:
        axes = axes.reshape(2, 1)

    raw_color = _PALETTE[0]
    layer_colors = _PALETTE[1:]
    handles = None
    for c, key in enumerate(comps):
        d = by_n_comp[key]
        panel, sel, layer_curves = d["panel"], d["selected"], d["layer_curves"]
        top, bot = axes[0][c], axes[1][c]
        _style_axes(top)
        _style_axes(bot)

        ns_r, med_r, lo_r, hi_r, draws_r = _selected_series(sel)
        layer_series = {name: _selected_series(curve) for name, curve in layer_curves.items()}
        none = panel.get("none + ridgecv")

        # top: absolute R² -- raw (honest per-rung best) plus one line per
        # named layer (single fixed recipe each).
        top.plot(ns_r, med_r, marker="o", markersize=4.5, linewidth=1.9,
                 color=raw_color, label="raw (best recipe per label budget)", zorder=3)
        multi = [i for i, dd in enumerate(draws_r) if dd > 1]
        if multi:
            top.fill_between([ns_r[i] for i in multi], [lo_r[i] for i in multi],
                             [hi_r[i] for i in multi], color=raw_color, alpha=0.13,
                             linewidth=0, zorder=1)
        solo = [i for i, dd in enumerate(draws_r) if dd <= 1]
        if solo:
            top.plot([ns_r[i] for i in solo], [med_r[i] for i in solo], marker="o",
                     markersize=6.5, markerfacecolor=_SURFACE,
                     markeredgecolor=raw_color, markeredgewidth=1.6, zorder=4,
                     linestyle="none")

        for (name, (ns_l, med_l, lo_l, hi_l, draws_l)), color in zip(layer_series.items(),
                                                                      layer_colors):
            top.plot(ns_l, med_l, marker="o", markersize=4, linewidth=1.6,
                     color=color, label=name, zorder=3, alpha=0.9)
            multi_l = [i for i, dd in enumerate(draws_l) if dd > 1]
            if multi_l:
                top.fill_between([ns_l[i] for i in multi_l], [lo_l[i] for i in multi_l],
                                 [hi_l[i] for i in multi_l], color=color, alpha=0.10,
                                 linewidth=0, zorder=1)

        if none is not None:
            keys = _n_keys(none["report"])
            ns_n = [int(k) for k in keys]
            med_n = [none["report"][k]["r2_median"] for k in keys]
            top.plot(ns_n, med_n, marker="none", linewidth=1.4, linestyle=(0, (4, 2)),
                     color=_NONE_COLOR, label="raw, no normalizer (naive)", zorder=2)

        ref = (raw_full_pool or {}).get(key)
        if ref:
            ref_r2, ref_sd, ref_label = ref
            if ref_sd:
                top.axhspan(ref_r2 - ref_sd, ref_r2 + ref_sd, color=raw_color,
                            alpha=0.06, zorder=0, linewidth=0)
            top.axhline(ref_r2, color=raw_color, linestyle=":", linewidth=1.2,
                       alpha=0.6, zorder=2)
            sd_txt = f" ±{ref_sd:.3f}" if ref_sd else ""
            top.annotate(f"full-pool CV, best-of-normalizer ({ref_label}): "
                        f"{ref_r2:.3f}{sd_txt}", xy=(ns_r[0], ref_r2), xytext=(2, 4),
                        textcoords="offset points", fontsize=7.2, color=raw_color,
                        ha="left", va="bottom", alpha=0.85)

        # endpoint labels: raw, and whichever layer tops the pack at the full
        # pool -- the other layers stay legend-identified without adding more
        # text to a chart that already has four series.
        best_name = max(layer_series, key=lambda name: layer_series[name][1][-1])
        best_color = layer_colors[list(layer_series).index(best_name) % len(layer_colors)]
        best_med_last = layer_series[best_name][1][-1]
        top_first = med_r[-1] >= best_med_last
        r_dy, e_dy = (16, -16) if top_first else (-16, 16)
        top.annotate(f"{med_r[-1]:.3f}", xy=(ns_r[-1], med_r[-1]), xytext=(-4, r_dy),
                    textcoords="offset points", fontsize=7.8, color=raw_color,
                    fontweight="600", ha="right",
                    va="bottom" if r_dy > 0 else "top")
        top.annotate(f"{best_med_last:.3f}", xy=(ns_r[-1], best_med_last), xytext=(-4, e_dy),
                    textcoords="offset points", fontsize=7.8, color=best_color,
                    fontweight="600", ha="right",
                    va="bottom" if e_dy > 0 else "top")
        top.axhline(0, color=_INK_2, linewidth=1.0, zorder=2)
        top.set_xscale("log")
        xlo, xhi = top.get_xlim()
        top.set_xlim(xlo, xhi * 1.5)  # headroom for the endpoint labels
        top.set_title(f"{key} component" + ("s" if key != "1" else ""),
                      fontsize=10.5, color=_INK, fontweight="600", pad=8)
        if c == 0:
            top.set_ylabel("held-out R²", fontsize=9, color=_INK_2)
            handles, labels_ = top.get_legend_handles_labels()

        # bottom: paired gap. Every layer shares draw indices with raw (same
        # seed, same pool per rung) -- pairing the per-draw ENVELOPE across
        # layers against raw's own draws asks "is any named layer ever ahead
        # on this specific draw", not just by median.
        gaps_med, gaps_lo, gaps_hi = [], [], []
        for k in _n_keys(sel):
            dr = sel[k].get("r2_draws") or [sel[k]["r2_median"]]
            de = _layer_envelope_draws(layer_curves, k)
            pairs = [(a, b) for a, b in zip(de, dr) if a is not None and b is not None]
            g = np.array([a - b for a, b in pairs]) if pairs else np.array([0.0])
            gaps_med.append(float(np.median(g)))
            gaps_lo.append(float(np.percentile(g, 25)) if len(g) > 1 else float(g[0]))
            gaps_hi.append(float(np.percentile(g, 75)) if len(g) > 1 else float(g[0]))

        bot.plot(ns_r, gaps_med, marker="o", markersize=4.5, linewidth=1.9,
                 color=_INK, zorder=3)
        bot.fill_between(ns_r, gaps_lo, gaps_hi, color=_INK, alpha=0.12,
                         linewidth=0, zorder=1)
        bot.axhline(0, color=_INK_2, linewidth=1.2, zorder=2)
        cross = _crossing_n(ns_r, gaps_med)
        if cross is not None:
            bot.axvline(cross, color=_INK_2, linestyle="--", linewidth=1.1,
                       alpha=0.7, zorder=1)
            # actual R2 pair at the nearest measured rung to the crossing,
            # naming whichever layer is ahead there, so the reader can see
            # whether the crossing happens somewhere meaningful (both sides
            # doing well) or in the noise floor. Pinned to the top of the
            # AXES (not the data y=0 line), via a blended transform, so it
            # can never collide with the legend below the figure.
            j = int(np.argmin([abs(np.log10(x) - np.log10(cross)) for x in ns_r]))
            j_key = str(ns_r[j])
            top_layer_j = max(layer_curves, key=lambda name: layer_curves[name][j_key]["r2_median"])
            top_med_j = layer_curves[top_layer_j][j_key]["r2_median"]
            trans = matplotlib.transforms.blended_transform_factory(
                bot.transData, bot.transAxes)
            bot.annotate(f"n≈{cross:,.0f}\nraw {med_r[j]:.2f} · {top_layer_j} {top_med_j:.2f}",
                        xy=(cross, 0.96), xycoords=trans, fontsize=7.5, color=_INK,
                        ha="center", va="top", linespacing=1.3)
        bot.set_xscale("log")
        bot.set_xlabel("n_train", fontsize=9, color=_INK_2)
        if c == 0:
            bot.set_ylabel("Δ R²\n(best layer − raw)", fontsize=9, color=_INK_2)

    if handles:
        leg = fig.legend(handles, labels_, fontsize=8.7, frameon=False,
                         loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035))
        for t in leg.get_texts():
            t.set_color(_INK_2)
    note = "top: absolute R² (median, IQR band; hollow marker = only one training draw at that label budget)"
    if n_eval_report:
        note += (f"  ·  scored on a {n_eval_report}-row held-out split, "
                 "disjoint from the recipe-selection split")
    if n_pool:
        note += f"  ·  training pool n={n_pool:,}"
    fig.text(0.01, -0.025, note, fontsize=7.6, color=_INK_MUTED, ha="left")
    fig.suptitle("Raw input vs. named embedding layers, across the label budget",
                 fontsize=12.5, color=_INK, fontweight="600", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.13, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return output_path
