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
    recipes = [sel[k]["recipe"] for k in keys]
    return ns, med, lo, hi, draws, recipes


def _panel_entry(panel: dict, arm: str, recipe: str):
    return panel.get(f"{arm} | {recipe}")


def plot_crossover_panel(by_n_comp: dict, output_path: str,
                          n_eval_report: int = None, n_pool: int = None) -> str:
    """
    Two rows per component count: absolute R² on top (so a "crossing" can be
    read against how good either side actually is, not just which is ahead),
    the paired gap with its crossing below. Both curves are the HONEST
    per-rung best recipe (selected on a split disjoint from the one scored),
    plus the fully naive raw baseline (no whitening, no z-scoring) as a
    neutral reference line.
    """
    comps = sorted(by_n_comp, key=lambda k: int(k))
    fig, axes = plt.subplots(2, len(comps), figsize=(4.3 * len(comps), 6.6),
                             sharex="col")
    if len(comps) == 1:
        axes = axes.reshape(2, 1)

    raw_color, emb_color = _PALETTE[0], _PALETTE[1]
    handles = None
    for c, key in enumerate(comps):
        d = by_n_comp[key]
        panel, sel = d["panel"], d["selected"]
        top, bot = axes[0][c], axes[1][c]
        _style_axes(top)
        _style_axes(bot)

        ns_r, med_r, lo_r, hi_r, draws_r, _ = _selected_series(sel["raw"])
        ns_e, med_e, lo_e, hi_e, draws_e, _ = _selected_series(sel["embedding"])
        none = _panel_entry(panel, "raw", "none + ridgecv")

        # top: absolute R²
        for ns, med, lo, hi, draws, color, label in [
                (ns_r, med_r, lo_r, hi_r, draws_r, raw_color, "raw (best per rung)"),
                (ns_e, med_e, lo_e, hi_e, draws_e, emb_color, "embedding (best per rung)")]:
            top.plot(ns, med, marker="o", markersize=4.5, linewidth=1.9,
                     color=color, label=label, zorder=3)
            multi = [i for i, dd in enumerate(draws) if dd > 1]
            if multi:
                top.fill_between([ns[i] for i in multi], [lo[i] for i in multi],
                                 [hi[i] for i in multi], color=color, alpha=0.13,
                                 linewidth=0, zorder=1)
            solo = [i for i, dd in enumerate(draws) if dd <= 1]
            if solo:
                top.plot([ns[i] for i in solo], [med[i] for i in solo], marker="o",
                         markersize=6.5, markerfacecolor=_SURFACE,
                         markeredgecolor=color, markeredgewidth=1.6, zorder=4,
                         linestyle="none")

        if none is not None:
            keys = _n_keys(none["report"])
            ns_n = [int(k) for k in keys]
            med_n = [none["report"][k]["r2_median"] for k in keys]
            top.plot(ns_n, med_n, marker="none", linewidth=1.4, linestyle=(0, (4, 2)),
                     color=_NONE_COLOR, label="raw, no normalizer (naive)", zorder=2)

        # value labels at the endpoints so the absolute level is readable, not
        # just the shape of the curve. Fixed pixel offsets (not value-scaled)
        # so two close endpoints (e.g. 0.867 vs 0.884) never overlap: whichever
        # is larger goes above, the other below, by a constant gap.
        top_first = med_r[-1] >= med_e[-1]
        r_dy, e_dy = (16, -16) if top_first else (-16, 16)
        top.annotate(f"{med_r[-1]:.3f}", xy=(ns_r[-1], med_r[-1]), xytext=(-4, r_dy),
                    textcoords="offset points", fontsize=7.8, color=raw_color,
                    fontweight="600", ha="right",
                    va="bottom" if r_dy > 0 else "top")
        top.annotate(f"{med_e[-1]:.3f}", xy=(ns_e[-1], med_e[-1]), xytext=(-4, e_dy),
                    textcoords="offset points", fontsize=7.8, color=emb_color,
                    fontweight="600", ha="right",
                    va="bottom" if e_dy > 0 else "top")
        top.axhline(0, color=_INK_2, linewidth=1.0, zorder=2)
        top.set_xscale("log")
        xlo, xhi = top.get_xlim()
        top.set_xlim(xlo, xhi * 1.5)  # headroom for the endpoint labels
        top.set_title(f"{key} component" + ("s" if key != "1" else ""),
                      fontsize=10.5, color=_INK, fontweight="600", pad=8)
        if c == 0:
            top.set_ylabel("held-out R²\n(best recipe per rung)", fontsize=9,
                           color=_INK_2)
            handles, labels_ = top.get_legend_handles_labels()

        # bottom: paired gap (both arms scored on the SAME draws at each
        # rung -- run_panel fixes the draw seed per n_train, independent of
        # which recipe wins -- so differencing draw-by-draw is valid here too)
        gaps_med, gaps_lo, gaps_hi = [], [], []
        for k in _n_keys(sel["raw"]):
            dr = sel["raw"][k].get("r2_draws") or [sel["raw"][k]["r2_median"]]
            de = sel["embedding"][k].get("r2_draws") or [sel["embedding"][k]["r2_median"]]
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
            bot.axvline(cross, color=_PALETTE[2], linestyle="--", linewidth=1.1,
                       alpha=0.7, zorder=1)
            # actual R2 pair at the nearest measured rung to the crossing, so
            # the reader can see whether the crossing happens somewhere
            # meaningful (both sides doing well) or in the noise floor. Pinned
            # to the top of the AXES (not the data y=0 line), via a blended
            # transform, so it can never collide with the legend below the
            # figure regardless of where the gap curve happens to sit.
            j = int(np.argmin([abs(np.log10(x) - np.log10(cross)) for x in ns_r]))
            trans = matplotlib.transforms.blended_transform_factory(
                bot.transData, bot.transAxes)
            bot.annotate(f"n≈{cross:,.0f}\nraw {med_r[j]:.2f} · emb {med_e[j]:.2f}",
                        xy=(cross, 0.96), xycoords=trans, fontsize=7.5, color=_INK,
                        ha="center", va="top", linespacing=1.3)
        bot.set_xscale("log")
        bot.set_xlabel("n_train", fontsize=9, color=_INK_2)
        if c == 0:
            bot.set_ylabel("Δ R²\n(embedding − raw)", fontsize=9, color=_INK_2)

    if handles:
        leg = fig.legend(handles, labels_, fontsize=8.7, frameon=False,
                         loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.035))
        for t in leg.get_texts():
            t.set_color(_INK_2)
    note = "top: absolute R² (median, IQR band; hollow marker = single-draw rung)"
    if n_eval_report:
        note += (f"  ·  scored on a {n_eval_report}-row held-out split, "
                 "disjoint from the recipe-selection split")
    if n_pool:
        note += f"  ·  training pool n={n_pool:,}"
    fig.text(0.01, -0.025, note, fontsize=7.6, color=_INK_MUTED, ha="left")
    fig.suptitle("Honest per-rung crossover: is the embedding ever ahead, and by how much?",
                 fontsize=12.5, color=_INK, fontweight="600", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0.13, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight", dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    return output_path
