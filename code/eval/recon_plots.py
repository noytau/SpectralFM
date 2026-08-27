"""
Dataset-level reconstruction figures for the 3AE eval.
No fairseq dependency (matplotlib + numpy + pandas only).

The pre-existing reconstruction figures (`report.py:_plot_signal_reconstruction`) are
sample-level: six overlay traces and a bar chart of their MSEs. They show what a
reconstruction looks like; they cannot show whether it works. This module adds the
aggregate view — per-dataset distributions, skill against trivial baselines, where along
the signal the error sits, whether amplitude survives, which spectra fail, and what
happens in the frequency domain — with single-component and multi-component data kept
separate throughout, never pooled into one number.

Every figure carries its own explanation. `_FIG_DOC` is the single source, feeding
(1) a footnote drawn onto the PNG, so the explanation travels with a figure read outside
the report, (2) the caption the report renders beneath the image, and (3) a FIGURES.md
written beside the PNGs. A figure without a `_FIG_DOC` entry raises rather than shipping
unexplained.
"""
from __future__ import annotations

import os
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from .evaluations.signal_reconstruction import (
    CROSS_HEAD_CAVEAT,
    PATHWAY_DETAIL,
    PATHWAY_LEGEND,
    PATHWAY_SHORT,
)
from .signal_features import FEATURE_LABELS

# Head colors match report.py's existing _PATHWAY_STYLE so the new figures read as the
# same family as the overlay panels.
_PATHWAY_STYLE = [
    ("fe",   "FE decoder",          "#1f77b4"),
    ("proj", "Projection decoder",  "#2ca02c"),
    ("tr",   "Transformer decoder", "#d62728"),
]
_HEAD_COLOR = {k: c for k, _, c in _PATHWAY_STYLE}
_HEAD_ORDER = [k for k, _, _ in _PATHWAY_STYLE]

# Dataset colors are independent of head colors: several figures use one visual channel
# for heads and another for datasets, so the two palettes must not collide in meaning.
_DATASET_COLORS = {
    "sanity":   "#7f7f7f",
    "in_dist":  "#17becf",
    "multi_ch": "#8c564b",
    "samples":  "#ff7f0e",
    "labeled":  "#9467bd",
}
_FALLBACK_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

# Line style carries the component group everywhere one axis mixes the two.
_GROUP_LS = {"single": "-", "multi": "--"}
_GROUP_LABEL = {
    "single": "single-component (one wav = one sample)",
    "multi":  "multi-component (sample = all wavs sharing a spec index)",
}

_REF_COLOR = "#b0b0b0"


# ── Figure documentation ──────────────────────────────────────────────────────

_FIG_DOC = {
    "recon_error_distribution": {
        "title": "Reconstruction error distribution per dataset",
        "what": (
            "Top row: the empirical cumulative distribution of per-sample reconstruction "
            "MSE, one panel per decoder head, one curve per dataset. The x-axis is MSE on "
            "a log scale; the y-axis is the fraction of samples with error at or below "
            "that value. Bottom row: the same distributions as violins of log10 MSE, with "
            "single-component datasets left of the divider and multi-component datasets "
            "right of it. The population is every sample drawn into the eval subset."
        ),
        "read": (
            "curves further LEFT are better; a steeply rising curve means error is "
            "consistent across samples, while a long flat right tail means most samples "
            "are fine and a few are catastrophic - which is exactly when the mean MSE "
            "stops describing the typical sample"
        ),
        "good": (
            "All curves far left and steep, with the multi-component curves close to the "
            "single-component ones. A large left-to-right gap between solid "
            "(single-component) and dashed (multi-component) curves is a generalization "
            "gap, not noise."
        ),
        "caveats": (
            "Medians, not means, are the headline statistic here: on the multi-component "
            "sets a handful of outliers pull the mean far above the typical sample. "
            + CROSS_HEAD_CAVEAT
        ),
    },
    "recon_summary_heatmap": {
        "title": "Reconstruction metric summary - datasets x metrics",
        "what": (
            "One block per decoder head. Rows are datasets (single-component block first, "
            "then multi-component); columns are the summary metrics. Cell text is the "
            "actual value; cell color is that value's rank within its own column, so "
            "colors compare datasets on one metric and mean nothing across columns."
        ),
        "read": (
            "read down a column to rank datasets on one metric; green is the better end of "
            "each column and red the worse, with the direction already accounted for "
            "(lower MSE is better, higher R-squared is better)"
        ),
        "good": (
            "mse_median small, r2_median near 1, frac_r2_positive at 1.0, pearson_median "
            "near 1, amp_ratio_median near 1. Watch for mse_mean sitting far above "
            "mse_median: that ratio is the outlier tax on this dataset."
        ),
        "caveats": (
            "amp_ratio_median below 1 means the reconstruction is systematically flatter "
            "than the target - see the amplitude-calibration figure. " + CROSS_HEAD_CAVEAT
        ),
    },
    "recon_skill_vs_baseline": {
        "title": "Reconstruction skill against a trivial baseline",
        "what": (
            "Left: median R-squared per head and dataset, where R2 = 1 - MSE / var(target). "
            "R2 = 0 is the score of the trivial predictor that outputs each sample's own "
            "mean value as a flat line; it is drawn as a solid reference line, not an axis "
            "origin. The percentage above each bar is the share of samples beating that "
            "baseline. Right: per-sample model MSE against the same baseline's MSE, "
            "log-log, with the break-even y = x line. When the checkpoint was trained "
            "with normalize=True the target is per-sample layer-normed, so var(target) "
            "is 1 for every sample and that axis would collapse to a vertical line; the "
            "panel then plots model MSE against peak prominence instead, with the "
            "baseline drawn as one horizontal break-even threshold. The panel title says "
            "which of the two it is."
        ),
        "read": (
            "bars must clear the R2 = 0 line to mean anything at all; on the right, points "
            "BELOW the diagonal are samples the model reconstructs better than a flat line "
            "at the sample mean, and points on or above it are samples where it adds nothing"
        ),
        "good": (
            "R-squared well above 0 with the beat-baseline share near 100%, and a point "
            "cloud clearly below the diagonal across the whole range of baseline difficulty."
        ),
        "caveats": (
            "This is the first figure to check. A decoder that has collapsed to emitting a "
            "smooth, mean-like envelope still posts a plausible-looking MSE, and only "
            "R-squared against the mean-predictor baseline exposes it. R-squared is "
            "undefined for constant targets - several multi_channel components are flat - "
            "and those samples are excluded from the median. " + CROSS_HEAD_CAVEAT
        ),
    },
    "recon_position_profile": {
        "title": "Where along the signal the error sits",
        "what": (
            "Top row: mean absolute reconstruction error at each of the 245 signal bins, "
            "one panel per dataset, one line per head, with the per-bin standard deviation "
            "of the target as a grey band for scale. Bottom row: the mean SIGNED error at "
            "each bin, so systematic over- or under-shoot is visible. Averaged over every "
            "sample in the eval subset."
        ),
        "read": (
            "a flat line means error is spread evenly along the signal; spikes at bin 0 and "
            "bin 244 are convolution edge artifacts; a repeating period-5 ripple points at "
            "the final (512, 5, 5) feature-extractor stage that compresses 245 bins to 47; "
            "in the bottom row any sustained departure from zero is a bias, not noise"
        ),
        "good": (
            "Flat, low, and - in the bottom row - hugging zero, with the error band well "
            "below the grey target-variability band."
        ),
        "caveats": (
            "Errors are averaged across samples, so a large per-bin value can come either "
            "from all samples erring a little or a few erring a lot; read this together "
            "with the error-distribution figure. " + CROSS_HEAD_CAVEAT
        ),
    },
    "recon_amplitude_calibration": {
        "title": "Amplitude calibration - is dynamic range preserved?",
        "what": (
            "Hexbin density of predicted value against target value over every "
            "(sample, bin) pair, one row per dataset and one column per head, with the "
            "y = x line of perfect calibration drawn. Color is point density on a log "
            "scale. The rightmost column plots each sample's prediction standard deviation "
            "against its target standard deviation, with y = x and y = 0.5x guides - or, "
            "when normalize=True has made every target's standard deviation 1, the "
            "distribution of the ratio std(prediction)/std(target) with a line at 1. The "
            "panel title says which."
        ),
        "read": (
            "a tight cloud along y = x is faithful reproduction; a cloud FLATTER than y = x, "
            "compressed vertically toward a horizontal band, means the decoder is hedging "
            "toward the mean and losing dynamic range; in the rightmost column, points "
            "below y = x quantify how much amplitude each sample loses"
        ),
        "good": (
            "A narrow diagonal cloud spanning the full target range, and rightmost-column "
            "points scattered tightly along y = x rather than along a shallower slope."
        ),
        "caveats": (
            "Dynamic-range collapse is the failure this figure exists to catch, and it is "
            "invisible in MSE - a flattened prediction can post a modest error while "
            "carrying almost no signal. Especially informative on multi-component data, "
            "where per-component target std spans more than a factor of ten. "
            + CROSS_HEAD_CAVEAT
        ),
    },
    "recon_spectral_fidelity": {
        "title": "Spectral fidelity - which frequencies survive reconstruction",
        "what": (
            "Top row: mean magnitude of the real FFT of the target (black) and of each "
            "head's reconstruction, one panel per dataset, log y-axis. Bottom row: the "
            "reconstruction-to-target magnitude ratio at each frequency, with the "
            "ratio = 1 line of perfect preservation. The zero-frequency (mean) bin is "
            "omitted throughout - normalize=True forces it to zero - and the top row's "
            "y-axis is clamped to the target's own range, so a head that collapsed to a "
            "constant output is annotated as being below the axis rather than being "
            "allowed to compress every other curve."
        ),
        "read": (
            "the left of each panel is the smooth envelope of the signal and the right is "
            "its fine structure, the narrow spectral lines; in the bottom row a ratio that "
            "falls below 1 as frequency increases means the decoder reproduces the envelope "
            "and smooths away the narrow features"
        ),
        "good": (
            "Reconstruction magnitude tracking the black target curve across the whole "
            "band, and a bottom-row ratio staying near 1 rather than rolling off."
        ),
        "caveats": (
            "This is the failure mode that matters most for spectral-line data and it does "
            "not show up in MSE: a low-pass reconstruction can score well on MSE while "
            "discarding exactly the features the signal is measured for. Magnitudes are "
            "averaged over samples, so a ratio near 1 does not guarantee the lines are in "
            "the right PLACE - read this with the position-profile figure. "
            + CROSS_HEAD_CAVEAT
        ),
    },
    "recon_error_vs_signal_properties": {
        "title": "Reconstruction error against signal properties",
        "what": (
            "Median reconstruction MSE (line) with the interquartile range (shaded) against "
            "binned properties of the target signal: contrast, number of local maxima, peak "
            "prominence, spectral centroid and peak position. Continuous properties are cut "
            "into quintiles; component index and components-per-spectrum are used as-is. "
            "For multi-component datasets a further row shows the per-spectrum view, where "
            "each spectrum's components are collapsed into their mean, worst and spread of "
            "error. One figure per dataset, because the available property axes differ "
            "between single- and multi-component data and are not forced into one grid."
        ),
        "read": (
            "a flat line means the model is indifferent to that property; an upward slope "
            "names the kind of spectrum it reconstructs badly; in the per-spectrum row a "
            "large gap between the mean and the worst curve means failure is concentrated "
            "in particular components rather than spread across whole spectra"
        ),
        "good": (
            "Flat, low lines with narrow interquartile bands across every axis - error that "
            "does not depend on what the spectrum looks like."
        ),
        "caveats": (
            "Bins with fewer than three samples are dropped, and axes that are constant "
            "within a dataset are omitted entirely rather than drawn flat - sampled_data, "
            "for instance, supplies all 28 components for every spectrum, so "
            "components-per-spectrum has no within-dataset variation there. Component index "
            "is not an ordinal difficulty scale; it behaves as an amplitude/shape class "
            "label. " + CROSS_HEAD_CAVEAT
        ),
    },
}


def _doc(key: str) -> dict:
    if key not in _FIG_DOC:
        raise KeyError(
            "recon_plots: figure %r has no _FIG_DOC entry. Every figure must ship with "
            "its explanation - add one before returning the figure." % key
        )
    return _FIG_DOC[key]


def _caption(key: str) -> str:
    d = _doc(key)
    return "%s - %s." % (d["title"], d["read"])


def _footnote(fig, key: str, extra: str = "", width: int = 150) -> None:
    """Draw the explanation onto the figure itself, below the axes."""
    d = _doc(key)
    parts = ["WHAT THIS SHOWS: " + d["what"],
             "HOW TO READ IT: " + d["read"][0].upper() + d["read"][1:] + ".",
             "A GOOD RESULT: " + d["good"],
             "CAVEATS: " + d["caveats"]]
    if extra:
        parts.append("THIS RUN: " + extra)
    body = "\n".join(textwrap.fill(p, width=width) for p in parts)
    fig.text(0.0, -0.015, body, ha="left", va="top", fontsize=6.5,
             color="#333333", family="monospace", linespacing=1.35)


def write_figure_docs(keys, out_dir: str, header: str = "") -> str:
    """Write FIGURES.md - the full explanation of every figure in this directory."""
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# Reconstruction figures - what each one measures", ""]
    if header:
        lines += [header, ""]
    lines += [
        "Each figure below also carries this text as a footnote on the image itself, so a "
        "PNG read outside this directory is still self-explanatory.", "",
        "### The three decoder heads", "",
        "A 3AE checkpoint holds one shared backbone and up to three decoder heads, each "
        "reading the backbone at a different depth and each reconstructing the same "
        "245-point input signal:", "",
    ]
    for k in _HEAD_ORDER:
        lines.append("- **%s** (`%s`) - %s" % (PATHWAY_SHORT[k], k, PATHWAY_LEGEND[k]))
    lines += ["", "> " + CROSS_HEAD_CAVEAT, "", "---", ""]

    for key in keys:
        d = _doc(key)
        lines += [
            "## " + d["title"], "", "`%s.png`" % key, "",
            "**What this shows.** " + d["what"], "",
            "**How to read it.** " + d["read"][0].upper() + d["read"][1:] + ".", "",
            "**A good result.** " + d["good"], "",
            "**Caveats.** " + d["caveats"], "", "---", "",
        ]
    path = os.path.join(out_dir, "FIGURES.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ── Small helpers ─────────────────────────────────────────────────────────────

def _save_fig(fig, path: str) -> str:
    """Same convention as report.py:_save_fig - kept local to avoid a circular import."""
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _dataset_color(alias: str, i: int) -> str:
    return _DATASET_COLORS.get(alias, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])


def _ds_label(r: dict, short: bool = False) -> str:
    """Dataset label: alias plus real subset name and n, so a panel is identifiable."""
    alias = r.get("dataset_alias") or "dataset"
    subset = r.get("dataset_subset") or ""
    n = r.get("n_samples") or len(r.get("results_df", []))
    if short:
        return "%s (n=%d)" % (alias, n)
    return "%s - %s, n=%d" % (alias, subset, n) if subset else "%s, n=%d" % (alias, n)


def _ckpt_tag(r: dict) -> str:
    meta = r.get("meta") or {}
    tag = meta.get("tag")
    if tag:
        return str(tag)
    for key in ("recon_ckpt", "fe_ckpt", "proj_ckpt", "tr_ckpt"):
        if r.get(key):
            return os.path.splitext(os.path.basename(str(r[key])))[0]
    return "unknown"


def _run_note(by_alias: dict) -> str:
    """One line describing the checkpoint and target convention behind these figures."""
    first = next(iter(by_alias.values()))
    norm = first.get("normalize")
    target_note = ("targets are per-sample zero-mean unit-std, matching training"
                   if norm else "targets are the raw signal, matching training")
    groups = {a: r.get("component_group", "single") for a, r in by_alias.items()}
    single = [a for a, g in groups.items() if g == "single"]
    multi = [a for a, g in groups.items() if g == "multi"]
    return ("checkpoint=%s; normalize=%s (%s); single-component datasets: %s; "
            "multi-component datasets: %s."
            % (_ckpt_tag(first), norm, target_note,
               ", ".join(single) or "none", ", ".join(multi) or "none"))


def _ordered_aliases(by_alias: dict) -> list:
    """Single-component datasets first, then multi-component - a stable reading order."""
    return sorted(by_alias,
                  key=lambda a: (by_alias[a].get("component_group", "single") == "multi", a))


def _group_divider(ax, aliases: list, by_alias: dict, positions) -> None:
    """Draw the single|multi boundary on a categorical axis and label both sides."""
    groups = [by_alias[a].get("component_group", "single") for a in aliases]
    if "single" not in groups or "multi" not in groups:
        return
    cut = groups.index("multi")
    x = (positions[cut - 1] + positions[cut]) / 2.0
    ax.axvline(x, color="#444444", lw=1.2, ls=":", zorder=0)
    # Above the axes, so these never fight with bars, annotations or violins.
    span = positions[-1] - positions[0]
    frac = (x - positions[0]) / span if span else 0.5
    ax.text(frac / 2, 1.005, "single-component", transform=ax.transAxes, fontsize=7,
            color="#444444", ha="center", va="bottom", style="italic")
    ax.text(frac + (1 - frac) / 2, 1.005, "multi-component", transform=ax.transAxes,
            fontsize=7, color="#444444", ha="center", va="bottom", style="italic")


def _head_title(k: str, width: int = 46) -> str:
    """Head name plus its tap, wrapped - the untruncated description overruns a panel."""
    return "%s\n%s" % (PATHWAY_SHORT[k], textwrap.fill(PATHWAY_DETAIL[k], width=width))


def _suptitle(fig, key: str, subtitle: str, width: int = 120, y: float = 1.0) -> None:
    fig.suptitle("%s\n%s" % (_doc(key)["title"], textwrap.fill(subtitle, width=width)),
                 fontsize=10.5, fontweight="bold", y=y)


def _error_axis(ax, values, axis: str = "x") -> None:
    """
    Label an error axis, choosing log or linear by the spread of the data.

    A log axis whose data spans less than a decade gets minor-tick labels from
    matplotlib and they collide into unreadable mush; worse, a decoder stuck at one
    output value spans no range at all. Linear is the honest choice there.
    """
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0], dtype=float)
    wide = v.size > 0 and (v.max() / v.min()) >= 10.0
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    label = ax.set_xlabel if axis == "x" else ax.set_ylabel
    setter("log" if wide else "linear")
    label("per-sample reconstruction MSE  (%s scale, \u2193 better)"
          % ("log" if wide else "linear"), fontsize=8)


def _heads_of(by_alias: dict) -> list:
    heads = set()
    for r in by_alias.values():
        heads.update(r.get("pathways") or [])
    return [k for k in _HEAD_ORDER if k in heads]


def _usable(by_alias: dict) -> dict:
    """Drop skipped / empty results so a partial run still plots what it has."""
    return {a: r for a, r in by_alias.items()
            if isinstance(r, dict) and not r.get("skipped")
            and isinstance(r.get("results_df"), pd.DataFrame) and len(r["results_df"])}


# ── F1: error distribution (ECDF + violins) ───────────────────────────────────

def _plot_error_distribution(by_alias: dict, out_dir: str) -> list:
    aliases = _ordered_aliases(by_alias)
    heads = _heads_of(by_alias)
    if not aliases or not heads:
        return []

    fig, axes = plt.subplots(2, len(heads), figsize=(5.6 * len(heads), 8.4), squeeze=False)

    for c, k in enumerate(heads):
        ax = axes[0][c]
        all_mse = np.concatenate(
            [by_alias[a]["results_df"][f"{k}_mse"].to_numpy()
             for a in aliases if f"{k}_mse" in by_alias[a]["results_df"].columns]
            or [np.array([])])
        for i, a in enumerate(aliases):
            r = by_alias[a]
            if f"{k}_mse" not in r["results_df"].columns:
                continue
            v = np.sort(r["results_df"][f"{k}_mse"].to_numpy())
            v = v[v > 0]
            if not len(v):
                continue
            group = r.get("component_group", "single")
            ax.plot(v, np.arange(1, len(v) + 1) / len(v),
                    color=_dataset_color(a, i), ls=_GROUP_LS[group], lw=1.6,
                    label="%s  [%s]" % (_ds_label(r, short=True), group))
            ax.axvline(np.median(v), color=_dataset_color(a, i), ls=":", lw=0.8, alpha=0.6)
        _error_axis(ax, all_mse)
        ax.set_ylabel("fraction of samples at or below", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(_head_title(k), fontsize=8, color=_HEAD_COLOR[k], fontweight="bold")
        ax.legend(fontsize=6.5, loc="lower right", title="dotted vline = median",
                  title_fontsize=6)

        ax = axes[1][c]
        data, labels, colors = [], [], []
        for i, a in enumerate(aliases):
            r = by_alias[a]
            if f"{k}_mse" not in r["results_df"].columns:
                continue
            v = r["results_df"][f"{k}_mse"].to_numpy()
            v = v[v > 0]
            if len(v) < 5:
                continue
            data.append(np.log10(v))
            labels.append(a)
            colors.append(_dataset_color(a, i))
        if data:
            pos = np.arange(1, len(data) + 1)
            parts = ax.violinplot(data, positions=pos, showmedians=True, widths=0.8)
            for body, col in zip(parts["bodies"], colors):
                body.set_facecolor(col)
                body.set_alpha(0.55)
            for piece in ("cbars", "cmins", "cmaxes", "cmedians"):
                if piece in parts:
                    parts[piece].set_color("#333333")
                    parts[piece].set_linewidth(1.0)
            ax.set_xticks(pos)
            ax.set_xticklabels(labels, fontsize=7.5)
            ax.set_ylabel("per-sample MSE, log\u2081\u2080 scale  (\u2193 better)", fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_title("distribution shape per dataset - %s" % PATHWAY_SHORT[k],
                         fontsize=8.5, color=_HEAD_COLOR[k])
            _group_divider(ax, labels, by_alias, pos)
            # A decoder locked to one output value produces no spread at all. Say so
            # rather than leaving the reader to decode a 1e-9 axis.
            spread = max(np.ptp(np.concatenate(data)), 0.0)
            if spread < 1e-6:
                ax.text(0.5, 0.5,
                        "distribution is degenerate:\nevery sample has essentially\n"
                        "the same error (spread < 1e-6 in log\u2081\u2080)",
                        transform=ax.transAxes, ha="center", va="center", fontsize=8,
                        color="#b00000", fontweight="bold",
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#b00000"))

    _suptitle(fig, "recon_error_distribution", _run_note(by_alias))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _footnote(fig, "recon_error_distribution")
    path = _save_fig(fig, os.path.join(out_dir, "recon_error_distribution.png"))
    return [(_caption("recon_error_distribution"), path)]


# ── F1b: cross-dataset summary heatmap ────────────────────────────────────────

# (metric column, display label, higher_is_better)
_SUMMARY_METRICS = [
    ("mse_median",       "MSE\nmedian",        False),
    ("mse_mean",         "MSE\nmean",          False),
    ("mse_p90",          "MSE\np90",           False),
    ("mae_median",       "MAE\nmedian",        False),
    ("r2_median",        "R²\nmedian",     True),
    ("frac_r2_positive", "frac beating\nbaseline", True),
    ("pearson_median",   "Pearson r\nmedian",  True),
    ("amp_ratio_median", "amplitude ratio\nmedian (1 = ideal)", None),
]


def _plot_summary_heatmap(by_alias: dict, out_dir: str) -> list:
    aliases = _ordered_aliases(by_alias)
    heads = _heads_of(by_alias)
    if not aliases or not heads:
        return []

    fig, axes = plt.subplots(1, len(heads),
                             figsize=(1.35 * len(_SUMMARY_METRICS) * len(heads),
                                      1.05 * len(aliases) + 3.0),
                             squeeze=False)
    for c, k in enumerate(heads):
        ax = axes[0][c]
        rows, row_labels = [], []
        for a in aliases:
            sdf = by_alias[a].get("summary_df")
            if not isinstance(sdf, pd.DataFrame) or k not in set(sdf["head"]):
                continue
            s = sdf.set_index("head").loc[k]
            rows.append([float(s[m]) for m, _, _ in _SUMMARY_METRICS])
            row_labels.append("%s  [%s]" % (a, by_alias[a].get("component_group", "single")))
        if not rows:
            ax.axis("off")
            continue
        M = np.array(rows, dtype=float)

        # Color by within-column rank so wildly different scales stay comparable; the
        # cell TEXT carries the real value.
        shade = np.full_like(M, 0.5)
        for j, (_, _, higher_better) in enumerate(_SUMMARY_METRICS):
            col = M[:, j]
            ok = np.isfinite(col)
            if ok.sum() < 2 or np.ptp(col[ok]) == 0:
                continue
            norm = (col - np.nanmin(col[ok])) / np.ptp(col[ok])
            if higher_better is None:            # amplitude ratio: distance from 1
                norm = 1.0 - np.abs(col - 1.0) / max(np.nanmax(np.abs(col[ok] - 1.0)), 1e-9)
            elif not higher_better:
                norm = 1.0 - norm
            shade[:, j] = norm
        ax.imshow(shade, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                txt = "-" if not np.isfinite(v) else (
                    "%.3f" % v if abs(v) < 100 and abs(v) >= 1e-3 else "%.2e" % v)
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5)
        ax.set_xticks(range(len(_SUMMARY_METRICS)))
        ax.set_xticklabels([lbl for _, lbl, _ in _SUMMARY_METRICS], fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(_head_title(k, width=40), fontsize=8, color=_HEAD_COLOR[k],
                     fontweight="bold")
        # Divider between the single- and multi-component row blocks.
        groups = [by_alias[a].get("component_group", "single") for a in aliases
                  if isinstance(by_alias[a].get("summary_df"), pd.DataFrame)]
        if "single" in groups and "multi" in groups:
            ax.axhline(groups.index("multi") - 0.5, color="#222222", lw=2.0)

    _suptitle(fig, "recon_summary_heatmap", _run_note(by_alias), y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _footnote(fig, "recon_summary_heatmap")
    path = _save_fig(fig, os.path.join(out_dir, "recon_summary_heatmap.png"))
    return [(_caption("recon_summary_heatmap"), path)]


# ── F2: skill against the mean-predictor baseline ─────────────────────────────

def _plot_skill_vs_baseline(by_alias: dict, out_dir: str) -> list:
    aliases = _ordered_aliases(by_alias)
    heads = _heads_of(by_alias)
    if not aliases or not heads:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(7.0 + 1.4 * len(aliases), 5.6),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ── Left: median R2 bars, grouped by dataset, one bar per head ────────────
    ax = axes[0]
    x = np.arange(len(aliases), dtype=float)
    width = 0.8 / max(len(heads), 1)
    bar_labels = []
    for j, k in enumerate(heads):
        vals, fracs, xs = [], [], []
        for i, a in enumerate(aliases):
            sdf = by_alias[a].get("summary_df")
            if not isinstance(sdf, pd.DataFrame) or k not in set(sdf["head"]):
                continue
            s = sdf.set_index("head").loc[k]
            vals.append(float(s["r2_median"]))
            fracs.append(float(s["frac_r2_positive"]))
            xs.append(x[i] + (j - (len(heads) - 1) / 2) * width)
        if not xs:
            continue
        ax.bar(xs, vals, width, color=_HEAD_COLOR[k], alpha=0.9,
               label=PATHWAY_LEGEND[k])
        bar_labels.extend(zip(xs, vals, fracs))
    ax.axhline(0.0, color="#000000", lw=1.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_ds_label(by_alias[a], short=True) for a in aliases],
                       fontsize=7.5, rotation=12, ha="right")
    ax.set_ylabel("median R\u00b2 = 1 \u2212 MSE / var(target)   (\u2191 better)",
                  fontsize=8.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Skill per head and dataset\n"
                 "(% above each bar = share of samples beating the baseline)",
                 fontsize=9)
    # Headroom first, so the annotations below land inside the axes.
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -0.05), hi + 0.12 * (hi - lo))
    lo, hi = ax.get_ylim()
    pad = 0.02 * (hi - lo)
    for xi, v, fr in bar_labels:
        if not np.isfinite(v):
            continue
        ax.text(xi, max(v, 0.0) + pad,
                "%.0f%%" % (100 * fr) if np.isfinite(fr) else "-",
                ha="center", va="bottom", fontsize=6.5)
    handles, labels_ = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#000000", lw=1.8))
    labels_.append("R\u00b2 = 0 \u2014 the trivial baseline: predicting each sample's "
                   "own mean value as a flat line")
    ax.legend(handles, labels_, fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    _group_divider(ax, aliases, by_alias, x)

    # ── Right: every sample, model error against the baseline it must beat ────
    #
    # The natural x-axis is the baseline's own MSE, which for the mean predictor IS the
    # target's variance. But when the checkpoint was trained with normalize=True the
    # target is per-sample layer-normed, so var(target) == 1 for EVERY sample and that
    # axis collapses to a single vertical line. In that case the baseline becomes one
    # horizontal threshold and the x-axis is better spent on a signal property that
    # actually varies, so the panel still answers "which samples does the model beat,
    # and are they the simple ones?".
    ax = axes[1]
    base_all = np.concatenate(
        [by_alias[a]["results_df"]["contrast"].to_numpy(dtype=float) ** 2
         for a in aliases if "contrast" in by_alias[a]["results_df"].columns]
        or [np.array([])])
    finite = base_all[np.isfinite(base_all) & (base_all > 0)]
    constant_baseline = finite.size > 0 and (
        np.percentile(finite, 99) / np.percentile(finite, 1)) < 2.0

    x_col, x_label = "peak_prominence", FEATURE_LABELS["peak_prominence"]
    plotted = False
    for a in aliases:
        r = by_alias[a]
        rdf = r["results_df"]
        if "contrast" not in rdf.columns:
            continue
        base = rdf["contrast"].to_numpy(dtype=float) ** 2
        xs = (rdf[x_col].to_numpy(dtype=float) if constant_baseline and x_col in rdf.columns
              else base)
        marker = "o" if r.get("component_group") == "single" else "^"
        for k in heads:
            if f"{k}_mse" not in rdf.columns:
                continue
            model = rdf[f"{k}_mse"].to_numpy(dtype=float)
            ok = np.isfinite(xs) & (model > 0) & (base > 0)
            if constant_baseline:
                ok &= xs > 0
            if ok.sum() < 5:
                continue
            ax.scatter(xs[ok], model[ok], s=5, alpha=0.3, color=_HEAD_COLOR[k],
                       marker=marker, linewidths=0, rasterized=True)
            plotted = True

    if plotted:
        ax.set_yscale("log")
        ax.set_ylabel("model MSE   (\u2193 better; below the line = model helps)",
                      fontsize=8.5)
        handles = [plt.Line2D([], [], color=_HEAD_COLOR[k], marker="o", ls="",
                              label=PATHWAY_SHORT[k]) for k in heads]
        handles += [plt.Line2D([], [], color="#555555", marker="o", ls="",
                               label="circle = single-component sample"),
                    plt.Line2D([], [], color="#555555", marker="^", ls="",
                               label="triangle = multi-component sample")]
        if constant_baseline:
            threshold = float(np.median(finite))
            ax.axhline(threshold, color="k", ls="--", lw=1.4)
            ax.set_xscale("log")
            ax.set_xlabel("%s\n(x-axis is a signal property, not the baseline: see title)"
                          % x_label, fontsize=8)
            handles.append(plt.Line2D([], [], color="k", ls="--",
                                      label="baseline MSE = %.3g (break-even)" % threshold))
            ax.set_title("Every sample: model error vs the break-even threshold\n"
                         "normalize=True makes var(target) = %.3g for every sample, so the "
                         "baseline is one line" % threshold, fontsize=8.5)
        else:
            lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
            hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.3)
            ax.set_xscale("log")
            ax.set_xlabel("baseline MSE = var(target)  \u2014 harder samples to the right",
                          fontsize=8.5)
            handles.append(plt.Line2D([], [], color="k", ls="--", label="y = x (break-even)"))
            ax.set_title("Every sample: model error vs the baseline's error", fontsize=9)
        ax.legend(handles=handles, fontsize=6.5, loc="best")
        ax.grid(True, alpha=0.3, which="both")
    else:
        ax.axis("off")

    _suptitle(fig, "recon_skill_vs_baseline", _run_note(by_alias), y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _footnote(fig, "recon_skill_vs_baseline")
    path = _save_fig(fig, os.path.join(out_dir, "recon_skill_vs_baseline.png"))
    return [(_caption("recon_skill_vs_baseline"), path)]


# ── F3: per-position error profile ────────────────────────────────────────────

def _plot_position_profile(by_alias: dict, out_dir: str) -> list:
    aliases = [a for a in _ordered_aliases(by_alias) if by_alias[a].get("profiles")]
    heads = _heads_of(by_alias)
    if not aliases or not heads:
        return []

    fig, axes = plt.subplots(2, len(aliases), figsize=(4.9 * len(aliases), 7.2),
                             squeeze=False, sharex=True)
    for c, a in enumerate(aliases):
        r = by_alias[a]
        prof = r["profiles"]
        group = r.get("component_group", "single")
        bins = np.arange(len(prof["per_position_target_std"]))

        ax = axes[0][c]
        ax.fill_between(bins, 0, prof["per_position_target_std"],
                        color=_REF_COLOR, alpha=0.45,
                        label="target variability (per-bin std) — scale reference")
        for k in heads:
            key = f"per_position_abs_err_{k}"
            if key in prof:
                ax.plot(bins, prof[key], color=_HEAD_COLOR[k], lw=1.3,
                        label="%s — mean |error|" % PATHWAY_SHORT[k])
        ax.set_ylabel("mean |error| at this bin  (↓ better)", fontsize=8)
        ax.set_title("%s\n[%s]" % (_ds_label(r), _GROUP_LABEL[group]), fontsize=8.5,
                     color="#444444")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, bins[-1])
        if c == 0:
            ax.legend(fontsize=6.5, loc="upper center")

        ax = axes[1][c]
        ax.axhline(0.0, color="#000000", lw=1.0)
        for k in heads:
            key = f"per_position_signed_err_{k}"
            if key in prof:
                ax.plot(bins, prof[key], color=_HEAD_COLOR[k], lw=1.3,
                        label="%s — mean signed error" % PATHWAY_SHORT[k])
        ax.set_ylabel("mean signed error  (0 = unbiased)", fontsize=8)
        ax.set_xlabel("signal bin index (0–%d)" % bins[-1], fontsize=8.5)
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.legend(fontsize=6.5, loc="upper center")

    _suptitle(fig, "recon_position_profile", _run_note(by_alias))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _footnote(fig, "recon_position_profile")
    path = _save_fig(fig, os.path.join(out_dir, "recon_position_profile.png"))
    return [(_caption("recon_position_profile"), path)]


# ── F4: amplitude calibration ─────────────────────────────────────────────────

def _plot_amplitude_calibration(by_alias: dict, out_dir: str) -> list:
    aliases = [a for a in _ordered_aliases(by_alias) if by_alias[a].get("_arrays")]
    heads = _heads_of(by_alias)
    if not aliases or not heads:
        return []

    n_cols = len(heads) + 1
    fig, axes = plt.subplots(len(aliases), n_cols,
                             figsize=(4.0 * n_cols, 3.7 * len(aliases)), squeeze=False)
    for rowi, a in enumerate(aliases):
        r = by_alias[a]
        t = r["_arrays"]["target"]
        preds = r["_arrays"]["preds"]
        group = r.get("component_group", "single")

        for c, k in enumerate(heads):
            ax = axes[rowi][c]
            if k not in preds:
                ax.axis("off")
                continue
            p = preds[k]
            tf, pf = t.ravel(), p.ravel()
            lim_lo = float(min(tf.min(), np.percentile(pf, 0.1)))
            lim_hi = float(max(tf.max(), np.percentile(pf, 99.9)))
            hb = ax.hexbin(tf, pf, gridsize=60, bins="log", cmap="viridis",
                           extent=(lim_lo, lim_hi, lim_lo, lim_hi), mincnt=1)
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="#ff4d4d", lw=1.4,
                    ls="--", label="y = x (perfect calibration)")
            # Least-squares slope through the cloud quantifies the compression.
            slope = float(np.polyfit(tf, pf, 1)[0])
            ax.plot([lim_lo, lim_hi], [slope * lim_lo, slope * lim_hi],
                    color="#ffffff", lw=1.2, label="best-fit slope = %.2f" % slope)
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.set_xlabel("target value at a bin", fontsize=8)
            if c == 0:
                ax.set_ylabel("%s\n\npredicted value at a bin" % _ds_label(r, short=True),
                              fontsize=8)
            ax.set_title("%s — %s" % (PATHWAY_SHORT[k], group), fontsize=8.5,
                         color=_HEAD_COLOR[k], fontweight="bold")
            ax.legend(fontsize=6, loc="upper left", framealpha=0.75)
            fig.colorbar(hb, ax=ax, label="point density (log)", fraction=0.046)

        # Rightmost column: per-sample dynamic range.
        #
        # Normally a prediction-std vs target-std scatter. But normalize=True layer-norms
        # every target to unit variance, so target std is 1 for every sample and the
        # scatter degenerates into a single vertical strip. The ratio std(pred)/std(target)
        # carries exactly the same information and stays readable, so plot its
        # distribution instead when that happens.
        ax = axes[rowi][n_cols - 1]
        ts = t.std(axis=1)
        degenerate = ts.size > 0 and (ts.max() - ts.min()) < 1e-4 * max(ts.max(), 1e-9)

        if degenerate:
            for k in heads:
                if k not in preds:
                    continue
                ratio = preds[k].std(axis=1) / np.where(ts > 0, ts, np.nan)
                ratio = ratio[np.isfinite(ratio)]
                if not ratio.size:
                    continue
                ax.hist(ratio, bins=40, color=_HEAD_COLOR[k], alpha=0.55,
                        label="%s (median %.2f)" % (PATHWAY_SHORT[k], np.median(ratio)))
            ax.axvline(1.0, color="k", ls="--", lw=1.4,
                       label="ratio = 1 (range preserved)")
            ax.axvline(0.5, color="#888888", ls=":", lw=1.0,
                       label="ratio = 0.5 (half the range lost)")
            ax.set_xlabel("amplitude ratio = std(prediction) / std(target)", fontsize=8)
            ax.set_ylabel("number of samples", fontsize=8)
            ax.set_title("Per-sample dynamic range — %s\n(target std is 1 for every "
                         "sample under normalize=True,\nso the ratio is plotted directly)"
                         % _ds_label(r, short=True), fontsize=8)
        else:
            hi = float(ts.max() * 1.15) if ts.size else 1.0
            for k in heads:
                if k not in preds:
                    continue
                ax.scatter(ts, preds[k].std(axis=1), s=6, alpha=0.4, linewidths=0,
                           color=_HEAD_COLOR[k], label=PATHWAY_SHORT[k], rasterized=True)
            ax.plot([0, hi], [0, hi], "k--", lw=1.2, label="y = x (range preserved)")
            ax.plot([0, hi], [0, 0.5 * hi], color="#888888", lw=1.0, ls=":",
                    label="y = 0.5x (half the range lost)")
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_xlabel("target std. dev. of this sample", fontsize=8)
            ax.set_ylabel("prediction std. dev.", fontsize=8)
            ax.set_title("Per-sample dynamic range — %s" % _ds_label(r, short=True),
                         fontsize=8.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="upper left")

    _suptitle(fig, "recon_amplitude_calibration", _run_note(by_alias))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _footnote(fig, "recon_amplitude_calibration")
    path = _save_fig(fig, os.path.join(out_dir, "recon_amplitude_calibration.png"))
    return [(_caption("recon_amplitude_calibration"), path)]


# ── F6: spectral fidelity ─────────────────────────────────────────────────────

def _plot_spectral_fidelity(by_alias: dict, out_dir: str) -> list:
    aliases = [a for a in _ordered_aliases(by_alias) if by_alias[a].get("profiles")]
    heads = _heads_of(by_alias)
    if not aliases or not heads:
        return []

    fig, axes = plt.subplots(2, len(aliases), figsize=(4.9 * len(aliases), 7.2),
                             squeeze=False, sharex=True)
    for c, a in enumerate(aliases):
        r = by_alias[a]
        prof = r["profiles"]
        group = r.get("component_group", "single")
        tgt_full = prof["mean_fft_target"]
        freq_full = prof.get("freq_bins",
                             np.arange(len(tgt_full)) / (2.0 * (len(tgt_full) - 1)))
        # Skip the zero-frequency (mean) bin: normalize=True forces it to zero, which on a
        # log axis reads as a spurious cliff. The mean level is covered by the signed-error
        # row of the position-profile figure instead.
        tgt, freq = tgt_full[1:], freq_full[1:]

        ax = axes[0][c]
        ax.plot(freq, tgt, color="black", lw=1.8, label="target (ground truth)")
        for k in heads:
            key = f"mean_fft_pred_{k}"
            if key in prof:
                ax.plot(freq, prof[key][1:], color=_HEAD_COLOR[k], lw=1.2,
                        label="%s reconstruction" % PATHWAY_SHORT[k])
        ax.set_yscale("log")
        # Clamp to the target's own range. A head that collapsed to a constant output has
        # an FFT magnitude near zero, and letting it set the axis compresses every
        # informative curve into a sliver at the top.
        finite_tgt = tgt[np.isfinite(tgt) & (tgt > 0)]
        if finite_tgt.size:
            ax.set_ylim(finite_tgt.min() / 100.0, finite_tgt.max() * 10.0)
            clipped = [PATHWAY_SHORT[k] for k in heads
                       if f"mean_fft_pred_{k}" in prof
                       and np.nanmax(prof[f"mean_fft_pred_{k}"][1:]) < finite_tgt.min() / 100.0]
            if clipped:
                ax.text(0.98, 0.03,
                        "below axis: %s\n(magnitude ~0 - constant output)"
                        % ", ".join(clipped),
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                        color="#b00000",
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#b00000"))
        ax.set_ylabel("mean |FFT| magnitude (log)", fontsize=8)
        ax.set_title("%s\n[%s]\nsmooth envelope ← | → narrow spectral lines"
                     % (_ds_label(r), _GROUP_LABEL[group]), fontsize=8.5, color="#444444")
        ax.grid(True, alpha=0.3, which="both")
        if c == 0:
            ax.legend(fontsize=6.5, loc="lower left")

        ax = axes[1][c]
        ax.axhline(1.0, color="black", lw=1.4, label="ratio = 1 (magnitude preserved)")
        ratios = []
        for k in heads:
            key = f"mean_fft_pred_{k}"
            if key in prof:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(tgt > 0, prof[key][1:] / tgt, np.nan)
                ratios.append(ratio)
                ax.plot(freq, ratio, color=_HEAD_COLOR[k], lw=1.2,
                        label="%s / target" % PATHWAY_SHORT[k])
        # Headroom for ratios above 1 - additive noise genuinely raises high-frequency
        # magnitude, and a hard 1.6 cap would silently hide it - but bounded so one
        # runaway curve cannot flatten the rest.
        top = 1.6
        if ratios:
            stacked = np.concatenate(ratios)
            stacked = stacked[np.isfinite(stacked)]
            if stacked.size:
                top = float(min(5.0, max(1.6, np.percentile(stacked, 99) * 1.1)))
        ax.set_ylim(0, top)
        ax.set_ylabel("reconstruction / target magnitude", fontsize=8)
        ax.set_xlabel("frequency (cycles per bin)\n"
                      "(the zero-frequency / mean bin is omitted)", fontsize=8)
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.legend(fontsize=6.5, loc="lower left")

    _suptitle(fig, "recon_spectral_fidelity", _run_note(by_alias))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _footnote(fig, "recon_spectral_fidelity")
    path = _save_fig(fig, os.path.join(out_dir, "recon_spectral_fidelity.png"))
    return [(_caption("recon_spectral_fidelity"), path)]


# ── F5: error vs signal properties (one figure per dataset) ───────────────────

_AXIS_LABELS = dict(FEATURE_LABELS)
_AXIS_LABELS.update({
    "comp": "Component index within the spectrum "
            "(an amplitude/shape class label, not an ordinal difficulty scale)",
    "n_comps": "Components per spectrum (whole dataset, after removing "
               "the duplicate comp20≡comp14 / comp21≡comp15)",
    "n_comps_in_split": "Components of this spectrum present in the eval subset",
})


def _plot_error_vs_properties(r: dict, out_dir: str) -> list:
    strat = r.get("strat_df")
    if not isinstance(strat, pd.DataFrame) or strat.empty:
        return []
    heads = [k for k in _HEAD_ORDER if k in (r.get("pathways") or [])]
    if not heads:
        return []

    axes_present = [ax for ax in strat["axis"].unique()]
    spec_df = r.get("spectrum_df")
    has_spec = isinstance(spec_df, pd.DataFrame) and not spec_df.empty
    group = r.get("component_group", "single")

    n_cols = min(4, max(1, len(axes_present)))
    n_rows = int(np.ceil(len(axes_present) / n_cols)) + (1 if has_spec else 0)
    fig, grid = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for i, axis in enumerate(axes_present):
        ax = grid[i // n_cols][i % n_cols]
        sub = strat[strat["axis"] == axis].sort_values("bin")
        x = np.arange(len(sub))
        for k in heads:
            col = f"{k}_mse_median"
            if col not in sub.columns:
                continue
            ax.plot(x, sub[col], marker="o", ms=4, lw=1.5, color=_HEAD_COLOR[k],
                    label=PATHWAY_SHORT[k])
            ax.fill_between(x, sub[f"{k}_mse_q25"], sub[f"{k}_mse_q75"],
                            color=_HEAD_COLOR[k], alpha=0.16)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["bin_label"], fontsize=6.5, rotation=25, ha="right")
        ax.set_xlabel(_AXIS_LABELS.get(axis, axis), fontsize=7.5, wrap=True)
        ax.set_ylabel("median MSE  (shaded = IQR, ↓ better)", fontsize=7.5)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ns = ", ".join("n=%d" % n for n in sub["n"])
        ax.set_title("%s\n(%s)" % (axis, ns), fontsize=8, color="#444444")
        if i == 0:
            ax.legend(fontsize=6.5)

    # Blank out unused cells in the stratifier block.
    for j in range(len(axes_present), n_cols * (n_rows - (1 if has_spec else 0))):
        grid[j // n_cols][j % n_cols].axis("off")

    # Per-spectrum row (multi-component only): mean vs worst vs spread by n_comps.
    if has_spec:
        row = n_rows - 1
        views = [("mse_mean", "mean over the spectrum's components", "-", "o"),
                 ("mse_max", "worst single component", "--", "s"),
                 ("mse_spread", "spread (worst − best)", ":", "^")]
        for ci in range(n_cols):
            grid[row][ci].axis("off")

        ax = grid[row][0]
        ax.axis("on")
        by_n = spec_df.groupby("n_comps_present")
        xs = sorted(by_n.groups)
        # Mean vs worst component, per head. The spread has its own panel to the right;
        # plotting it here too would stretch the log axis to a scale where neither reads.
        for k in heads:
            for suffix, label, ls, marker in views[:2]:
                col = f"{k}_{suffix}"
                if col not in spec_df.columns:
                    continue
                ax.plot(xs, [by_n.get_group(n)[col].median() for n in xs],
                        ls=ls, marker=marker, ms=4, lw=1.4, color=_HEAD_COLOR[k],
                        label="%s — %s" % (PATHWAY_SHORT[k], label))
        ax.set_xlabel("components of the spectrum present in this eval subset", fontsize=7.5)
        ax.set_ylabel("median per-spectrum MSE  (↓ better)", fontsize=7.5)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title("Per-spectrum view (L2): is failure spectrum-wide or "
                     "component-specific?\n%d spectra  (solid = mean over the spectrum's "
                     "components, dashed = worst single component)" % len(spec_df),
                     fontsize=8)
        ax.legend(fontsize=5.5, ncol=1)

        if n_cols > 1:
            ax = grid[row][1]
            ax.axis("on")
            for k in heads:
                col = f"{k}_mse_spread"
                if col not in spec_df.columns:
                    continue
                v = spec_df[col].to_numpy()
                v = v[np.isfinite(v) & (v > 0)]
                if len(v) < 5:
                    continue
                sv = np.sort(v)
                ax.plot(sv, np.arange(1, len(sv) + 1) / len(sv),
                        color=_HEAD_COLOR[k], lw=1.5, label=PATHWAY_SHORT[k])
            ax.set_xscale("log")
            ax.set_xlabel("within-spectrum MSE spread (worst − best component)", fontsize=7.5)
            ax.set_ylabel("fraction of spectra at or below", fontsize=7.5)
            ax.grid(True, alpha=0.3, which="both")
            ax.set_title("How unevenly error is distributed inside one spectrum\n"
                         "(curve far left = all components reconstruct alike)", fontsize=8)
            ax.legend(fontsize=6.5)

    d = _doc("recon_error_vs_signal_properties")
    extra = ("%s, %s. Axes shown: %s.%s"
             % (_ds_label(r), _GROUP_LABEL[group], ", ".join(axes_present),
                "" if has_spec else
                " No per-spectrum row: this is single-component data."))
    fig.suptitle("%s\n%s — checkpoint=%s, normalize=%s"
                 % (d["title"], _ds_label(r), _ckpt_tag(r), r.get("normalize")),
                 fontsize=10.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _footnote(fig, "recon_error_vs_signal_properties", extra=extra)
    path = _save_fig(fig, os.path.join(out_dir, "recon_error_vs_signal_properties.png"))
    return [(_caption("recon_error_vs_signal_properties"), path)]


# ── Public entry points ───────────────────────────────────────────────────────

# Cross-dataset figures, in the order a reader should meet them: how big is the error,
# then how it compares to doing nothing, then where it lives, then what it destroys.
_SUMMARY_FIGURES = [
    ("recon_error_distribution",     _plot_error_distribution),
    ("recon_summary_heatmap",        _plot_summary_heatmap),
    ("recon_skill_vs_baseline",      _plot_skill_vs_baseline),
    ("recon_position_profile",       _plot_position_profile),
    ("recon_amplitude_calibration",  _plot_amplitude_calibration),
    ("recon_spectral_fidelity",      _plot_spectral_fidelity),
]

DATASET_LEVEL_FIGURE_KEYS = ["recon_error_vs_signal_properties"]
SUMMARY_FIGURE_KEYS = [k for k, _ in _SUMMARY_FIGURES]


def plot_recon_dataset_level(r: dict, out_dir: str) -> list:
    """
    Per-dataset dataset-level figures, appended after the existing sample-level ones.

    Called once per `signal_reconstruction_<alias>` result. Currently the stratification
    figure, whose shape depends on whether the dataset is single- or multi-component —
    which is why it is per-dataset rather than pooled.
    """
    if not isinstance(r, dict) or r.get("skipped"):
        return []
    # No FIGURES.md here: these figures are written flat and then relocated by the
    # report, so a doc file written now would be stranded at the run root. The summary
    # pass writes one covering every figure instead.
    return _plot_error_vs_properties(r, out_dir)


def plot_recon_across_datasets(by_alias: dict, out_dir: str) -> list:
    """
    Cross-dataset reconstruction figures — the story the per-dataset panels cannot tell.

    by_alias: {dataset_alias: signal_reconstruction result dict}. Skipped or empty
    results are dropped, so a partial run still plots whatever completed.
    """
    usable = _usable(by_alias)
    if not usable:
        return []
    os.makedirs(out_dir, exist_ok=True)

    figures, keys = [], []
    for key, builder in _SUMMARY_FIGURES:
        try:
            figs = builder(usable, out_dir)
        except Exception as e:                      # one bad figure must not lose the rest
            print("[ReconPlots] %s failed: %s: %s" % (key, type(e).__name__, e))
            continue
        if figs:
            figures += figs
            keys.append(key)

    if keys:
        # Document the per-dataset figures here too - they have no doc file of their own.
        write_figure_docs(keys + DATASET_LEVEL_FIGURE_KEYS, out_dir,
                          header=_run_note(usable))
    return figures


def summary_frame(by_alias: dict) -> pd.DataFrame:
    """
    One tidy table of every head x dataset summary row, for CSV export and the report.
    Single-component datasets first, with `component_group` as an explicit column so the
    two populations are never silently averaged together downstream.
    """
    usable = _usable(by_alias)
    frames = []
    for a in _ordered_aliases(usable):
        r = usable[a]
        sdf = r.get("summary_df")
        if not isinstance(sdf, pd.DataFrame) or sdf.empty:
            continue
        f = sdf.copy()
        f.insert(0, "dataset", a)
        f.insert(1, "subset", r.get("dataset_subset", ""))
        f.insert(2, "component_group", r.get("component_group", "single"))
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Self-test ─────────────────────────────────────────────────────────────────

def _synthetic_results(seed: int = 0) -> dict:
    """
    Build signal_reconstruction-shaped results with known ground truth, so the metrics
    and every figure can be checked without a checkpoint or a GPU.

    Four datasets mirroring the real matrix: two single-component, two multi-component.
    The heads are given deliberately different pathologies so a figure that fails to
    distinguish them is visibly broken:
      fe    a faithful reconstruction plus modest noise
      proj  amplitude compressed to 50% — should show up in amp_ratio and in F4's slope
      tr    each sample's own mean, i.e. the trivial baseline — R2 must land at 0
    """
    from .evaluations.signal_reconstruction import (
        _per_sample_metrics, _profiles, _spectrum_table, _stratified_table,
        _summary_table)
    from .signal_features import STRATIFIER_ORDER, compute_signal_features

    rng = np.random.default_rng(seed)
    specs = [("sanity", "single_channel_1k", 120, False),
             ("in_dist", "single_channel_10k", 200, False),
             ("multi_ch", "multi_channel", 200, True),
             ("samples", "sampled_data", 160, True)]

    out = {}
    for alias, subset, n, multi in specs:
        L = 245
        base = np.cumsum(rng.normal(0, 1, (n, L)), axis=1)
        base += 3.0 * np.exp(-0.5 * ((np.arange(L) - rng.integers(40, 200, (n, 1))) / 6.0) ** 2)
        t = ((base - base.mean(1, keepdims=True)) / base.std(1, keepdims=True)).astype(np.float32)
        # Multi-component data is harder here, mirroring the real generalization gap.
        noise = 0.5 if multi else 0.25
        preds = {
            "fe":   (t + rng.normal(0, noise, t.shape)).astype(np.float32),
            "proj": (0.5 * t + rng.normal(0, noise, t.shape)).astype(np.float32),
            "tr":   np.repeat(t.mean(1, keepdims=True), L, axis=1).astype(np.float32),
        }
        heads = list(preds)

        fnames = ([f"dataset0002_comp{i % 6}_spec_{i // 6}.wav" for i in range(n)] if multi
                  else [f"spectra0000_batch0_spec_{i}.wav" for i in range(n)])
        rows = {"index": np.arange(n), "filename": fnames}
        r = {"skipped": False, "normalize": True, "pathways": heads,
             "n_samples": n, "dataset_alias": alias, "dataset_subset": subset,
             "meta": {"tag": "synthetic_selftest"},
             "component_group": "multi" if multi else "single"}
        for k in heads:
            mse = ((preds[k].astype(np.float64) - t) ** 2).mean(1)
            rows[f"{k}_mse"] = mse
            rows[f"{k}_mae"] = np.abs(preds[k].astype(np.float64) - t).mean(1)
            for name, v in _per_sample_metrics(t, preds[k], mse).items():
                rows[f"{k}_{name}"] = v
        rdf = pd.concat([pd.DataFrame(rows), compute_signal_features(t)], axis=1)
        if multi:
            rdf["dataset_id"] = 2
            rdf["comp"] = [i % 6 for i in range(n)]
            rdf["spec"] = [i // 6 for i in range(n)]
            rdf["n_comps"] = 6
            rdf["n_comps_in_split"] = 6
            rdf["component_group"] = "multi"
        else:
            rdf["component_group"] = "single"

        axes = list(STRATIFIER_ORDER) + (["comp", "n_comps"] if multi else [])
        r["results_df"] = rdf
        r["summary_df"] = _summary_table(rdf, heads)
        r["strat_df"] = _stratified_table(rdf, heads, axes)
        if multi:
            r["spectrum_df"] = _spectrum_table(rdf, heads)
        r["profiles"] = _profiles(t, preds)
        r["_arrays"] = {"target": t, "preds": preds}
        out[alias] = r
    return out


def _selftest(out_dir: str) -> int:
    """Render every figure from synthetic data and assert the known metric signatures."""
    by_alias = _synthetic_results()
    os.makedirs(out_dir, exist_ok=True)

    failures = []
    s = by_alias["in_dist"]["summary_df"].set_index("head")
    checks = [
        ("tr is the mean-predictor baseline, so R2 must be ~0",
         abs(s.loc["tr", "r2_median"]) < 0.02),
        ("tr predicts a constant, so amplitude ratio must be ~0",
         abs(s.loc["tr", "amp_ratio_median"]) < 0.02),
        ("tr must beat the baseline on ~no samples",
         s.loc["tr", "frac_r2_positive"] < 0.05),
        ("proj is amplitude-compressed to 50%, ratio must be ~0.5",
         abs(s.loc["proj", "amp_ratio_median"] - 0.5) < 0.08),
        ("fe is a faithful reconstruction, R2 must be high",
         s.loc["fe", "r2_median"] > 0.8),
        ("fe must beat the baseline on essentially every sample",
         s.loc["fe", "frac_r2_positive"] > 0.99),
        ("multi-component data is noisier here, so its MSE must be larger",
         by_alias["multi_ch"]["summary_df"].set_index("head").loc["fe", "mse_median"]
         > s.loc["fe", "mse_median"]),
    ]
    for msg, ok in checks:
        print("  %s  %s" % ("PASS" if ok else "FAIL", msg))
        if not ok:
            failures.append(msg)

    figs = plot_recon_across_datasets(by_alias, out_dir)
    for alias, r in by_alias.items():
        d = os.path.join(out_dir, "per_dataset_%s" % alias)
        os.makedirs(d, exist_ok=True)
        figs += plot_recon_dataset_level(r, d)

    expected = len(SUMMARY_FIGURE_KEYS) + len(by_alias) * len(DATASET_LEVEL_FIGURE_KEYS)
    print("\n  rendered %d/%d figures into %s" % (len(figs), expected, out_dir))
    for caption, path in figs:
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        print("    %7.0f KB  %s" % (size / 1024, os.path.relpath(path, out_dir)))
        if size < 5000:
            failures.append("figure looks empty: %s" % path)
    if len(figs) != expected:
        failures.append("expected %d figures, got %d" % (expected, len(figs)))

    print("\n%s" % ("ALL CHECKS PASSED" if not failures
                    else "FAILURES:\n  - " + "\n  - ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "recon_plots_selftest"
    raise SystemExit(_selftest(target))
