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
from . import recon_analysis
from .recon_analysis import LIFT_AXIS_LABELS
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

# Heads are also distinguished by line style, marker and hatch, not only by hue. That
# keeps every figure readable in greyscale - on an e-ink reader, or printed - and for a
# reader who cannot separate the three hues.
_HEAD_LS = {"fe": "-", "proj": "--", "tr": ":"}
_HEAD_MARKER = {"fe": "o", "proj": "s", "tr": "^"}
_HEAD_HATCH = {"fe": "", "proj": "///", "tr": "xxx"}

# Greyscale palettes for the e-ink style. Values are spaced far enough apart to survive
# the limited contrast range of an e-paper panel; nothing lighter than #b0b0b0 is used
# for a line, since it disappears.
_EINK_HEAD_COLOR = {"fe": "#000000", "proj": "#5a5a5a", "tr": "#9a9a9a"}
# Only five greys are usable on e-paper, and the dataset palette reuses them, so two
# datasets in one panel can land on the same shade. Where datasets are the lines rather
# than the panels, they get a line style each as well.
_EINK_DATASET_LS = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
_EINK_DATASET_COLOR = {"sanity": "#8a8a8a", "in_dist": "#000000",
                       "multi_ch": "#000000", "samples": "#8a8a8a",
                       "labeled": "#5a5a5a"}

# A summary-heatmap column has to vary by this fraction of its own magnitude before it is
# shaded at full strength. Half means a column whose best and worst differ by half the
# column's typical value is coloured hard, while one differing by a per cent or two is
# barely tinted - the honest reading of a small difference.
_FULL_SATURATION_AT = 0.5

_STYLE = "screen"


def set_style(name: str) -> None:
    """
    Switch the figure style. 'screen' is the default colour styling; 'eink' swaps in a
    greyscale palette, larger type and heavier lines, and drops the explanation footnote
    from the image - the PDF carries that as real text, which is far easier to read on an
    e-paper panel than 6.5pt type baked into a bitmap.
    """
    global _STYLE
    if name not in ("screen", "eink"):
        raise ValueError("unknown style %r (expected 'screen' or 'eink')" % name)
    _STYLE = name
    if name == "eink":
        _HEAD_COLOR.update(_EINK_HEAD_COLOR)
        _DATASET_COLORS.update(_EINK_DATASET_COLOR)
        plt.rcParams.update({
            "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
            "figure.titlesize": 13, "lines.linewidth": 1.9,
            "axes.linewidth": 1.0, "grid.alpha": 0.45, "savefig.facecolor": "white",
            "image.cmap": "Greys",
        })
    else:
        _HEAD_COLOR.update({k: c for k, _, c in _PATHWAY_STYLE})
        _DATASET_COLORS.update(_SCREEN_DATASET_COLOR)
        plt.rcParams.update(plt.rcParamsDefault)
        matplotlib.use("Agg")


def is_eink() -> bool:
    return _STYLE == "eink"


def _head_text_color(k: str) -> str:
    """
    Colour for a panel title naming a head.

    The greyscale head palette runs to #9a9a9a, which is fine for a line against white
    but too light to read as text. Titles go black in the e-ink style; the head is already
    named in the title, so the colour was only ever a convenience.
    """
    return "#000000" if _STYLE == "eink" else _HEAD_COLOR[k]


def _rank_cmap():
    """
    Colormap for the rank-shaded summary heatmap.

    RdYlGn is the wrong choice on e-paper twice over: there is no colour, and red and
    green sit at almost the same luminance, so the two ends of the scale become the same
    grey. The e-ink version runs white (better) to mid grey (worse) instead - a single
    luminance ramp, and light enough throughout that the black cell values stay readable.
    """
    if _STYLE != "eink":
        return "RdYlGn"
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("eink_rank", ["#9a9a9a", "#ffffff"])


def _density_cmap():
    """Sequential map for the hexbin density; viridis is not luminance-ordered in grey."""
    return "Greys" if _STYLE == "eink" else "viridis"

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
_SCREEN_DATASET_COLOR = dict(_DATASET_COLORS)

# Line style carries the component group everywhere one axis mixes the two.
_GROUP_LS = {"single": "-", "multi": "--"}
# Datasets share an axis with each other in the ECDF panels, so they need a mark too.
_DATASET_MARKERS = ["o", "s", "^", "D", "v"]
_GROUP_LABEL = {
    "single": "single-component (one wav = one sample)",
    "multi":  "multi-component (sample = all wavs sharing a spec index)",
}
# Same distinction where a panel title has no room for the parenthetical.
_GROUP_SHORT = {"single": "single-component", "multi": "multi-component"}

# A component carrying this many times its share of the dataset's error is called out by
# name on the component figure. 2x is well clear of sampling noise on a component with
# tens of samples, and far below the 13x sampled_data's comp 26 and 29 actually post.
_COMP_LIFT_MARK = 2.0

_REF_COLOR = "#b0b0b0"


# ── Figure documentation ──────────────────────────────────────────────────────

_FIG_DOC = {
    "recon_overlay": {
        "title": "Reconstruction overlay - individual samples",
        "caption": (
            "Individual reconstructions against the target (black), one column per "
            "decoder head. The y-axis is fixed to the target's range per row, so a "
            "prediction drawn flat at the edge is off-scale, not zero."
        ),
        "what": (
            "Six samples drawn evenly across the eval subset, one row each. The target is "
            "black; each column overlays one decoder head's reconstruction on it. The "
            "per-sample MSE is printed in each panel title. Sample choice is "
            "deterministic - evenly spaced by index, not best or worst - so it is a fair "
            "look rather than a flattering one."
        ),
        "read": (
            "look for whether the reconstruction follows the peaks and the fine structure "
            "or only the broad envelope; a prediction that hugs the middle of the panel "
            "and ignores the peaks is the mean-like failure the skill figure quantifies"
        ),
        "good": (
            "The colored line sits on the black one, including at the narrow peaks, on "
            "every row rather than just the easy ones."
        ),
        "caveats": (
            "Six samples cannot tell you how often any of this happens - that is what the "
            "dataset-level figures are for. The y-axis is clamped to the target's range "
            "per row, so a prediction that leaves the panel is off-scale rather than zero. "
            + CROSS_HEAD_CAVEAT
        ),
    },
    "recon_mse_bars": {
        "title": "Per-sample error for the overlay samples",
        "caption": (
            "Error for the six traces above, log scale - how much they differ from each "
            "other, and from the dataset mean quoted in the legend."
        ),
        "what": (
            "Per-sample reconstruction MSE for exactly the six samples shown in the "
            "overlay above, one bar group per sample and one bar per decoder head, on a "
            "log scale. The legend quotes each head's mean MSE over the whole subset."
        ),
        "read": (
            "compare bar heights within a sample to rank the heads on it, and compare a "
            "bar against the mean in the legend to see whether that sample is typical or "
            "an outlier"
        ),
        "good": (
            "Low bars that sit near the legend's mean, rather than a few towering ones - "
            "though with only six samples this is an impression, not a measurement."
        ),
        "caveats": (
            "Six samples, chosen by index rather than by difficulty. Read the error "
            "distribution figure for the real spread. " + CROSS_HEAD_CAVEAT
        ),
    },
    "recon_error_distribution": {
        "title": "Reconstruction error distribution per dataset",
        "caption": (
            "How per-sample error is spread within each dataset. Further left is better; "
            "a long flat tail means a few samples are far worse than the median."
        ),
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
        "caption": (
            "Every dataset scored on every metric, one block per decoder head. Shading "
            "marks rank within a column, better end lighter in greyscale and green in "
            "colour; read down a column, not across."
        ),
        "what": (
            "One block per decoder head. Rows are datasets (single-component block first, "
            "then multi-component); columns are the summary metrics. Cell text is the "
            "actual value; cell shading is that value's rank within its own column, so "
            "shading compares datasets on one metric and means nothing across columns. "
            "Shading strength tracks how much the column actually varies - a column whose "
            "values differ by half their own magnitude or more is shaded at full strength, "
            "one varying by a per cent or two is left almost neutral - so a difference in "
            "the fourth decimal cannot look like a real gap."
        ),
        "read": (
            "read down a column to rank datasets on one metric; the shading marks rank "
            "within that column, with the direction already accounted for (lower MSE is "
            "better, higher R-squared is better) - in colour green is the better end and "
            "red the worse, in greyscale lighter is better and darker worse; a pale or "
            "neutral column is one whose values are all close together"
        ),
        "good": (
            "mse_median small, r2_median near 1, frac_r2_positive at 1.0, pearson_median "
            "near 1, amp_ratio_median near 1. Watch for mse_mean sitting far above "
            "mse_median: that ratio is the outlier tax on this dataset. A column left "
            "unshaded is one whose values are all equal - see the caveats."
        ),
        "caveats": (
            "amp_ratio_median below 1 means the reconstruction is systematically flatter "
            "than the target - see the amplitude-calibration figure. " + CROSS_HEAD_CAVEAT
        ),
    },
    "recon_skill_vs_baseline": {
        "title": "Reconstruction skill against a trivial baseline",
        "caption": (
            "Does the model beat a flat line at each sample's own mean? Bars must clear "
            "R² = 0 to mean anything, and the percentage is how many samples clear it."
        ),
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
        "caption": (
            "Where along the 245 bins the error sits. Flat and low is good; spikes at the "
            "two ends are convolution edge artifacts, and the lower row shows systematic "
            "bias."
        ),
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
        "caption": (
            "Predicted against true amplitude. A tight diagonal cloud is faithful; a "
            "flattened cloud means the output is collapsing toward the mean."
        ),
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
        "caption": (
            "Which frequencies survive. A ratio dropping below 1 at high frequency means "
            "narrow spectral lines are being smoothed away — invisible in MSE."
        ),
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
        "caption": (
            "Which kinds of spectra reconstruct badly. A flat line means the model is "
            "indifferent to that property; an upward slope names a weakness."
        ),
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
    "recon_reference_ladder": {
        "title": "Reconstruction skill against matched-rate references",
        "caption": (
            "How each head compares with simply resampling the target at the same rate its "
            "bottleneck provides. Effective resolution is the head's error read back as a "
            "number of signal samples."
        ),
        "what": (
            "The reference curve reconstructs each target from K evenly spaced samples of "
            "itself, linearly interpolated back to full length (solid), and from an ideal "
            "band-limit at the same rate (dashed) - neither dominates, so the pair "
            "brackets what the rate can express. The feature extractor compresses 245 "
            "bins to 47 timesteps, so K = 47 is the temporal rate every head decodes from "
            "and is marked on each panel. Reading a head's median error back onto the "
            "interpolation curve gives its EFFECTIVE RESOLUTION: the number of samples of "
            "the signal it actually delivers. The remaining panels compare that number "
            "across heads and datasets, express the same thing as a ratio against the "
            "reference at 47, and show the per-sample spread behind the medians."
        ),
        "read": (
            "further right on the ladder is a finer reconstruction; a head sitting to the "
            "LEFT of the K = 47 line delivers less resolution than the rate it decodes "
            "from, and in the ratio panel any bar above 1.0 is a head with more error than "
            "resampling the target at its own rate would incur"
        ),
        "good": (
            "Effective resolution at or above 47 and a ratio at or below 1.0 - the head is "
            "extracting everything its temporal rate carries. A head well short of 47 is "
            "limited by what it learned, not by the bottleneck."
        ),
        "caveats": (
            "The reference is an ORACLE at its sample points: it reads true target values "
            "the model never sees, so it is not a bound on what the model could achieve. "
            "It is a bound on what temporal downsampling alone costs, which is why the "
            "useful direction is the negative one - a head below the reference at its own "
            "rate cannot blame the bottleneck. The real bottleneck is also 47 timesteps of "
            "512 or 768 channels, not 47 scalars, so the model has far more capacity per "
            "step than the interpolant. Interpolation flatters smooth signals; the "
            "subtitle reports what share of this data has structure narrower than the "
            "reference's sample spacing, so the reader can discount accordingly. "
            + CROSS_HEAD_CAVEAT
        ),
    },
    "recon_failure_anatomy": {
        "title": "Anatomy of the worst reconstructions",
        "caption": (
            "How much of the total error a few samples carry, and what those samples have "
            "in common - component index, source dataset, or a property of the signal."
        ),
        "what": (
            "Top: the cumulative share of a dataset's total squared error carried by its "
            "worst samples, ranked worst first, one panel per head and one curve per "
            "dataset; the diagonal is error spread perfectly evenly. Then the same thing "
            "as a single number per head and dataset. The remaining panels open up ONE "
            "dataset - the one whose error is most concentrated, named in their titles - "
            "showing which levels of which property are over-represented in its tail "
            "(lift = the level's share of the tail divided by its share of the dataset), "
            "how far the tail sits from the rest on the property that separates them best, "
            "and what the worst reconstructions actually look like. Heads are named in "
            "short here; what each one taps and decodes with is spelled out in the "
            "error-distribution figure."
        ),
        "read": (
            "a curve bending sharply toward the top-left means a handful of samples carry "
            "most of the error, so the mean describes them rather than the dataset; in the "
            "lift panel a bar at 5 means that level appears five times as often in the "
            "tail as in the data, and a level whose bar is tall AND whose tail share is "
            "large is a named failure mode rather than a coincidence"
        ),
        "good": (
            "Curves close to the diagonal and no level far above lift 1 - failure spread "
            "evenly rather than concentrated in one component or one kind of spectrum. "
            "Concentration is not itself a defect, but it does mean the average is the "
            "wrong summary and that there is a specific thing to fix."
        ),
        "caveats": (
            "Concentration is close to independent of how GOOD the model is: measured "
            "against an under-trained checkpoint of the same run, effective resolution "
            "fell by up to 2.6x while the worst-5% share barely moved. This figure "
            "describes the data's hard subpopulation, not the training state - the "
            "reference-ladder figure is the one that tracks progress. "
            "The tail is defined per head, and the heads do not fail on the same samples, "
            "so each head has its own; the opened-up panels follow the most concentrated "
            "one, and every dataset's full table is exported to recon_tail_lift.csv "
            "(and to tail_lift_df.csv beside each dataset's own figures). A level "
            "needs enough tail and population rows to be charted - what that excludes is "
            "stated on the panel rather than dropped silently. Lift on a quintile axis "
            "cannot exceed 5 by construction, so categorical axes like component index can "
            "outrank them structurally; compare bars within an axis type. Separation is "
            "reported as a median difference in units of the population's interquartile "
            "range, not a ratio, because several descriptors cross zero. When the "
            "checkpoint used normalize=True the contrast descriptor is dropped: layer-norm "
            "forces it to 1 for every sample and what remains is float noise."
        ),
    },
    "recon_component_error": {
        "title": "Where the reconstruction error lives, by component",
        "caption": (
            "Error per component index, and what share of the dataset's whole error "
            "budget each one carries. A component well above its share of the samples is "
            "a named problem rather than a hard dataset."
        ),
        "what": (
            "Multi-component data gives every wav a component index from its filename, "
            "and a component behaves as a class of signal rather than a point on a scale. "
            "The first panels plot median reconstruction MSE against that index, one per "
            "dataset, with the dataset's own median drawn for scale. The rest open up the "
            "dataset whose budget is most uneven: what share of the total squared error "
            "each component carries against what share of the samples it contributes "
            "(their ratio is the xN on each bar), whether a bad component is uniformly "
            "bad or has its own tail, what a typical sample of it actually looks like, "
            "and which signal property separates the bad components from the rest. Each "
            "example trace is labelled with that component's failure signature: R-squared "
            "against the flat-line baseline, the amplitude ratio, and the correlation "
            "with the target - an amplitude far below 1 with a high correlation is a "
            "decoder hedging toward the mean, while an amplitude above 1 with a "
            "correlation near zero is one emitting something large and unrelated. "
            "Heads are named in short here; what each one taps and decodes with is "
            "spelled out in the error-distribution figure."
        ),
        "read": (
            "a component sitting far above the dataset median line is reconstructing badly, "
            "and in the budget panel a bar rising well above its sample-share dash is one "
            "carrying more than its weight - xN says how much more, so x13 means thirteen "
            "times the error its sample count would account for"
        ),
        "good": (
            "A flat line near the dataset median and every budget bar level with its "
            "sample-share dash - error that does not depend on which component a signal "
            "came from. Two components holding most of a dataset's error is not a modelling "
            "verdict on its own; it is a pointer at those components."
        ),
        "caveats": (
            "The share of ERROR is the number that matters here and it is not the median: "
            "a component can look unremarkable at the median and still dominate a dataset "
            "through its tail. Components with fewer than five samples in the draw are "
            "omitted, since a share estimated from a handful of samples is noise. "
            "Component index is a label, not a difficulty scale - the x-axis is ordered by "
            "index only so the same component sits in the same place across panels. The "
            "example traces come from the bounded raw-signal subsample and show a sample "
            "with that component's TYPICAL error, not its worst. Multi-component datasets "
            "only; single-component ones carry no component field. " + CROSS_HEAD_CAVEAT
        ),
    },
}


_REQUIRED_DOC_KEYS = ("title", "caption", "what", "read", "good", "caveats")


def doc_for_figure(path: str) -> dict:
    """
    The doc entry for a figure file, matched on its basename, or None if unregistered.
    Longest key first so `recon_error_vs_signal_properties` is not matched by a shorter
    key that happens to be a prefix.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    for key in sorted(_FIG_DOC, key=len, reverse=True):
        if base.startswith(key):
            return dict(_FIG_DOC[key], key=key)
    return None


def _doc(key: str) -> dict:
    if key in _FIG_DOC:
        missing = [k for k in _REQUIRED_DOC_KEYS if not _FIG_DOC[key].get(k)]
        if missing:
            raise KeyError("recon_plots: figure %r is missing _FIG_DOC keys %s"
                           % (key, missing))
    if key not in _FIG_DOC:
        raise KeyError(
            "recon_plots: figure %r has no _FIG_DOC entry. Every figure must ship with "
            "its explanation - add one before returning the figure." % key
        )
    return _FIG_DOC[key]


def _caption(key: str) -> str:
    """
    The one-line caption the report prints under the figure. Deliberately short: the
    long-form what/how/caveats lives in the on-image footnote and FIGURES.md, and a
    paragraph under every image just gets skipped.
    """
    return _doc(key)["caption"]


def _footnote(fig, key: str, extra: str = "", width: int = 150) -> None:
    """
    Draw the explanation onto the figure itself, below the axes.

    Skipped in the e-ink style: the PDF sets the same text as real type at a readable
    size, and 6.5pt monospace rendered into a bitmap is exactly what an e-paper panel
    handles worst.
    """
    if _STYLE == "eink":
        return
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
            "*" + d["caption"] + "*", "",
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

def _tight(fig, reserve_in: float = 0.85) -> None:
    """
    tight_layout leaving a fixed height (in inches) clear for the suptitle.

    A fractional rect does not travel between layouts: the 0.93 that leaves the right gap
    on a 5-inch-tall figure leaves an inch of blank paper on the 11-inch ones the e-ink
    layouts produce.
    """
    h = float(fig.get_size_inches()[1])
    fig.tight_layout(rect=[0, 0, 1, max(0.70, 1.0 - reserve_in / max(h, 1e-6))])


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
    # Above the axes, so these never fight with bars, annotations or violins. The panel
    # title lives there too, so push it up by one label line first - callers set the
    # title before calling this, and without the extra pad the two overlap.
    title = ax.get_title()
    if title:
        ax.set_title(title, fontsize=ax.title.get_fontsize(),
                     color=ax.title.get_color(),
                     fontweight=ax.title.get_fontweight(), pad=16)

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


def _two_row_grid(n_datasets: int, cell_w: float, cell_h: float, **kw):
    """
    A 2-rows-by-datasets grid, transposed in the e-ink style.

    Four datasets side by side gives a figure nearly three times wider than it is tall,
    which on a portrait e-reader page has to be shrunk until nothing is legible. Turned
    on its side - one dataset per row, the two quantities as columns - it is close to
    page-shaped. The returned `cell(row, col)` addresses cells logically either way.
    """
    if _STYLE == "eink":
        fig, axes = plt.subplots(n_datasets, 2, squeeze=False,
                                 figsize=(2 * cell_h * 0.92, n_datasets * cell_w * 0.62),
                                 **kw)
        return fig, (lambda r, c: axes[c][r])
    fig, axes = plt.subplots(2, n_datasets, squeeze=False,
                             figsize=(cell_w * n_datasets, cell_h * 2), **kw)
    return fig, (lambda r, c: axes[r][c])


def _flow_axes(n: int, cell_w: float, cell_h: float,
               ncols_screen: int, ncols_eink: int = 2):
    """
    n panels flowed into a grid, narrower in the e-ink style.

    `_two_row_grid` assumes exactly two quantities per dataset. The reference-ladder and
    failure-anatomy figures mix per-dataset panels with aggregate ones, so they need a
    plain flow instead. Unused cells are switched off rather than left as empty axes.
    """
    ncols = max(1, min(ncols_eink if _STYLE == "eink" else ncols_screen, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(cell_w * ncols, cell_h * nrows))
    flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


def _comp_discriminator(cdf: pd.DataFrame, head: str) -> str:
    """
    The signal descriptor whose per-component median best tracks per-component error.

    Rank correlation, over components rather than samples: the question is what the bad
    COMPONENTS have in common, and a component is a class of signal, so its median
    descriptor is the right summary. Returns None when nothing varies enough to rank.
    """
    col = f"{head}_mse_median"
    if col not in cdf.columns or len(cdf) < 4:
        return None
    y = cdf[col].to_numpy(dtype=float)
    best, best_rho = None, 0.0
    for axis in FEATURE_LABELS:
        if axis not in cdf.columns:
            continue
        x = cdf[axis].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 4 or np.unique(x[ok]).size < 3:
            continue
        xr = pd.Series(x[ok]).rank().to_numpy()
        yr = pd.Series(y[ok]).rank().to_numpy()
        sx, sy = xr.std(), yr.std()
        if sx <= 0 or sy <= 0:
            continue
        rho = float(np.mean((xr - xr.mean()) * (yr - yr.mean())) / (sx * sy))
        if abs(rho) > abs(best_rho):
            best, best_rho = axis, rho
    return best


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

    fig, cell = _two_row_grid(len(heads), 5.6, 4.2)

    for c, k in enumerate(heads):
        ax = cell(0, c)
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
                    marker=_DATASET_MARKERS[i % len(_DATASET_MARKERS)],
                    markevery=max(len(v) // 12, 1), ms=4.5, markerfacecolor="none",
                    label="%s  [%s]" % (_ds_label(r, short=True), group))
            ax.axvline(np.median(v), color=_dataset_color(a, i), ls=":", lw=0.8, alpha=0.6)
        _error_axis(ax, all_mse)
        ax.set_ylabel("fraction of samples at or below", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(_head_title(k), fontsize=8, color=_head_text_color(k),
                     fontweight="bold")
        ax.legend(fontsize=6.5, loc="lower right", title="dotted vline = median",
                  title_fontsize=6)

        ax = cell(1, c)
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
                         fontsize=8.5, color=_head_text_color(k))
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
    _tight(fig, reserve_in=1.0)
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
    ("amp_ratio_median", "amplitude ratio\nmedian (1 = ideal)", 1.0),
    ("peak_err_median",  "peak error\nmedian (0 = ideal)", 0.0),
    ("k_eff_median",     "effective\nresolution", True),
]


def _plot_summary_heatmap(by_alias: dict, out_dir: str) -> list:
    aliases = _ordered_aliases(by_alias)
    heads = _heads_of(by_alias)
    if not aliases or not heads:
        return []

    # One block per head, stacked VERTICALLY. Side by side, 8 metric columns times three
    # heads makes a 30-inch-wide figure in which the cell values are unreadably small;
    # stacking keeps the width at one block and gives every cell room.
    fig, axes = plt.subplots(len(heads), 1,
                             figsize=(1.15 * len(_SUMMARY_METRICS) + 2.0,
                                      (0.52 * len(aliases) + 1.5) * len(heads)),
                             squeeze=False)
    for c, k in enumerate(heads):
        ax = axes[c][0]
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
            if ok.sum() < 2:
                continue
            spread = float(np.ptp(col[ok]))
            scale = max(abs(float(np.nanmedian(col[ok]))), 1e-12)

            # Rank alone is the wrong thing to colour. Normalising a column to its own
            # min and max spans the full ramp however small the spread is, so two values
            # differing in the fourth decimal come out fully red and fully green - the
            # figure then asserts a difference that is not there. Colour SATURATION is
            # therefore scaled by the spread relative to the column's own magnitude: a
            # column varying by _FULL_SATURATION_AT or more is shaded at full strength,
            # anything tighter stays near neutral, and a column that is effectively
            # constant is left alone entirely.
            saturation = min(1.0, (spread / scale) / _FULL_SATURATION_AT)
            if saturation < 0.02:
                continue

            norm = (col - np.nanmin(col[ok])) / spread
            # A float in place of the direction flag names an ideal VALUE rather than a
            # direction - amplitude ratio is best at 1, peak error best at 0 - and the
            # shading then ranks by distance from it. isinstance(True, int) is True, so
            # the bool check has to come first.
            if not isinstance(higher_better, bool) and isinstance(higher_better, (int, float)):
                tgt = float(higher_better)
                norm = 1.0 - (np.abs(col - tgt)
                              / max(np.nanmax(np.abs(col[ok] - tgt)), 1e-9))
            elif not higher_better:
                norm = 1.0 - norm
            shade[:, j] = 0.5 + (norm - 0.5) * saturation
        ax.imshow(shade, cmap=_rank_cmap(), vmin=0, vmax=1, aspect="auto")

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isfinite(v):
                    txt = "-"
                elif abs(v) < 1e-9:
                    txt = "0"                      # not "-2.22e-16"
                elif 1e-3 <= abs(v) < 1e4:
                    txt = "%.3f" % v
                else:
                    txt = "%.2e" % v
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(_SUMMARY_METRICS)))
        # Only the bottom block needs metric labels; repeating them three times just
        # eats vertical space between the blocks.
        if c == len(heads) - 1:
            ax.set_xticklabels([lbl for _, lbl, _ in _SUMMARY_METRICS], fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_title(_head_title(k, width=70), fontsize=9, color=_head_text_color(k),
                     fontweight="bold")
        # Divider between the single- and multi-component row blocks.
        groups = [by_alias[a].get("component_group", "single") for a in aliases
                  if isinstance(by_alias[a].get("summary_df"), pd.DataFrame)]
        if "single" in groups and "multi" in groups:
            ax.axhline(groups.index("multi") - 0.5, color="#222222", lw=2.0)

    _suptitle(fig, "recon_summary_heatmap", _run_note(by_alias), y=1.02)
    _tight(fig, reserve_in=1.0)
    _footnote(fig, "recon_summary_heatmap")
    path = _save_fig(fig, os.path.join(out_dir, "recon_summary_heatmap.png"))
    return [(_caption("recon_summary_heatmap"), path)]


# ── F2: skill against the mean-predictor baseline ─────────────────────────────

def _plot_skill_vs_baseline(by_alias: dict, out_dir: str) -> list:
    aliases = _ordered_aliases(by_alias)
    heads = _heads_of(by_alias)
    if not aliases or not heads:
        return []

    if _STYLE == "eink":
        fig, axes = plt.subplots(2, 1, figsize=(7.4, 9.2),
                                 gridspec_kw={"height_ratios": [1.15, 1.0]})
    else:
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
               hatch=_HEAD_HATCH[k], edgecolor="white", linewidth=0.6,
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
              bbox_to_anchor=(0.5, -0.34 if _STYLE == "eink" else -0.22), frameon=False)
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
        handles = [plt.Line2D([], [], color=_HEAD_COLOR[k], marker=_HEAD_MARKER[k],
                              ls="", label=PATHWAY_SHORT[k]) for k in heads]
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
    _tight(fig, reserve_in=1.0)
    # The right panel has two modes and the fixed guidance text describes the diagonal
    # one, so say which mode actually ran.
    mode_note = (
        "the right panel is in threshold mode - var(target) is effectively constant "
        "under normalize=True, so the break-even line is HORIZONTAL, not a diagonal, and "
        "the x-axis is peak prominence rather than baseline error. Points below the line "
        "still mean the model beats the baseline."
        if constant_baseline else
        "the right panel is in diagonal mode - baseline error varies across samples, so "
        "the break-even line is the y = x diagonal.")
    _footnote(fig, "recon_skill_vs_baseline", extra=mode_note)
    path = _save_fig(fig, os.path.join(out_dir, "recon_skill_vs_baseline.png"))
    return [(_caption("recon_skill_vs_baseline"), path)]


# ── F3: per-position error profile ────────────────────────────────────────────

def _plot_position_profile(by_alias: dict, out_dir: str) -> list:
    aliases = [a for a in _ordered_aliases(by_alias) if by_alias[a].get("profiles")]
    heads = _heads_of(by_alias)
    if not aliases or not heads:
        return []

    fig, cell = _two_row_grid(len(aliases), 4.9, 3.6, sharex=True)
    for c, a in enumerate(aliases):
        r = by_alias[a]
        prof = r["profiles"]
        group = r.get("component_group", "single")
        bins = np.arange(len(prof["per_position_target_std"]))

        ax = cell(0, c)
        ax.fill_between(bins, 0, prof["per_position_target_std"],
                        color=_REF_COLOR, alpha=0.45,
                        label="target variability (per-bin std) — scale reference")
        for k in heads:
            key = f"per_position_abs_err_{k}"
            if key in prof:
                ax.plot(bins, prof[key], color=_HEAD_COLOR[k], lw=1.3,
                        ls=_HEAD_LS[k],
                        label="%s — mean |error|" % PATHWAY_SHORT[k])
        ax.set_ylabel("mean |error| at this bin  (↓ better)", fontsize=8)
        ax.set_title("%s\n[%s]" % (_ds_label(r), _GROUP_LABEL[group]), fontsize=8.5,
                     color="#444444")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, bins[-1])
        if c == 0:
            ax.legend(fontsize=6.5, loc="upper center")

        ax = cell(1, c)
        ax.axhline(0.0, color="#000000", lw=1.0)
        for k in heads:
            key = f"per_position_signed_err_{k}"
            if key in prof:
                ax.plot(bins, prof[key], color=_HEAD_COLOR[k], lw=1.3,
                        ls=_HEAD_LS[k],
                        label="%s — mean signed error" % PATHWAY_SHORT[k])
        ax.set_ylabel("mean signed error  (0 = unbiased)", fontsize=8)
        ax.set_xlabel("signal bin index (0–%d)" % bins[-1], fontsize=8.5)
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.legend(fontsize=6.5, loc="upper center")

    _suptitle(fig, "recon_position_profile", _run_note(by_alias))
    _tight(fig, reserve_in=1.0)
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
    range_modes = set()
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
            hb = ax.hexbin(tf, pf, gridsize=60, bins="log", cmap=_density_cmap(),
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
            # Name the DATASET here, not just the component group: the grid has two
            # single-component rows and two multi-component ones, so a group-only title
            # repeats itself over different data and leaves the row unidentifiable.
            ax.set_title("%s\n%s — %s" % (PATHWAY_SHORT[k], _ds_label(r, short=True),
                                          _GROUP_SHORT[group]),
                         fontsize=8.5, color=_head_text_color(k), fontweight="bold")
            ax.legend(fontsize=6, loc="upper left", framealpha=0.75)
            if _STYLE != "eink":
                # The colourbars cost a quarter of the width and the density scale is not
                # what the figure is for; on a page that width is worth more.
                fig.colorbar(hb, ax=ax, label="point density (log)", fraction=0.046)

        # Rightmost column: per-sample dynamic range.
        #
        # Normally a prediction-std vs target-std scatter. But normalize=True layer-norms
        # every target to unit variance, so target std is 1 for every sample and the
        # scatter degenerates into a single vertical strip. The ratio std(pred)/std(target)
        # carries exactly the same information and stays readable, so plot its
        # distribution instead when that happens.
        #
        # The test has to be RELATIVE, not an absolute range. Measured target std spans
        # 0.9837-0.99999 on sampled_data - every value is still 1.0 for plotting purposes,
        # but the absolute spread (1.6e-2) clears an absolute 1e-4 threshold, so the old
        # test took the scatter branch and drew exactly the vertical strip it existed to
        # avoid. contrast_is_collapsed is the same robust p99/p1 predicate the skill and
        # stratification figures already use for this, so all of them now agree.
        ax = axes[rowi][n_cols - 1]
        ts = t.std(axis=1)
        degenerate = recon_analysis.contrast_is_collapsed(ts)
        range_modes.add("ratio-histogram" if degenerate else "std-vs-std scatter")

        if degenerate:
            for k in heads:
                if k not in preds:
                    continue
                ratio = preds[k].std(axis=1) / np.where(ts > 0, ts, np.nan)
                ratio = ratio[np.isfinite(ratio)]
                if not ratio.size:
                    continue
                ax.hist(ratio, bins=40, color=_HEAD_COLOR[k], alpha=0.55,
                        hatch=_HEAD_HATCH[k], edgecolor="white",
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
                ax.scatter(ts, preds[k].std(axis=1), s=8, alpha=0.4, linewidths=0,
                           marker=_HEAD_MARKER[k], color=_HEAD_COLOR[k],
                           label=PATHWAY_SHORT[k], rasterized=True)
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
    _tight(fig, reserve_in=1.0)
    note = "the rightmost column is in %s mode" % " and ".join(sorted(range_modes))
    if "ratio-histogram" in range_modes:
        note += (" - target std is 1 for every sample under normalize=True, so the ratio "
                 "std(prediction)/std(target) is plotted directly instead of a scatter "
                 "against a constant")
    _footnote(fig, "recon_amplitude_calibration", extra=note + ".")
    path = _save_fig(fig, os.path.join(out_dir, "recon_amplitude_calibration.png"))
    return [(_caption("recon_amplitude_calibration"), path)]


# ── F6: spectral fidelity ─────────────────────────────────────────────────────

def _plot_spectral_fidelity(by_alias: dict, out_dir: str) -> list:
    aliases = [a for a in _ordered_aliases(by_alias) if by_alias[a].get("profiles")]
    heads = _heads_of(by_alias)
    if not aliases or not heads:
        return []

    fig, cell = _two_row_grid(len(aliases), 4.9, 3.6, sharex=True)
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

        ax = cell(0, c)
        ax.plot(freq, tgt, color="black", lw=1.8, label="target (ground truth)")
        for k in heads:
            key = f"mean_fft_pred_{k}"
            if key in prof:
                ax.plot(freq, prof[key][1:], color=_HEAD_COLOR[k], lw=1.2,
                        ls=_HEAD_LS[k],
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

        ax = cell(1, c)
        ax.axhline(1.0, color="black", lw=1.4, label="ratio = 1 (magnitude preserved)")
        ratios = []
        for k in heads:
            key = f"mean_fft_pred_{k}"
            if key in prof:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(tgt > 0, prof[key][1:] / tgt, np.nan)
                ratios.append(ratio)
                ax.plot(freq, ratio, color=_HEAD_COLOR[k], lw=1.2, ls=_HEAD_LS[k],
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
    _tight(fig, reserve_in=1.0)
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

    n_cols = min(2 if _STYLE == "eink" else 4, max(1, len(axes_present)))
    n_rows = int(np.ceil(len(axes_present) / n_cols)) + (1 if has_spec else 0)
    cell_h = 2.75 if _STYLE == "eink" else 3.5
    fig, grid = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, cell_h * n_rows),
                             squeeze=False)

    for i, axis in enumerate(axes_present):
        ax = grid[i // n_cols][i % n_cols]
        sub = strat[strat["axis"] == axis].sort_values("bin")
        x = np.arange(len(sub))
        for k in heads:
            col = f"{k}_mse_median"
            if col not in sub.columns:
                continue
            ax.plot(x, sub[col], marker=_HEAD_MARKER[k], ms=4, lw=1.5,
                    ls=_HEAD_LS[k], color=_HEAD_COLOR[k], label=PATHWAY_SHORT[k])
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
                        color=_HEAD_COLOR[k], lw=1.5, ls=_HEAD_LS[k],
                        label=PATHWAY_SHORT[k])
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
    _tight(fig, reserve_in=1.0)
    _footnote(fig, "recon_error_vs_signal_properties", extra=extra)
    path = _save_fig(fig, os.path.join(out_dir, "recon_error_vs_signal_properties.png"))
    return [(_caption("recon_error_vs_signal_properties"), path)]


# ── Public entry points ───────────────────────────────────────────────────────

# Cross-dataset figures, in the order a reader should meet them: how big is the error,
# then how it compares to doing nothing, then where it lives, then what it destroys.
# ── F7: skill against matched-rate reference reconstructions ──────────────────

def _accent() -> str:
    """Colour for a reference line that is neither a head nor a dataset."""
    return "#000000" if _STYLE == "eink" else "#b00020"


def _reference_of(r: dict):
    ref = r.get("reference")
    return ref if isinstance(ref, dict) and ref.get("ladder") else None


def _plot_reference_ladder(by_alias: dict, out_dir: str) -> list:
    aliases = [a for a in _ordered_aliases(by_alias) if _reference_of(by_alias[a])]
    if not aliases:
        return []
    heads = _heads_of({a: by_alias[a] for a in aliases})
    if not heads:
        return []

    n = len(aliases)
    fig, axes = _flow_axes(n + 3, 4.7, 3.8, ncols_screen=max(2, min(n, 4)))

    # ── One ladder per dataset ────────────────────────────────────────────────
    for i, a in enumerate(aliases):
        r, ax = by_alias[a], axes[i]
        ref = _reference_of(r)
        rdf = r["results_df"]
        ks = np.asarray(ref["ladder"], dtype=float)
        ax.plot(ks, ref["interp_mse_median"], "-o", color="#333333", ms=3.2, lw=1.5,
                label="reference: interpolate the target through K points")
        lp = ref.get("lowpass_mse_median")
        if lp is not None:
            ax.plot(ks, lp, "--", color="#777777", lw=1.2,
                    label="reference: ideal band-limit at the same rate")

        placed = []
        coarsest = float(np.asarray(ref["interp_mse_median"], dtype=float)[0])
        for k in heads:
            if f"{k}_k_eff" not in rdf.columns or f"{k}_mse" not in rdf.columns:
                continue
            mse = float(rdf[f"{k}_mse"].median())
            k_eff = float(rdf[f"{k}_k_eff"].median())
            ax.axhline(mse, color=_HEAD_COLOR[k], ls=_HEAD_LS[k], lw=1.1, alpha=0.75)
            # A head with more error than the coarsest rung has no place on the ladder.
            # Its marker is pinned to the left edge, so the label has to say the number
            # is a bound rather than let it read as a measurement.
            placed.append((k, mse, k_eff, mse > coarsest))

        ax.set_xscale("log")
        ax.set_yscale("log")
        for k, mse, k_eff, is_off in placed:
            ax.plot([k_eff], [mse], marker=_HEAD_MARKER[k], color=_HEAD_COLOR[k],
                    ms=8, mec="white", mew=0.9, ls="none", zorder=5,
                    label="%s: effective resolution %s%.0f"
                          % (PATHWAY_SHORT[k], "\u2264 " if is_off else "", k_eff))
            ax.plot([k_eff, k_eff], [ax.get_ylim()[0], mse], color=_HEAD_COLOR[k],
                    ls=":", lw=0.9, alpha=0.8, zorder=1)

        bk = float(ref["bottleneck_k"])
        ax.axvline(bk, color=_accent(), lw=1.5, ls="-.", zorder=2)
        ax.text(bk, ax.get_ylim()[1], "K=%d: the rate the\nfeature extractor delivers " % bk,
                color=_accent(), fontsize=6.8, va="top", ha="right")
        ax.set_xlabel("K — samples of the signal the reference keeps  "
                      "(→ finer)", fontsize=8)
        ax.set_ylabel("median MSE  (↓ better)", fontsize=8)
        smooth = ""
        fw, spacing = ref.get("peak_fwhm_median"), ref.get("sample_spacing")
        if fw is not None and np.isfinite(fw) and spacing:
            smooth = ("\npeaks ~%.0f bins wide vs a %.1f-bin reference spacing at K=%d"
                      % (fw, spacing, int(ref.get("bottleneck_k", 0))))
        ax.set_title(_ds_label(r) + smooth, fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        leg = ax.legend(fontsize=6.2, loc="lower left", framealpha=0.9,
                        title=("\u2264 marks a head off the coarse end of the ladder"
                               if any(p[3] for p in placed) else None))
        if leg.get_title().get_text():
            leg.get_title().set_fontsize(6.0)
            leg.get_title().set_color(_accent())

    # ── Effective resolution, per head and dataset ───────────────────────────
    ax = axes[n]
    x = np.arange(len(aliases), dtype=float)
    width = 0.8 / max(len(heads), 1)
    bk = float(_reference_of(by_alias[aliases[0]])["bottleneck_k"])
    for j, k in enumerate(heads):
        xs, vals = [], []
        for i, a in enumerate(aliases):
            rdf = by_alias[a]["results_df"]
            if f"{k}_k_eff" not in rdf.columns:
                continue
            xs.append(x[i] + (j - (len(heads) - 1) / 2) * width)
            vals.append(float(rdf[f"{k}_k_eff"].median()))
        if not xs:
            continue
        ax.bar(xs, vals, width, color=_HEAD_COLOR[k], alpha=0.9, hatch=_HEAD_HATCH[k],
               edgecolor="white", linewidth=0.6, label=PATHWAY_SHORT[k])
        for xi, v in zip(xs, vals):
            ax.text(xi, v, "%.0f\n%.0f%%" % (v, 100 * v / bk), ha="center", va="bottom",
                    fontsize=6.2)
    ax.axhline(bk, color=_accent(), lw=1.6, ls="-.",
               label="the %d the bottleneck provides" % bk)
    ax.set_xticks(x)
    ax.set_xticklabels([_ds_label(by_alias[a], short=True) for a in aliases],
                       fontsize=7.5, rotation=12, ha="right")
    ax.set_ylabel("effective resolution\n(samples delivered, ↑ better)", fontsize=8)
    ax.set_ylim(0, max(ax.get_ylim()[1], bk * 1.6))
    ax.set_title("How much resolution each head actually delivers", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _group_divider(ax, aliases, by_alias, x)
    ax.legend(fontsize=6.2, loc="upper center", ncol=2, framealpha=0.9)

    # ── The same thing as a ratio against the reference at the bottleneck rate ─
    ax = axes[n + 1]
    for j, k in enumerate(heads):
        xs, vals = [], []
        for i, a in enumerate(aliases):
            r = by_alias[a]
            ref = _reference_of(r)
            col = "ref_interp%d_mse" % int(ref["bottleneck_k"])
            rdf = r["results_df"]
            if col not in rdf.columns or f"{k}_mse" not in rdf.columns:
                continue
            denom = float(rdf[col].median())
            if not np.isfinite(denom) or denom <= 0:
                continue
            xs.append(x[i] + (j - (len(heads) - 1) / 2) * width)
            vals.append(float(rdf[f"{k}_mse"].median()) / denom)
        if not xs:
            continue
        ax.bar(xs, vals, width, color=_HEAD_COLOR[k], alpha=0.9, hatch=_HEAD_HATCH[k],
               edgecolor="white", linewidth=0.6, label=PATHWAY_SHORT[k])
        for xi, v in zip(xs, vals):
            ax.text(xi, v, "%.1fx" % v, ha="center", va="bottom", fontsize=6.4)
    ax.axhline(1.0, color=_accent(), lw=1.6)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([_ds_label(by_alias[a], short=True) for a in aliases],
                       fontsize=7.5, rotation=12, ha="right")
    ax.set_ylabel("median MSE ÷ reference MSE at K=%d\n(↓ better, 1.0 = matched)"
                  % bk, fontsize=8)
    ax.set_title("Error relative to resampling the target at the same rate", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y", which="both")
    _group_divider(ax, aliases, by_alias, x)

    # ── Per-sample spread behind those medians ───────────────────────────────
    ax = axes[n + 2]
    positions, tick_pos, tick_lab, drawn = [], [], [], False
    for i, a in enumerate(aliases):
        rdf = by_alias[a]["results_df"]
        base = i * (len(heads) + 1.0)
        for j, k in enumerate(heads):
            col = f"{k}_k_eff"
            if col not in rdf.columns:
                continue
            v = rdf[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if len(v) < 5:
                continue
            pos = base + j
            parts = ax.violinplot([v], positions=[pos], widths=0.85,
                                  showmedians=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(_HEAD_COLOR[k])
                body.set_alpha(0.65)
            if "cmedians" in parts:
                parts["cmedians"].set_color("#000000")
            positions.append(pos)
            drawn = True
        tick_pos.append(base + (len(heads) - 1) / 2.0)
        tick_lab.append(_ds_label(by_alias[a], short=True))
    if not drawn:
        ax.axis("off")
    else:
        ax.axhline(bk, color=_accent(), lw=1.6, ls="-.",
                   label="the %d the bottleneck provides" % bk)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=7.5, rotation=12, ha="right")
        ax.set_ylabel("effective resolution, per sample", fontsize=8)
        ax.set_title("Spread behind the medians\n(one violin per head, order as above)",
                     fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(top=max(ax.get_ylim()[1], bk * 1.45))
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=_HEAD_COLOR[k],
                                         alpha=0.65, label=PATHWAY_SHORT[k])
                           for k in heads]
                  + [plt.Line2D([], [], color=_accent(), lw=1.6, ls="-.",
                                label="the %d the bottleneck provides" % bk)],
                  fontsize=6.2, loc="upper center", ncol=2, framealpha=0.9)

    # How smooth each dataset is decides how strong its reference is, and that varies a
    # lot between them - so it is stated per panel, not once for whichever came first.
    shares = [(_reference_of(by_alias[a]) or {}).get("narrow_peak_share") for a in aliases]
    shares = [x for x in shares if x is not None and np.isfinite(x)]
    note = ""
    if shares:
        lo, hi = 100 * min(shares), 100 * max(shares)
        span = "%.0f%%" % hi if round(lo) == round(hi) else "%.0f-%.0f%%" % (lo, hi)
        note = (" %s of samples have a main peak narrower than the reference's sample "
                "spacing, so it is a %s reference for this data."
                % (span, "strong" if max(shares) < 0.1 else "partly strained"))
    _suptitle(fig, "recon_reference_ladder", _run_note(by_alias) + note, width=150)
    _tight(fig, reserve_in=0.7)
    _footnote(fig, "recon_reference_ladder")
    return [(_caption("recon_reference_ladder"),
             _save_fig(fig, os.path.join(out_dir, "recon_reference_ladder.png")))]


# ── F8: anatomy of the worst reconstructions ─────────────────────────────────

def _tail_of(r: dict):
    t = r.get("tail")
    return t if isinstance(t, dict) and t.get("per_head") else None


def _plot_failure_anatomy(by_alias: dict, out_dir: str) -> list:
    aliases = [a for a in _ordered_aliases(by_alias) if _tail_of(by_alias[a])]
    if not aliases:
        return []
    heads = _heads_of({a: by_alias[a] for a in aliases})
    if not heads:
        return []

    frac = float(_tail_of(by_alias[aliases[0]]).get("frac", 0.05))

    # The dataset worth opening up is the one whose error is most concentrated: on a
    # dataset where the worst 5% carry 13% of the error there is no tail to explain.
    def _peak_share(a):
        per = _tail_of(by_alias[a])["per_head"]
        vals = [v["share"] for v in per.values() if np.isfinite(v.get("share", np.nan))]
        return max(vals) if vals else -1.0

    focus = max(aliases, key=_peak_share)
    focus_r = by_alias[focus]
    focus_head = _tail_of(focus_r).get("reference_head") or heads[0]

    fig, axes = _flow_axes(len(heads) + 4, 4.7, 3.8,
                           ncols_screen=max(2, min(len(heads) + 1, 4)))

    # ── Lorenz curves, one panel per head ────────────────────────────────────
    for j, k in enumerate(heads):
        ax = axes[j]
        ax.plot([0, 1], [0, 1], color="#999999", lw=1.0, ls="--",
                label="error spread perfectly evenly")
        for i, a in enumerate(aliases):
            rdf = by_alias[a]["results_df"]
            if f"{k}_mse" not in rdf.columns:
                continue
            xs, ys = recon_analysis.lorenz_curve(rdf[f"{k}_mse"].to_numpy(dtype=float))
            share = _tail_of(by_alias[a])["per_head"].get(k, {}).get("share", np.nan)
            # The head is the panel here, so the line style has to separate DATASETS -
            # keying it on the head made every curve in a panel identical in greyscale.
            ax.plot(xs, ys, color=_dataset_color(a, i), lw=1.6,
                    ls=(_EINK_DATASET_LS[i % len(_EINK_DATASET_LS)]
                        if _STYLE == "eink" else "-"),
                    label="%s — worst %.0f%% carry %.0f%%"
                          % (a, 100 * frac, 100 * share) if np.isfinite(share) else a)
        ax.axvline(frac, color=_accent(), lw=1.2, ls=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("fraction of samples, worst first", fontsize=8)
        ax.set_ylabel("share of the dataset's total error", fontsize=8)
        ax.set_title(_head_title(k), fontsize=8.5, color=_head_text_color(k))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.2, loc="lower right", framealpha=0.9)

    # ── Concentration as one number per head and dataset ─────────────────────
    ax = axes[len(heads)]
    x = np.arange(len(aliases), dtype=float)
    width = 0.8 / max(len(heads), 1)
    for j, k in enumerate(heads):
        xs, vals = [], []
        for i, a in enumerate(aliases):
            share = _tail_of(by_alias[a])["per_head"].get(k, {}).get("share", np.nan)
            if not np.isfinite(share):
                continue
            xs.append(x[i] + (j - (len(heads) - 1) / 2) * width)
            vals.append(100 * share)
        if not xs:
            continue
        ax.bar(xs, vals, width, color=_HEAD_COLOR[k], alpha=0.9, hatch=_HEAD_HATCH[k],
               edgecolor="white", linewidth=0.6, label=PATHWAY_SHORT[k])
        for xi, v in zip(xs, vals):
            ax.text(xi, v, "%.0f%%" % v, ha="center", va="bottom", fontsize=6.4)
    ax.axhline(100 * frac, color=_accent(), lw=1.6, ls="-.",
               label="an even distribution would give %.0f%%" % (100 * frac))
    ax.set_xticks(x)
    ax.set_xticklabels([_ds_label(by_alias[a], short=True) for a in aliases],
                       fontsize=7.5, rotation=12, ha="right")
    ax.set_ylabel("%% of total error carried by the worst %.0f%%\n(↓ better)"
                  % (100 * frac), fontsize=8)
    ax.set_title("How concentrated the failure is", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _group_divider(ax, aliases, by_alias, x)
    ax.set_ylim(top=max(ax.get_ylim()[1] * 1.35, 100 * frac * 2))
    ax.legend(fontsize=6.2, loc="upper center", ncol=2, framealpha=0.9)

    # ── Lift: what the focus dataset's tail is made of ───────────────────────
    ax = axes[len(heads) + 1]
    lift = focus_r.get("tail_lift_df")
    pooled = _tail_of(focus_r).get("pooled", {})
    if isinstance(lift, pd.DataFrame) and len(lift):
        top = lift.head(10).iloc[::-1]
        ypos = np.arange(len(top), dtype=float)
        ax.barh(ypos, top["lift"].to_numpy(), color="#555555" if _STYLE == "eink"
                else "#4c72b0", alpha=0.9, edgecolor="white", linewidth=0.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels(["%s = %s" % (a, b) for a, b in
                            zip(top["axis"], top["bin_label"])], fontsize=7)
        for y, (lv, sh, npop) in enumerate(zip(top["lift"], top["tail_share"],
                                               top["n_pop"])):
            ax.text(lv, y, "  %.0f%% of the tail, n=%d" % (100 * sh, npop),
                    va="center", fontsize=6.2)
        ax.axvline(1.0, color="#000000", lw=1.4)
        ceiling = pooled.get("quantile_lift_ceiling")
        if ceiling:
            ax.axvline(ceiling, color=_accent(), lw=1.2, ls=":",
                       label="lift ceiling for a quintile axis")
            ax.legend(fontsize=6.2, loc="lower right", framealpha=0.9)
        ax.set_xlim(0, max(1.6, float(top["lift"].max()) * 1.55))
        ax.set_xlabel("lift — share of the tail ÷ share of the dataset",
                      fontsize=8)
        sub = ""
        if pooled.get("worst_axis_n_tail"):
            sub = "\n" + textwrap.fill(
                "up to %d of %d tail samples sit in levels too small to chart (worst "
                "axis: %s)" % (pooled["worst_axis_n_tail"], pooled.get("n_tail_total", 0),
                               pooled.get("worst_axis")), 52)
        ax.set_title("What the worst %.0f%% is made of — dataset `%s`, %s head%s"
                     % (100 * frac, focus, PATHWAY_SHORT[focus_head], sub), fontsize=8.5)
        ax.grid(True, alpha=0.3, axis="x")
    else:
        ax.axis("off")

    # ── The property that separates that tail from the rest ──────────────────
    ax = axes[len(heads) + 2]
    eff = focus_r.get("tail_effect_df")
    rdf = focus_r["results_df"]
    axis_name = None
    if isinstance(eff, pd.DataFrame) and len(eff):
        axis_name = str(eff.iloc[0]["axis"])
    if axis_name and axis_name in rdf.columns and f"{focus_head}_mse" in rdf.columns:
        mask = recon_analysis.tail_mask(rdf[f"{focus_head}_mse"].to_numpy(dtype=float), frac)
        v = rdf[axis_name].to_numpy(dtype=float)
        for sel, label, color, lw in (
                (~mask, "the other %.0f%%" % (100 * (1 - frac)), "#999999", 1.3),
                (mask, "the worst %.0f%%" % (100 * frac), _accent(), 2.2)):
            vals = np.sort(v[sel & np.isfinite(v)])
            if len(vals) < 2:
                continue
            ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals), color=color, lw=lw,
                    label="%s (n=%d)" % (label, len(vals)))
        row = eff.iloc[0]
        ax.set_xlabel(LIFT_AXIS_LABELS.get(axis_name, axis_name), fontsize=8)
        ax.set_ylabel("fraction of samples at or below", fontsize=8)
        ax.set_title("The property that separates them best — dataset `%s`\n%s"
                     % (focus, textwrap.fill(
                         "median %.3g in the tail against %.3g in the rest "
                         "(%+.1f interquartile ranges apart, KS %.2f)"
                         % (row["tail_median"], row["rest_median"], row["effect"],
                            row["ks"]), 52)), fontsize=8.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.6, loc="lower right", framealpha=0.9)
    else:
        ax.axis("off")

    # ── What the worst reconstructions look like ─────────────────────────────
    ax = axes[len(heads) + 3]
    worst = (focus_r.get("_worst") or {}).get(focus_head)
    if worst is not None and len(worst.get("mse", [])):
        order = np.argsort(worst["mse"])[::-1][:3]
        # A fixed offset works only if the traces happen to be unit-scale; the worst
        # reconstructions are exactly the ones that are not, and they overlapped into an
        # unreadable band. Space them by the range actually being drawn.
        drawn = np.concatenate([worst["target"][order], worst["pred"][order]])
        step = 1.25 * float(np.ptp(drawn)) if np.ptp(drawn) > 0 else 1.0
        for rank, idx in enumerate(order):
            off = rank * step
            ax.plot(worst["target"][idx] + off, color="#000000", lw=1.1,
                    label="target" if rank == 0 else None)
            ax.plot(worst["pred"][idx] + off, color=_HEAD_COLOR[focus_head],
                    ls=_HEAD_LS[focus_head], lw=1.3,
                    label=PATHWAY_SHORT[focus_head] if rank == 0 else None)
            ax.text(2, off + 0.42 * step, "MSE %.3g" % worst["mse"][idx], fontsize=6.6,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.0))
        ax.set_xlabel("signal bin (0-%d)" % (worst["target"].shape[1] - 1), fontsize=8)
        ax.set_ylabel("amplitude (traces offset for legibility)", fontsize=8)
        ax.set_title("The three worst reconstructions — dataset `%s`\n(%s head, from the "
                     "whole split, not the figure subsample)"
                     % (focus, PATHWAY_SHORT[focus_head]),
                     fontsize=8.5)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.6, loc="upper right", framealpha=0.9)
    else:
        ax.axis("off")

    dropped = _tail_of(focus_r).get("dropped_axes") or []
    note = (" Opened up below: dataset `%s`, %s head - the most concentrated tail (all "
            "datasets: recon_tail_lift.csv)." % (focus, PATHWAY_SHORT[focus_head]))
    if dropped:
        note += " Dropped as degenerate: %s." % ", ".join(dropped)
    _suptitle(fig, "recon_failure_anatomy", _run_note(by_alias) + note, width=150)
    _tight(fig, reserve_in=0.7)
    _footnote(fig, "recon_failure_anatomy")

    # Every dataset's lift table in one file, so the tail can be chased past the top ten.
    frames = []
    for a in aliases:
        t = by_alias[a].get("tail_lift_df")
        if isinstance(t, pd.DataFrame) and len(t):
            frames.append(t.assign(dataset=a,
                                   head=_tail_of(by_alias[a]).get("reference_head")))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(out_dir, "recon_tail_lift.csv"), index=False)

    return [(_caption("recon_failure_anatomy"),
             _save_fig(fig, os.path.join(out_dir, "recon_failure_anatomy.png")))]


# ── F9: where the error lives by component ───────────────────────────────────

def _comp_tables(by_alias: dict) -> dict:
    """{dataset: component_error_table} for every multi-component dataset with one."""
    out = {}
    for a, r in by_alias.items():
        if r.get("component_group") != "multi":
            continue
        df = recon_analysis.component_error_table(r["results_df"], r.get("pathways") or [])
        if isinstance(df, pd.DataFrame) and len(df):
            out[a] = df
    return out


def _comp_ticks(ax, comps) -> None:
    """Component indices as categorical ticks - they are labels, not a numeric scale."""
    ax.set_xticks(np.arange(len(comps)))
    ax.set_xticklabels([("%d" % c) if float(c).is_integer() else ("%g" % c) for c in comps],
                       fontsize=6.5, rotation=90 if len(comps) > 14 else 0)


def _plot_component_error(by_alias: dict, out_dir: str) -> list:
    by_comp = _comp_tables(by_alias)
    if not by_comp:
        return []
    heads = _heads_of(by_alias)
    aliases = [a for a in _ordered_aliases(by_alias) if a in by_comp]
    focus, focus_head, focus_lift = recon_analysis.component_focus(by_comp, heads)
    if focus is None:
        return []
    fdf, focus_r = by_comp[focus], by_alias[focus]

    fig, axes = _flow_axes(len(aliases) + 4, 4.8, 3.8,
                           ncols_screen=max(2, min(len(aliases) + 1, 3)))

    # ── Median error per component, one panel per dataset ────────────────────
    for i, a in enumerate(aliases):
        ax, df, r = axes[i], by_comp[a], by_alias[a]
        comps = df["comp"].to_list()
        x = np.arange(len(comps), dtype=float)
        for k in heads:
            col = f"{k}_mse_median"
            if col not in df.columns:
                continue
            ax.plot(x, df[col], marker=_HEAD_MARKER[k], ms=4, lw=1.2, ls=_HEAD_LS[k],
                    color=_HEAD_COLOR[k], label=PATHWAY_SHORT[k])
        ref = k if (k := focus_head) in heads else heads[0]
        if f"{ref}_mse_q25" in df.columns:
            ax.fill_between(x, df[f"{ref}_mse_q25"], df[f"{ref}_mse_q75"],
                            color=_HEAD_COLOR[ref], alpha=0.15, lw=0,
                            label="%s interquartile range" % PATHWAY_SHORT[ref])
        med = float(r["results_df"][f"{ref}_mse"].median())
        ax.axhline(med, color=_accent(), lw=1.3, ls="-.",
                   label="this dataset's median (%.3g)" % med)
        # Name the components that stand out - the whole point of the panel is that they
        # are identifiable, not that the line has a bump somewhere.
        lift_col = f"{ref}_budget_lift"
        if lift_col in df.columns:
            # Offenders often come in adjacent pairs (sampled_data's comp 26 and 29), and
            # two call-outs at the same height overprint each other into mush - stagger
            # consecutive ones.
            marked = 0
            for xi, (c, lift, val) in enumerate(zip(comps, df[lift_col],
                                                    df[f"{ref}_mse_median"])):
                if not (np.isfinite(lift) and lift >= _COMP_LIFT_MARK):
                    continue
                dy = 12 + 20 * (marked % 2)
                marked += 1
                # Centred text on the first or last component runs off the panel, and the
                # offenders are often exactly there.
                near_end = xi / max(len(comps) - 1, 1)
                ha = "left" if near_end < 0.12 else ("right" if near_end > 0.88
                                                     else "center")
                dx = {"left": -6, "right": 6, "center": 0}[ha]
                ax.annotate("comp %s\n%.0fx its share" % (
                    ("%d" % c) if float(c).is_integer() else ("%g" % c), lift),
                    (xi, val), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=6.4, color=_accent(), fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=_accent(), lw=0.6,
                                    shrinkA=0, shrinkB=3))
        ax.set_yscale("log")
        _comp_ticks(ax, comps)
        ax.set_xlabel("component index (a label, not a scale)", fontsize=8)
        ax.set_ylabel("median MSE  (↓ better)", fontsize=8)
        ax.set_title("%s\n%s" % (_ds_label(r), "%d components" % len(comps)), fontsize=9)
        ax.grid(True, alpha=0.3, axis="y", which="both")
        # The call-outs sit 12pt above their marker and the legend takes the top-left, so
        # a log axis needs real headroom or both get clipped.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 25)
        ax.legend(fontsize=6.2, loc="upper left", framealpha=0.9, ncol=2)

    # ── Error budget: share of the dataset's error against share of its samples ──
    ax = axes[len(aliases)]
    share_col, lift_col = f"{focus_head}_error_share", f"{focus_head}_budget_lift"
    top = fdf.sort_values(share_col, ascending=False).head(12)
    x = np.arange(len(top), dtype=float)
    ax.bar(x, 100 * top[share_col], 0.72, color=_HEAD_COLOR[focus_head], alpha=0.9,
           hatch=_HEAD_HATCH[focus_head], edgecolor="white", linewidth=0.6,
           label="share of the dataset's total error")
    ax.plot(x, 100 * top["sample_share"], ls="none", marker="_", ms=16, mew=2.4,
            color=_accent(), label="share of its samples")
    for xi, (sh, lift) in enumerate(zip(top[share_col], top[lift_col])):
        # Only components carrying their weight or more get a ratio: "0x" under every
        # small bar was rounding noise pretending to be a measurement.
        if np.isfinite(lift) and lift >= 1.0:
            ax.text(xi, 100 * sh, "%.0fx" % lift if lift >= 10 else "%.1fx" % lift,
                    ha="center", va="bottom", fontsize=6.6)
    _comp_ticks(ax, top["comp"].to_list())
    ax.set_xlabel("component index", fontsize=8)
    ax.set_ylabel("% of the dataset  (↓ better for error)", fontsize=8)
    ax.set_title("Where the error budget goes — dataset `%s`, %s head\n"
                 "(bar above the dash = carrying more than its weight; xN is the ratio)"
                 % (focus, PATHWAY_SHORT[focus_head]), fontsize=8.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=6.4, loc="upper right", framealpha=0.9)

    # ── Spread inside each component, not just its median ────────────────────
    ax = axes[len(aliases) + 1]
    rdf = focus_r["results_df"]
    order = fdf.sort_values(share_col, ascending=False)["comp"].to_list()
    picked = order[:4] + order[len(order) // 2:len(order) // 2 + 2]
    picked = list(dict.fromkeys(picked))
    data, labels = [], []
    for c in picked:
        v = rdf.loc[rdf["comp"] == c, f"{focus_head}_mse"].to_numpy(dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size >= 5:
            data.append(v)
            labels.append(c)
    if data:
        bp = ax.boxplot(data, positions=np.arange(len(data)), widths=0.6, showfliers=False,
                        patch_artist=True)
        for j, box in enumerate(bp["boxes"]):
            hot = np.isfinite(fdf.set_index("comp").loc[labels[j], lift_col]) and \
                  fdf.set_index("comp").loc[labels[j], lift_col] >= _COMP_LIFT_MARK
            box.set_facecolor(_accent() if hot else "#bbbbbb")
            box.set_alpha(0.75)
        for part in ("medians", "whiskers", "caps"):
            for ln in bp[part]:
                ln.set_color("#000000")
        ax.set_yscale("log")
        _comp_ticks(ax, labels)
        ax.set_xlabel("component index (worst first, then typical ones)", fontsize=8)
        ax.set_ylabel("per-sample MSE  (↓ better)", fontsize=8)
        ax.set_title("Is the whole component bad, or a few of its samples?\n"
                     "dataset `%s`, %s head" % (focus, PATHWAY_SHORT[focus_head]),
                     fontsize=8.5)
        ax.grid(True, alpha=0.3, axis="y", which="both")
    else:
        ax.axis("off")

    # ── What those components look like ──────────────────────────────────────
    ax = axes[len(aliases) + 2]
    arrays = focus_r.get("_arrays") or {}
    kept_idx = arrays.get("index")
    drawn = False
    if kept_idx is not None and len(kept_idx) and focus_head in (arrays.get("preds") or {}):
        comp_of_kept = rdf["comp"].to_numpy()[np.asarray(kept_idx, dtype=int)]
        mse_of_kept = rdf[f"{focus_head}_mse"].to_numpy()[np.asarray(kept_idx, dtype=int)]
        show = (order[:2] + [order[len(order) // 2]])[:3]
        step = None
        for rank, c in enumerate(show):
            where = np.flatnonzero(comp_of_kept == c)
            if not where.size:
                continue
            # The sample whose error is typical FOR THAT COMPONENT - a worst-case trace
            # would say what the tail looks like, and the tail figure already does that.
            pick = where[np.argmin(np.abs(mse_of_kept[where]
                                          - np.median(mse_of_kept[where])))]
            tgt, prd = arrays["target"][pick], arrays["preds"][focus_head][pick]
            if step is None:
                step = 1.55 * float(np.ptp(np.concatenate([tgt, prd])) or 1.0)
            off = rank * step
            ax.plot(tgt + off, color="#000000", lw=1.0,
                    label="target" if not drawn else None)
            ax.plot(prd + off, color=_HEAD_COLOR[focus_head], ls=_HEAD_LS[focus_head],
                    lw=1.3, label=PATHWAY_SHORT[focus_head] if not drawn else None)
            row = fdf.loc[fdf["comp"] == c]
            sig = ""
            if len(row):
                row = row.iloc[0]
                bits = []
                for col, fmt in ((f"{focus_head}_r2_median", "R\u00b2 %.1f"),
                                 (f"{focus_head}_amp_ratio_median", "amplitude %.1fx"),
                                 (f"{focus_head}_pearson_median", "r %.2f")):
                    if col in row.index and np.isfinite(row[col]):
                        bits.append(fmt % row[col])
                if bits:
                    sig = "\n" + ", ".join(bits)
            ax.text(2, off + 0.40 * step, "comp %s — median MSE %.3g%s"
                    % (("%d" % c) if float(c).is_integer() else ("%g" % c),
                       float(np.median(mse_of_kept[where])), sig),
                    fontsize=6.4, bbox=dict(fc="white", ec="none", alpha=0.78, pad=1.0))
            drawn = True
    if drawn:
        if step:
            ax.set_ylim(top=(len(show) - 1) * step + 0.95 * step)
        ax.set_xlabel("signal bin", fontsize=8)
        ax.set_ylabel("amplitude (traces offset)", fontsize=8)
        ax.set_title("What the worst components look like — dataset `%s`\n"
                     "(a typical sample of each, not its worst)" % focus, fontsize=8.5)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6.6, loc="upper right", framealpha=0.9)
    else:
        ax.axis("off")

    # ── What separates them: error against a per-component signal property ───
    ax = axes[len(aliases) + 3]
    axis_name = _comp_discriminator(fdf, focus_head)
    if axis_name:
        lifts = fdf[lift_col].to_numpy(dtype=float)
        hot = np.isfinite(lifts) & (lifts >= _COMP_LIFT_MARK)
        ax.scatter(fdf.loc[~hot, axis_name], fdf.loc[~hot, f"{focus_head}_mse_median"],
                   s=34, color="#999999", label="other components")
        ax.scatter(fdf.loc[hot, axis_name], fdf.loc[hot, f"{focus_head}_mse_median"],
                   s=64, color=_accent(), marker="D",
                   label="carrying %gx their share or more" % _COMP_LIFT_MARK)
        for _, row in fdf.iterrows():
            ax.annotate(("%d" % row["comp"]) if float(row["comp"]).is_integer()
                        else ("%g" % row["comp"]),
                        (row[axis_name], row[f"{focus_head}_mse_median"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6.2)
        ax.set_yscale("log")
        ax.set_xlabel(LIFT_AXIS_LABELS.get(axis_name, axis_name), fontsize=8)
        ax.set_ylabel("component's median MSE  (↓ better)", fontsize=8)
        ax.set_title("What makes the bad components different — dataset `%s`\n"
                     "(one point per component)" % focus, fontsize=8.5)
        ax.margins(0.16)
        ax.grid(True, alpha=0.3, which="both")
        # The bad components are the high-error ones, so the top of the panel is theirs.
        ax.legend(fontsize=6.4, loc="lower right", framealpha=0.9)
    else:
        ax.axis("off")

    note = (" Opened up below: dataset `%s`, %s head — the most uneven component budget "
            "(worst component carries %.0fx its share of the samples)."
            % (focus, PATHWAY_SHORT[focus_head], focus_lift))
    _suptitle(fig, "recon_component_error", _run_note(by_alias) + note, width=150)
    _tight(fig, reserve_in=0.7)
    _footnote(fig, "recon_component_error")

    frames = [df.assign(dataset=a) for a, df in by_comp.items()]
    pd.concat(frames, ignore_index=True).to_csv(
        os.path.join(out_dir, "recon_component_error.csv"), index=False)

    return [(_caption("recon_component_error"),
             _save_fig(fig, os.path.join(out_dir, "recon_component_error.png")))]


_SUMMARY_FIGURES = [
    ("recon_error_distribution",     _plot_error_distribution),
    ("recon_summary_heatmap",        _plot_summary_heatmap),
    ("recon_skill_vs_baseline",      _plot_skill_vs_baseline),
    ("recon_position_profile",       _plot_position_profile),
    ("recon_amplitude_calibration",  _plot_amplitude_calibration),
    ("recon_spectral_fidelity",      _plot_spectral_fidelity),
    ("recon_reference_ladder",       _plot_reference_ladder),
    ("recon_failure_anatomy",        _plot_failure_anatomy),
    ("recon_component_error",        _plot_component_error),
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
      fe    the target resampled through 37 points plus modest noise — a faithful
            reconstruction whose effective resolution is therefore known in advance
      proj  amplitude compressed to 50% — should show up in amp_ratio and in F4's slope
      tr    each sample's own mean, i.e. the trivial baseline — R2 must land at 0

    On the multi-component datasets one component index is made deliberately terrible, so
    the failure-anatomy figure has a planted answer to find: its tail should be that
    component and nothing else, at a lift equal to the number of components.
    """
    from .evaluations.signal_reconstruction import (
        _per_sample_metrics, _profiles, _spectrum_table, _stratified_table,
        _summary_table)
    from .recon_analysis import (DEFAULT_REF_LADDER, FE_BOTTLENECK_K,
                                 effective_resolution, reference_operators,
                                 tail_analysis)
    from .signal_features import STRATIFIER_ORDER, compute_signal_features

    rng = np.random.default_rng(seed)
    ref_ops = reference_operators(245, DEFAULT_REF_LADDER)
    ref_ks = sorted(ref_ops["interp"])
    # The component the planted failure lives on, and how much worse it is made.
    bad_comp, bad_scale = 3, 6.0
    specs = [("sanity", "single_channel_1k", 120, False),
             ("in_dist", "single_channel_10k", 200, False),
             ("multi_ch", "multi_channel", 200, True),
             ("samples", "sampled_data", 160, True)]

    out = {}
    for alias, subset, n, multi in specs:
        L = 245
        base = np.cumsum(rng.normal(0, 1, (n, L)), axis=1)
        base += 3.0 * np.exp(-0.5 * ((np.arange(L) - rng.integers(40, 200, (n, 1))) / 6.0) ** 2)
        t = (base - base.mean(1, keepdims=True)) / base.std(1, keepdims=True)
        # Real layer-normed targets do NOT come back at std exactly 1: measured across the
        # four datasets they span 0.9837-0.99999. Reproducing that here is what makes the
        # figures' "has this axis collapsed?" tests face the case they actually meet - an
        # exactly-constant fixture let an absolute-range test pass that a relative one had
        # to catch.
        t = (t * rng.uniform(0.984, 1.0, (n, 1))).astype(np.float32)
        # Multi-component data is harder here, mirroring the real generalization gap.
        noise = 0.22 if multi else 0.10
        # fe is built FROM a reference rung, so its effective resolution is known before
        # the figure computes it: the added noise pushes it a little below 37, and the
        # heavier multi-component noise pushes it further, which is the ordering the
        # reference-ladder figure has to reproduce.
        fe_base = (t.astype(np.float64) @ ref_ops["interp"][37].T)
        preds = {
            "fe":   (fe_base + rng.normal(0, noise, t.shape)).astype(np.float32),
            "proj": (0.5 * t + rng.normal(0, noise, t.shape)).astype(np.float32),
            "tr":   np.repeat(t.mean(1, keepdims=True), L, axis=1).astype(np.float32),
        }
        heads = list(preds)
        comps = np.array([i % 6 for i in range(n)]) if multi else np.zeros(n, int)
        if multi:
            hit = comps == bad_comp
            for k in ("fe", "proj"):
                preds[k] = preds[k] + (bad_scale * rng.normal(0, 1, t.shape)
                                       * hit[:, None]).astype(np.float32)

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
        t64 = t.astype(np.float64)
        ref_interp = np.stack([((t64 @ ref_ops["interp"][k].T - t64) ** 2).mean(1)
                               for k in ref_ks], axis=1)
        ref_lowpass = np.stack([((t64 @ ref_ops["lowpass"][k].T - t64) ** 2).mean(1)
                                for k in ref_ks], axis=1)
        for j, k in enumerate(ref_ks):
            rows[f"ref_interp{k}_mse"] = ref_interp[:, j]
        for k in heads:
            rows[f"{k}_k_eff"] = effective_resolution(
                rows[f"{k}_mse"], ref_ks, ref_interp)["k_eff"]

        rdf = pd.concat([pd.DataFrame(rows), compute_signal_features(t)], axis=1)
        if multi:
            rdf["dataset_id"] = 2
            rdf["comp"] = comps
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
        # The fixture keeps every sample, so the figure subsample is the identity map.
        r["_arrays"] = {"target": t, "preds": preds, "index": np.arange(n),
                        "n_kept": n, "n_total": n}
        r["reference"] = {
            "ladder": list(ref_ks), "bottleneck_k": FE_BOTTLENECK_K, "length": L,
            "sample_spacing": L / float(FE_BOTTLENECK_K),
            "interp_mse_median": np.median(ref_interp, axis=0),
            "lowpass_mse_median": np.median(ref_lowpass, axis=0),
            "k_eff_median": {k: float(np.median(rdf[f"{k}_k_eff"])) for k in heads},
            "peak_fwhm_median": 14.0, "narrow_peak_share": 0.02, "narrow_peak_n": n,
        }
        tail = tail_analysis(rdf, heads, frac=0.05)
        r["tail_lift_df"] = tail.pop("lift_df")
        r["tail_effect_df"] = tail.pop("effect_df")
        r["tail"] = tail
        r["_worst"] = {}
        for k in heads:
            order = np.argsort(rdf[f"{k}_mse"].to_numpy())[::-1][:16]
            r["_worst"][k] = {"index": order,
                              "mse": rdf[f"{k}_mse"].to_numpy()[order],
                              "target": t[order], "pred": preds[k][order]}
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
    # Reference ladder: a head built from a known rung has to come back at that rung.
    from .recon_analysis import (DEFAULT_REF_LADDER, effective_resolution,
                                 reference_operators, tail_mask)
    ops = reference_operators(245, DEFAULT_REF_LADDER)
    ks = sorted(ops["interp"])
    tgt = by_alias["in_dist"]["_arrays"]["target"].astype(np.float64)
    ref = np.stack([((tgt @ ops["interp"][k].T - tgt) ** 2).mean(1) for k in ks], axis=1)
    exact = {}
    for k_true in (17, 31, 47):
        m = ((tgt @ ops["interp"][k_true].T - tgt) ** 2).mean(1)
        exact[k_true] = float(np.median(effective_resolution(m, ks, ref)["k_eff"]))
    k_single = by_alias["in_dist"]["reference"]["k_eff_median"]["fe"]
    k_multi = by_alias["multi_ch"]["reference"]["k_eff_median"]["fe"]

    # Failure anatomy: the planted bad component must be what the tail is made of.
    lift = by_alias["multi_ch"]["tail_lift_df"]
    top = lift.iloc[0] if len(lift) else None

    checks += [
        ("a head equal to a reference rung must report that rung as its resolution",
         all(abs(exact[k] - k) < 0.5 for k in exact)),
        ("fe is built from rung 37 plus noise, so its resolution must sit below it",
         20.0 < k_single < 37.0),
        ("noisier multi-component data must report a lower effective resolution",
         k_multi < k_single),
        ("the mean-predictor head must fall off the coarse end of the ladder",
         by_alias["in_dist"]["reference"]["k_eff_median"]["tr"] <= ks[0] + 1e-6),
        ("the planted bad component must be the top of the tail lift table",
         top is not None and top["axis"] == "comp" and str(top["bin_label"]) == "3"),
        ("that component must account for the whole tail",
         top is not None and top["tail_share"] > 0.95),
        ("its lift must equal the number of components (6)",
         top is not None and abs(top["lift"] - 6.0) < 0.5),
        ("the mean-predictor head fails evenly, so its tail must not be concentrated",
         abs(by_alias["multi_ch"]["tail"]["per_head"]["tr"]["share"] - 0.05) < 0.03),
        ("contrast must be dropped as degenerate when targets are layer-normed",
         "contrast" in (by_alias["in_dist"]["tail"].get("dropped_axes") or [])),
        # Target std spans ~1.6% here, as it does on real data. That is not usable as a
        # scatter axis, and an absolute-range test called it usable - which is how the
        # amplitude figure came to draw a vertical strip on a real run.
        ("an all-but-constant target std must count as collapsed",
         all(recon_analysis.contrast_is_collapsed(
             by_alias[a]["_arrays"]["target"].std(axis=1)) for a in by_alias)),
    ]

    # Component budget: the planted bad component must dominate its dataset's error, and
    # the share of error must be what says so - its sample count never changes.
    cdf = recon_analysis.component_error_table(
        by_alias["multi_ch"]["results_df"], ["fe", "proj", "tr"])
    worst = (cdf.sort_values("proj_error_share", ascending=False).iloc[0]
             if len(cdf) else None)
    flat = recon_analysis.component_error_table(
        by_alias["in_dist"]["results_df"], ["fe", "proj", "tr"])
    checks += [
        ("the planted bad component must carry most of its dataset's error budget",
         worst is not None and worst["comp"] == 3 and worst["proj_error_share"] > 0.5),
        ("and must be flagged as carrying well over its share of the samples",
         worst is not None and worst["proj_budget_lift"] > _COMP_LIFT_MARK),
        ("the mean-predictor head must spread its error evenly across components",
         worst is not None
         and abs(float(cdf["tr_budget_lift"].max()) - 1.0) < 0.35),
        ("single-component data has no component axis, so no table",
         isinstance(flat, pd.DataFrame) and flat.empty),
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
