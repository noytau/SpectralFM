"""
Per-sample signal descriptors used as stratification axes for reconstruction plots.
No fairseq dependency (numpy + scipy only).

Ported from `code/eval_metrics.py:compute_signal_processing_features` ('peaks',
'centroid' and 'moments' modes), reshaped from "feature matrix for a probe" into
"named scalar per sample for grouping". The original lives outside the `code/eval/`
package, which must stay independently importable, so the relevant maths is copied
here rather than imported across the package boundary.

Every descriptor is computed on the *target* signal — the same tensor the decoder was
asked to reproduce (post per-sample layer_norm when the checkpoint recorded
`normalize`), so error-vs-property plots relate like to like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Axis key → human-readable label used verbatim in plot titles and axis labels.
# Keep these in sync with the stratification figures; outsiders read these strings.
FEATURE_LABELS = {
    "contrast":        "Contrast — std. dev. of the signal",
    "peak_count":      "Number of local maxima (scipy find_peaks)",
    "peak_prominence": "Peak prominence — (max − median) / std",
    "centroid":        "Spectral centroid — intensity-weighted mean bin index",
    "peak_position":   "Peak position — argmax bin index (0–244)",
    "bandwidth":       "Spectral bandwidth — intensity-weighted std. of bin index",
    "baseline":        "Baseline level — median value",
}

# Axes offered as stratifiers, in the order they should be plotted.
STRATIFIER_ORDER = ["contrast", "peak_count", "peak_prominence", "centroid", "peak_position"]


def compute_signal_features(signals: np.ndarray) -> pd.DataFrame:
    """
    Describe each signal with the scalars used to stratify reconstruction error.

    signals: [N, L] float array (the reconstruction target).
    Returns a DataFrame with one row per signal and the columns named in
    FEATURE_LABELS. Row order matches `signals`.
    """
    from scipy.signal import find_peaks

    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"compute_signal_features expects [N, L], got {x.shape}")
    n, length = x.shape

    std = x.std(axis=1)
    median = np.median(x, axis=1)
    # Guard against constant signals: several multi_channel components are monotone
    # or flat, which would otherwise divide by zero in the prominence ratio.
    safe_std = np.where(std > 1e-12, std, np.nan)

    bins = np.arange(length, dtype=np.float64)
    magnitudes = np.abs(x) + 1e-12
    total = magnitudes.sum(axis=1)
    centroid = (magnitudes * bins).sum(axis=1) / total
    bandwidth = np.sqrt(
        (((bins - centroid[:, None]) ** 2) * magnitudes).sum(axis=1) / total
    )

    peak_count = np.empty(n, dtype=np.int64)
    for i in range(n):
        peaks, _ = find_peaks(x[i])
        peak_count[i] = len(peaks)

    return pd.DataFrame({
        "contrast":        std,
        "peak_count":      peak_count,
        "peak_prominence": (x.max(axis=1) - median) / safe_std,
        "centroid":        centroid,
        "bandwidth":       bandwidth,
        "peak_position":   x.argmax(axis=1),
        "baseline":        median,
    })


def quantile_bins(values: np.ndarray, n_bins: int = 5) -> tuple:
    """
    Assign each value to a quantile bin, for "median error vs signal property" plots.

    Returns (bin_index [N] with -1 for NaN inputs, list of edge-label strings).
    Falls back to fewer bins when the distribution is too degenerate to split — many
    multi_channel components are constant, so `contrast` can be near-single-valued.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    out = np.full(len(v), -1, dtype=np.int64)
    if ok.sum() < n_bins:
        return out, []

    # An effectively-constant axis must be dropped, not split into bins that all carry
    # the same label. This is not hypothetical: `normalize=True` layer-norms every target
    # to unit variance, so `contrast` is 1.0 for every sample up to float32 noise, and
    # naive quantiles would still produce five distinct-but-identical-looking edges.
    span = float(v[ok].max() - v[ok].min())
    if span <= 1e-6 * max(abs(float(np.median(v[ok]))), 1.0):
        return out, []

    edges = np.unique(np.quantile(v[ok], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:                      # not enough distinct values to bin
        return out, []
    # right=False so the lowest bin includes the minimum; clip the top edge in.
    idx = np.digitize(v[ok], edges[1:-1], right=False)
    out[ok] = idx

    # Bin labels are read by people, so escalate precision until the edges are
    # actually distinguishable — quantiles of a narrow distribution otherwise all
    # render as the same string (e.g. every bin labelled "1-1").
    for precision in (3, 4, 5, 6, 8, 10, 12):
        labels = [f"{edges[i]:.{precision}g}–{edges[i + 1]:.{precision}g}"
                  for i in range(len(edges) - 1)]
        if len(set(labels)) == len(labels):
            break
    else:
        # Edges this close cannot be told apart in decimal at any sane width - label the
        # bins by rank instead of printing five identical-looking ranges.
        labels = [f"q{i + 1}" for i in range(len(edges) - 1)]
    return out, labels
