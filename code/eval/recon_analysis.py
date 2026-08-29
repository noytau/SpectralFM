"""
Analysis helpers shared by the reconstruction figures and the generated findings.

Pure numpy/pandas/scipy — no matplotlib, no fairseq. It exists so `recon_plots.py`
(figures) and `findings.py` (prose) compute the same numbers from the same code instead
of one importing the other, and so `evaluations/signal_reconstruction.py` can build the
reference operators without pulling in a plotting stack.

Two families of helper live here:

**Reference reconstructions** (`resample_operator`, `lowpass_operator`,
`effective_resolution`). R² = 0 — a flat line at each sample's own mean — is a very weak
baseline, and every head clears it comfortably, which makes "the head is doing real work"
an easy and uninformative conclusion. A fair reference is one that discards the same
amount of information the model's bottleneck does: the conv feature extractor compresses
245 bins to 47 timesteps, so linear interpolation through 47 evenly spaced points of the
target is a reconstruction at the same temporal rate. Comparing a head against a ladder of
such references converts its MSE into an *effective resolution* — the number of
independent samples of the signal it actually delivers.

The interpolant is an ORACLE at those points: it reads true target values the model never
sees, and the real bottleneck is 47 timesteps x 512/768 channels rather than 47 scalars.
So this is a bound on what temporal downsampling alone costs, not on what the model could
achieve. Its use is the negative direction: a head below the reference at its own rate
cannot blame the bottleneck.

**Tail anatomy** (`concentration`, `lorenz_curve`, `lift_table`, `robust_effect`,
`tail_analysis`). Distribution figures stop at "there is a tail". These name it: how much
of the total error the worst few per cent carry, and which component index, source dataset
or signal property that tail is made of.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signal_features import FEATURE_LABELS, quantile_bins

# ── Reference ladder ─────────────────────────────────────────────────────────

# Resolutions to build reference reconstructions at. Spans "much coarser than the
# bottleneck" to "much finer", with 47 included exactly because that is the rate the FE
# conv stack actually delivers. The very coarse rungs are not decoration: multi_channel
# components are close to smooth ramps, so interpolation through nine points already has
# less error than any head, and without rungs below that every head would clamp to the
# floor and the figure would report a bound instead of a measurement.
DEFAULT_REF_LADDER = (3, 5, 7, 9, 13, 17, 25, 31, 37, 47, 61, 81, 121)

# The conv feature extractor maps 245 input bins to 47 timesteps, so 47 is the temporal
# rate every head decodes from. Not a tunable — it is a property of the architecture.
FE_BOTTLENECK_K = 47

_TINY = 1e-30


def interp_matrix(src, dst) -> np.ndarray:
    """
    Linear-interpolation weights as a matrix: `A @ v` interpolates values `v` given at
    positions `src` onto positions `dst`. Reproduces np.interp (including its constant
    clamping outside the source range) but applies to a whole batch in one matmul.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    A = np.zeros((len(dst), n), dtype=np.float64)
    if n == 1:
        A[:, 0] = 1.0
        return A
    j = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, n - 2)
    span = src[j + 1] - src[j]
    w = np.where(span > 0, (dst - src[j]) / np.where(span > 0, span, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0)
    rows = np.arange(len(dst))
    A[rows, j] = 1.0 - w
    A[rows, j + 1] = w
    return A


def resample_operator(length: int, k: int) -> np.ndarray:
    """
    [length, length] operator: keep k evenly spaced samples of the signal, then linearly
    interpolate back to full length. Downsample-then-upsample is linear, so it collapses
    to one matrix applied as `signals @ M.T` — a single matmul per batch instead of a
    per-row np.interp call, which matters at 136k samples x 9 rungs.
    """
    full = np.arange(length, dtype=np.float64)
    grid = np.linspace(0.0, length - 1.0, int(k))
    return interp_matrix(grid, full) @ interp_matrix(full, grid)


def lowpass_operator(length: int, k: int) -> np.ndarray:
    """
    [length, length] operator: keep only the rFFT bins a k-point sampling could carry
    (0 .. k//2) and zero the rest. The information-theoretic sibling of
    `resample_operator` — an ideal band-limit rather than a piecewise-linear fit.

    Neither dominates: band-limiting wins on signals that really are band-limited, while
    interpolation wins on ones with sharp ends, which is why the multi_channel ramps come
    out better under interpolation at fine rates. Drawing both brackets what the rate can
    express rather than resting the comparison on one arbitrary reconstruction rule.
    """
    keep = int(k) // 2 + 1
    spec = np.fft.rfft(np.eye(length), axis=1)
    spec[:, keep:] = 0.0
    return np.fft.irfft(spec, n=length, axis=1).T


def reference_operators(length: int, ladder=DEFAULT_REF_LADDER) -> dict:
    """{'interp': {k: M}, 'lowpass': {k: M}} — built once, reused for every batch."""
    ks = [int(k) for k in ladder if 1 < int(k) <= length]
    return {
        "interp": {k: resample_operator(length, k) for k in ks},
        "lowpass": {k: lowpass_operator(length, k) for k in ks},
    }


def effective_resolution(mse: np.ndarray, ladder, ref_mse: np.ndarray) -> dict:
    """
    Convert a head's per-sample MSE into an effective resolution in signal samples.

    mse      [N]        the head's per-sample MSE
    ladder   [m]        the resolutions the references were computed at
    ref_mse  [N, m]     that sample's own reference MSE at each rung

    Each sample is placed on its OWN reference curve — reference difficulty varies a lot
    between samples, so a shared curve would misplace them. Interpolation is in
    (log k, log MSE), where the curve is close to straight.

    Returns {'k_eff': [N], 'below': [N] bool, 'above': [N] bool} where `below` marks
    samples worse than the coarsest rung and `above` samples better than the finest —
    both clamped in `k_eff`, and reported so a plot can say the value is a bound.
    """
    ks = np.asarray([float(k) for k in ladder], dtype=np.float64)
    r = np.asarray(ref_mse, dtype=np.float64)
    m = np.asarray(mse, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != len(ks) or len(r) != len(m):
        raise ValueError("ref_mse must be [N, len(ladder)] matching mse")

    # Enforce monotonicity along the ladder. Reference error falls as k rises, but a
    # near-constant target can produce ties or float noise that break the inversion.
    r = np.minimum.accumulate(np.maximum(r, _TINY), axis=1)

    lr = -np.log(r)                       # increasing along the ladder
    lm = -np.log(np.maximum(m, _TINY))
    logk = np.log(ks)

    below = lm <= lr[:, 0]
    above = lm >= lr[:, -1]

    j = np.clip((lr < lm[:, None]).sum(axis=1), 1, len(ks) - 1)
    rows = np.arange(len(m))
    x0, x1 = lr[rows, j - 1], lr[rows, j]
    span = x1 - x0
    w = np.clip(np.where(span > 0, (lm - x0) / np.where(span > 0, span, 1.0), 0.0), 0.0, 1.0)
    k_eff = np.exp(logk[j - 1] + w * (logk[j] - logk[j - 1]))
    k_eff = np.clip(k_eff, ks[0], ks[-1])
    return {"k_eff": k_eff, "below": below, "above": above}


def peak_fwhm(signals: np.ndarray) -> np.ndarray:
    """
    Full width at half maximum of each signal's tallest peak, in bins, measured from the
    sample's own median as the floor. Used only to report what share of the population has
    structure narrower than the reference's sample spacing — interpolation flatters smooth
    signals, and a reader needs to know whether that applies here.

    Loops in Python (the contiguous run containing the argmax does not vectorise
    cleanly), so call it on a bounded subsample, not on a whole split.
    """
    out = np.full(len(signals), np.nan)
    for i, row in enumerate(np.asarray(signals, dtype=np.float64)):
        med = np.median(row)
        top = row.max()
        if top - med <= 1e-12:
            continue
        half = med + 0.5 * (top - med)
        a = int(row.argmax())
        lo = a
        while lo > 0 and row[lo - 1] >= half:
            lo -= 1
        hi = a
        while hi < len(row) - 1 and row[hi + 1] >= half:
            hi += 1
        out[i] = hi - lo + 1
    return out


# ── Tail anatomy ─────────────────────────────────────────────────────────────

# Categorical levels the lift chart will rank. Higher than the stratified-median figure's
# cap of 12: that figure draws every bin as an x-position and needs to stay readable,
# while this one ranks levels and shows only the top few, so it can afford `comp` on
# `sampled_data` (28 distinct values) as real categories instead of quantile-blurring the
# very levels the tail is made of.
LIFT_MAX_LEVELS = 40

# Identity-like axes: always categorical, because the level IS the finding ("the tail is
# comp 26 and 29"). Quantile-binning them would blur away the very levels being named.
_CATEGORICAL_AXES = ("comp", "n_comps", "n_comps_in_split", "dataset_id", "worst_comp")

# Count-like axes: categorical only while the cardinality stays chartable, else quantile
# bins. `peak_count` has 33 distinct values on sampled_data, and treating those as levels
# scattered the whole tail across levels too small to pass the support threshold - it hid
# the single strongest discriminator on that dataset (tail median 73 local maxima against
# 8 for the rest). Same cap the stratified-median table uses.
_COUNT_AXES = ("peak_count",)
_COUNT_MAX_LEVELS = 12

# Axis key → label, for axes that are metadata rather than signal descriptors.
LIFT_AXIS_LABELS = dict(FEATURE_LABELS)
LIFT_AXIS_LABELS.update({
    "comp":              "Component index (from the filename)",
    "n_comps":           "Components per spectrum (dataset-wide, after dedup)",
    "n_comps_in_split":  "Components per spectrum present in this split",
    "dataset_id":        "Source dataset id (from the filename)",
    "worst_comp":        "Worst component of the spectrum",
})


def contrast_is_collapsed(values: np.ndarray, tol: float = 2.0) -> bool:
    """
    True when per-sample target variance carries no usable spread.

    `normalize=True` layer-norms every target, forcing var(target) - and therefore the
    `contrast` descriptor - to 1 for every sample. What survives is float noise, and
    because that noise has a very tight IQR it can post a huge standardised effect size
    (measured: -6.6 IQRs on `sampled_data`) while meaning nothing at all. Any axis or
    x-axis built on target variance has to be dropped when this returns True, not
    rescaled. Same robust p99/p1 test the skill and calibration figures already use.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 10:
        return False
    lo, hi = np.percentile(v, [1, 99])
    return bool(lo > 0 and hi / lo < tol)


def tail_mask(values: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask of the worst `frac` of samples by `values` (higher = worse)."""
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    if not finite.any() or not (0.0 < frac < 1.0):
        return np.zeros(len(v), dtype=bool)
    thr = np.quantile(v[finite], 1.0 - frac)
    return finite & (v >= thr)


def gini(values: np.ndarray) -> float:
    """
    Gini coefficient of the error distribution: 0 = every sample contributes equally,
    approaching 1 = one sample carries everything. Reported alongside the worst-x% share
    because the share alone depends on where you cut.
    """
    v = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if len(v) == 0 or v.sum() <= 0:
        return float("nan")
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * v).sum()) / (n * v.sum()) - (n + 1.0) / n)


def concentration(values: np.ndarray, frac: float) -> dict:
    """
    How concentrated the error is. `values` are per-sample MSE; total error is their sum,
    so this is the share of the dataset's whole error budget carried by its worst `frac`.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0 or v.sum() <= 0:
        return {"share": float("nan"), "gini": float("nan"), "n": 0, "n_tail": 0}
    k = max(1, int(round(frac * len(v))))
    worst = np.sort(v)[::-1][:k]
    return {"share": float(worst.sum() / v.sum()), "gini": gini(v),
            "n": int(len(v)), "n_tail": int(k)}


def lorenz_curve(values: np.ndarray, n_points: int = 300) -> tuple:
    """
    (x, y) for the cumulative-error curve: x = share of samples ranked worst-first,
    y = share of total error they carry. The diagonal is a perfectly even distribution.
    Downsampled to `n_points` so a 136k-sample curve stays a small figure.
    """
    v = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])[::-1]
    if len(v) == 0 or v.sum() <= 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cum = np.concatenate([[0.0], np.cumsum(v) / v.sum()])
    x = np.arange(len(cum)) / (len(cum) - 1)
    if len(cum) > n_points:
        pick = np.unique(np.concatenate([
            np.linspace(0, len(cum) - 1, n_points).astype(int), [len(cum) - 1]]))
        return x[pick], cum[pick]
    return x, cum


def robust_effect(tail: np.ndarray, rest: np.ndarray) -> dict:
    """
    Separation between the tail and the rest on one axis.

    `effect` is the median difference in units of the whole population's IQR — NOT a
    ratio of medians. A ratio is meaningless for any axis that crosses zero, and the
    `baseline` descriptor does exactly that (tail median +0.31 against the rest's -0.26),
    which produced a nonsense 3e11 "ratio" the first time this was measured by hand.
    `ks` is the two-sample Kolmogorov-Smirnov statistic, a scale-free companion.
    """
    t = np.asarray(tail, dtype=np.float64)
    r = np.asarray(rest, dtype=np.float64)
    t, r = t[np.isfinite(t)], r[np.isfinite(r)]
    out = {"effect": float("nan"), "ks": float("nan"),
           "tail_median": float("nan"), "rest_median": float("nan"),
           "n_tail": int(len(t)), "n_rest": int(len(r))}
    if len(t) < 2 or len(r) < 2:
        return out
    pop = np.concatenate([t, r])
    iqr = float(np.percentile(pop, 75) - np.percentile(pop, 25))
    out["tail_median"] = float(np.median(t))
    out["rest_median"] = float(np.median(r))
    if iqr > 1e-12:
        out["effect"] = (out["tail_median"] - out["rest_median"]) / iqr
    try:
        from scipy.stats import ks_2samp
        out["ks"] = float(ks_2samp(t, r).statistic)
    except Exception:
        # Fall back to a direct ECDF gap rather than losing the column.
        grid = np.unique(np.concatenate([t, r]))
        ecdf_t = np.searchsorted(np.sort(t), grid, side="right") / len(t)
        ecdf_r = np.searchsorted(np.sort(r), grid, side="right") / len(r)
        out["ks"] = float(np.abs(ecdf_t - ecdf_r).max())
    return out


def _levels(rdf: pd.DataFrame, axis: str, n_bins: int = 5):
    """(level index per row, level labels) for one axis, categorical or quantile-binned."""
    if axis not in rdf.columns:
        return None, []
    col = rdf[axis]
    if col.notna().sum() < 2:
        return None, []
    n_levels = col.nunique(dropna=True)
    categorical = ((axis in _CATEGORICAL_AXES and n_levels <= LIFT_MAX_LEVELS)
                   or (axis in _COUNT_AXES and n_levels <= _COUNT_MAX_LEVELS))
    if categorical:
        vals = sorted(col.dropna().unique(), key=lambda v: (str(type(v)), v))
        lookup = {v: i for i, v in enumerate(vals)}
        idx = col.map(lookup).to_numpy(dtype=float)
        # Component and dataset ids arrive as floats from a merge with missing rows, and
        # "comp = 26.0" is noise on a chart. Fall back to str() for anything non-numeric
        # rather than letting a string level raise.
        def fmt(v):
            try:
                f = float(v)
            except (TypeError, ValueError):
                return str(v)
            return str(int(round(f))) if abs(f - round(f)) < 1e-9 else "%g" % f
        return idx, [fmt(v) for v in vals]
    return quantile_bins(col.to_numpy(dtype=float), n_bins=n_bins)


def lift_table(rdf: pd.DataFrame, mask: np.ndarray, axes: list,
               min_tail: int = 5, min_pop: int = 20, n_bins: int = 5) -> tuple:
    """
    How over-represented each level is in the tail: lift = P(level | tail) / P(level).

    A level needs `min_tail` tail rows and `min_pop` population rows to get a row here —
    otherwise a single sample in a rare category posts an enormous, meaningless lift. What
    the thresholds exclude is returned as `pooled`, not dropped silently: a chart that
    quietly hides half the tail reads as though it explained all of it.
    """
    mask = np.asarray(mask, dtype=bool)
    rows, pooled_levels = [], 0
    pooled_by_axis = {}
    n_tail_total = int(mask.sum())
    for axis in axes:
        idx, labels = _levels(rdf, axis, n_bins=n_bins)
        if idx is None or not labels:
            continue
        for b, label in enumerate(labels):
            in_level = idx == b
            n_pop = int(np.nansum(in_level))
            n_tail = int(np.nansum(in_level & mask))
            if n_tail < min_tail or n_pop < min_pop:
                if n_tail:
                    pooled_levels += 1
                    pooled_by_axis[axis] = pooled_by_axis.get(axis, 0) + n_tail
                continue
            base = n_pop / len(rdf)
            # A level that IS the dataset carries no information: its lift is 1.0 by
            # construction and it only crowds out the levels that mean something.
            if base > 0.98:
                continue
            share = n_tail / max(n_tail_total, 1)
            rows.append({
                "axis": axis, "axis_label": LIFT_AXIS_LABELS.get(axis, axis),
                "bin": b, "bin_label": label,
                "n_pop": n_pop, "n_tail": n_tail,
                "base_rate": base, "tail_share": share,
                "lift": share / base if base > 0 else float("nan"),
                "tail_rate": n_tail / n_pop,
            })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("lift", ascending=False).reset_index(drop=True)
    # Each axis partitions the tail on its own, so excluded counts are reported per axis
    # and summarised by the worst single axis. Summing across axes would have been
    # nonsense - it can exceed the tail size, which is exactly what it first did.
    worst = max(pooled_by_axis.items(), key=lambda kv: kv[1], default=(None, 0))
    return df, {"levels": pooled_levels, "n_tail_total": n_tail_total,
                "by_axis": pooled_by_axis,
                "worst_axis": worst[0], "worst_axis_n_tail": int(worst[1]),
                "n_bins": n_bins, "quantile_lift_ceiling": float(n_bins)}


def component_error_table(rdf: pd.DataFrame, pathways: list,
                          min_n: int = 5) -> pd.DataFrame:
    """
    Per component index: how many samples it contributes and how much of the dataset's
    whole error budget it carries.

    The share of ERROR is the number that matters, and it is not the median. A component
    can sit at an unremarkable median and still dominate a dataset, or - as sampled_data's
    comp 26 and 29 do - sit at a median 87x the dataset's and carry 84% of its total
    squared error from 6.5% of its samples. `budget_lift` is that share divided by the
    component's share of samples: 1.0 means it carries its own weight, 13 means it carries
    thirteen times it.

    Returns one row per component, plus the median of every signal descriptor present, so
    a plot can put the offenders' error next to what makes them different.
    """
    if "comp" not in rdf.columns or not len(rdf):
        return pd.DataFrame()
    heads = [k for k in pathways if f"{k}_mse" in rdf.columns]
    if not heads:
        return pd.DataFrame()

    feats = [c for c in FEATURE_LABELS if c in rdf.columns]
    rows = []
    n_total = len(rdf)
    totals = {k: float(rdf[f"{k}_mse"].sum()) for k in heads}
    for comp, g in rdf.groupby("comp"):
        if len(g) < min_n:
            continue
        row = {"comp": comp, "n": int(len(g)), "sample_share": len(g) / n_total}
        for k in heads:
            v = g[f"{k}_mse"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if not v.size:
                continue
            row[f"{k}_mse_median"] = float(np.median(v))
            row[f"{k}_mse_q25"] = float(np.percentile(v, 25))
            row[f"{k}_mse_q75"] = float(np.percentile(v, 75))
            share = float(v.sum() / totals[k]) if totals[k] > 0 else float("nan")
            row[f"{k}_error_share"] = share
            row[f"{k}_budget_lift"] = (share / row["sample_share"]
                                       if row["sample_share"] > 0 else float("nan"))
        for f in feats:
            row[f] = float(g[f].median())
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("comp").reset_index(drop=True) if len(df) else df


def component_focus(by_comp: dict, pathways: list) -> tuple:
    """
    (dataset, head) whose component error budget is most unevenly distributed.

    `by_comp` is {dataset: component_error_table(...)}. A dataset whose worst component
    carries a fair share has nothing to open up; the one with the highest budget lift is
    the one worth spending panels on.
    """
    best, best_lift = (None, None), -1.0
    for ds, df in by_comp.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for k in pathways:
            col = f"{k}_budget_lift"
            if col not in df.columns:
                continue
            v = df[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size and v.max() > best_lift:
                best, best_lift = (ds, k), float(v.max())
    return best[0], best[1], best_lift


def tail_analysis(rdf: pd.DataFrame, pathways: list, frac: float = 0.05,
                  feature_axes: list = None) -> dict:
    """
    Everything the failure-anatomy figure and the findings section need about the tail.

    The tail is defined PER HEAD — the heads do not fail on the same samples — so
    concentration is reported for each, and the lift/effect tables are built for the head
    whose tail is most concentrated (named in `reference_head`), which is the one worth
    explaining. `spectrum_df` is optional and only adds the `worst_comp` axis.
    """
    feat = [c for c in (feature_axes or list(FEATURE_LABELS)) if c in rdf.columns]
    meta = [c for c in ("comp", "dataset_id", "n_comps") if c in rdf.columns]
    dropped_axes = []
    if "contrast" in feat and contrast_is_collapsed(rdf["contrast"].to_numpy(dtype=float)):
        feat.remove("contrast")
        dropped_axes.append("contrast")

    per_head, masks = {}, {}
    for k in pathways:
        col = f"{k}_mse"
        if col not in rdf.columns:
            continue
        v = rdf[col].to_numpy(dtype=float)
        per_head[k] = concentration(v, frac)
        masks[k] = tail_mask(v, frac)

    if not per_head:
        return {"frac": frac, "per_head": {}, "reference_head": None,
                "lift_df": pd.DataFrame(), "effect_df": pd.DataFrame(), "pooled": {},
                "dropped_axes": dropped_axes}

    ref_head = max(per_head, key=lambda k: (per_head[k]["share"]
                                            if np.isfinite(per_head[k]["share"]) else -1))
    mask = masks[ref_head]

    lift_df, pooled = lift_table(rdf, mask, meta + feat)

    effects = []
    for axis in feat:
        v = rdf[axis].to_numpy(dtype=float)
        stats = robust_effect(v[mask], v[~mask])
        stats.update({"axis": axis, "axis_label": LIFT_AXIS_LABELS.get(axis, axis)})
        effects.append(stats)
    effect_df = pd.DataFrame(effects)
    if len(effect_df):
        effect_df = (effect_df.assign(abs_effect=effect_df["effect"].abs())
                     .sort_values("abs_effect", ascending=False)
                     .drop(columns="abs_effect").reset_index(drop=True))

    return {"frac": frac, "per_head": per_head, "masks": masks,
            "reference_head": ref_head, "lift_df": lift_df,
            "effect_df": effect_df, "pooled": pooled,
            "dropped_axes": dropped_axes}
