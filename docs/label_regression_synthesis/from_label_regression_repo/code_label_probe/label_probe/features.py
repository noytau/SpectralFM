"""
Feature-vector construction: normalization, component selection/dedup, and
the wide/long/stacked layouts.

Key findings baked in (see plan.md "The components"):
  - comp14==comp20 and comp15==comp21 on 100% of spectra -- UNIQUE_COMPS
    drops the duplicates. There are 12 unique components, not 14.
  - components are NOT exchangeable. A pooled model across per-component rows
    (the "long" layout without per-component identity) collapses to R2~0.006
    -- this reproduces the historical bug in commit bf94422 exactly. The
    "wide" layout (components concatenated into one row per spectrum) is the
    only sound default; "long" is kept only as a deliberate negative control
    and for the grouped-vs-ungrouped leak-guard test.
  - z-scoring each component independently (the eval's existing
    normalize_like_fairseq) costs 0.06-0.17 R2 at 2 comps by destroying
    per-component amplitude and cross-component scale ratios. Report raw
    AND z-scored; don't silently pick one.
"""
from __future__ import annotations

import numpy as np

ALL_COMPS = (0, 1, 2, 3, 8, 9, 14, 15, 20, 21, 27, 28, 30, 31)
DUPLICATE_PAIRS = ((14, 20), (15, 21))
UNIQUE_COMPS = tuple(c for c in ALL_COMPS if c not in (20, 21))  # 12 unique

# comp-count ladders used across steps 1/3/4
COMP_LADDER = {
    1: (0,),
    2: (0, 1),
    3: (0, 1, 2),
    7: UNIQUE_COMPS[:7],
    12: UNIQUE_COMPS,
}


def normalize_like_fairseq(arr: np.ndarray) -> np.ndarray:
    """Per-row (per-sample) z-score. Ported verbatim from
    evaluations/label_regression.py:32-35 -- identical semantics, needed so
    the embedding side (which the backbone expects normalized) stays
    comparable to the historical eval."""
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-8
    return ((arr - mean) / std).astype(np.float32)


def make_wide(X_by_comp: np.ndarray, comp_idx: list) -> np.ndarray:
    """X_by_comp: [N, K, D] (K = len(loaded comps), D = feature dim per comp).
    comp_idx: positions into K to select. Returns [N, len(comp_idx)*D]."""
    n = X_by_comp.shape[0]
    return X_by_comp[:, comp_idx, :].reshape(n, -1)


def make_long(X_by_comp: np.ndarray, y: np.ndarray, comp_idx: list):
    """Component = row. Returns (X_long [N*k, D], y_long [N*k], groups [N*k])
    where groups is the spectrum index, for GroupKFold.

    WARNING (measured, see plan.md gate 7): a pooled linear model on this
    layout collapses to R2~0.006 regardless of grouping, because components
    are not exchangeable -- grouping guards against a leak a weak model
    can't exploit anyway. Use make_stacked for a layout that actually works.
    """
    n, k = X_by_comp.shape[0], len(comp_idx)
    X_long = X_by_comp[:, comp_idx, :].reshape(n * k, -1)
    y_long = np.repeat(y, k)
    groups = np.repeat(np.arange(n), k)
    return X_long, y_long, groups


def make_pairwise_features(X_by_comp: np.ndarray, comp_idx: list) -> np.ndarray:
    """Explicit cross-component features: per-component (mean, std,
    log-energy, max, argmax) plus pairwise differences and log energy
    ratios between all selected components. Tests H2/H8 (plan.md step 3):
    z-scoring destroys exactly this information, and cross-component
    structure is the dominant carrier (1->2 comps: raw R2 0.41->0.83)."""
    n = X_by_comp.shape[0]
    sel = X_by_comp[:, comp_idx, :]                    # [N, k, D]
    k = sel.shape[1]

    mean = sel.mean(axis=-1)                           # [N, k]
    std = sel.std(axis=-1)
    energy = (sel ** 2).sum(axis=-1)
    log_energy = np.log(energy + 1e-8)
    mx = sel.max(axis=-1)
    argmax = sel.argmax(axis=-1).astype(np.float64)

    per_comp = np.concatenate([mean, std, log_energy, mx, argmax], axis=-1)  # [N,5k]

    pair_feats = []
    for i in range(k):
        for j in range(i + 1, k):
            pair_feats.append(mean[:, i] - mean[:, j])
            pair_feats.append(log_energy[:, i] - log_energy[:, j])
    pairwise = np.stack(pair_feats, axis=-1) if pair_feats else np.zeros((n, 0))

    return np.concatenate([per_comp, pairwise], axis=-1)
