"""
Component bookkeeping and feature-vector layout for the labeled_data subset.
"""
from __future__ import annotations

import numpy as np

# comp14==comp20 and comp15==comp21 on 100% of spectra in labeled_data;
# UNIQUE_COMPS drops the duplicates (12 unique components, not 14).
ALL_COMPS = (0, 1, 2, 3, 8, 9, 14, 15, 20, 21, 27, 28, 30, 31)
UNIQUE_COMPS = tuple(c for c in ALL_COMPS if c not in (20, 21))

# comp-count ladders: which components to concatenate for an n-comp probe.
COMP_LADDER = {
    1: (0,),
    2: (0, 1),
    3: (0, 1, 2),
    7: UNIQUE_COMPS[:7],
    12: UNIQUE_COMPS,
}


def normalize_like_fairseq(arr: np.ndarray) -> np.ndarray:
    """Per-row (per-sample) z-score — the normalization the backbone expects
    its input in. arr: [N, L]."""
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-8
    return ((arr - mean) / std).astype(np.float32)


def make_wide(X_by_comp: np.ndarray, comp_idx: list) -> np.ndarray:
    """X_by_comp: [N, K, D] (K = number of loaded comps, D = feature dim per
    comp). comp_idx: positions into K to select and concatenate.

    This is the only sound feature layout for this dataset: components are
    not exchangeable (concatenate, don't pool/average per-component rows —
    a pooled model collapses to R2~0 regardless of grouping).
    """
    n = X_by_comp.shape[0]
    return X_by_comp[:, comp_idx, :].reshape(n, -1)
