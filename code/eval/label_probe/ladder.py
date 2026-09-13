"""
Label-efficiency ladder: repeated random-subsampling comparison of several
named, already-normalized feature matrices across a grid of training-set
sizes, using shared draw indices so every readout is compared on identical
training sets at a given n (a paired comparison — this tightens the
resolvable delta and costs nothing).
"""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from .protocol import r2


def draw_indices(pool: np.ndarray, n_train: int, n_draws: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [rng.choice(pool, size=n_train, replace=False) for _ in range(n_draws)]


def n_draws_for(n_train: int) -> int:
    """Fewer draws at larger n, where each draw is more expensive (RidgeCV's
    LOOCV path is O(n*d^2)) and the estimator is already less noisy."""
    if n_train <= 50:
        return 100
    if n_train <= 200:
        return 40
    if n_train <= 500:
        return 15
    if n_train <= 1000:
        return 6
    if n_train <= 2000:
        return 3
    return 1


def score_readout(X: np.ndarray, y: np.ndarray, probe_fn, eval_idx: np.ndarray,
                   n_trains: list, seed: int = 42) -> dict:
    """
    X, y: the full pool (already normalized). probe_fn: () -> a fresh
    sklearn-style estimator. eval_idx: fixed held-out rows scored at every
    rung. Draws are generated fresh per n_train from the complement of
    eval_idx, shared across every readout the caller compares (call this
    with the SAME n_trains/seed for each readout being compared).
    """
    all_idx = np.arange(len(y))
    pool = np.setdiff1d(all_idx, eval_idx)
    X_ev, y_ev = X[eval_idx], y[eval_idx]

    out = {}
    for n_train in n_trains:
        nd = n_draws_for(n_train)
        draws = draw_indices(pool, min(n_train, len(pool)), nd, seed=seed + n_train)
        scores = []
        for tr in draws:
            try:
                with threadpool_limits(limits=1):
                    m = probe_fn()
                    m.fit(X[tr], y[tr])
                    pred = np.asarray(m.predict(X_ev)).ravel()
                scores.append(r2(y_ev, pred))
            except Exception:
                scores.append(float("-inf"))
        arr = np.array(scores, dtype=float)
        fin = arr[np.isfinite(arr)]
        out[n_train] = {
            "n_train": n_train,
            "n_draws": len(arr),
            # Every draw's score, kept so a caller can difference two readouts
            # DRAW BY DRAW. Draws are shared across readouts at a given rung
            # (same seed, same pool), so the paired difference is the honest
            # uncertainty on "how far apart are these two" -- differencing the
            # two medians instead would throw the pairing away.
            "r2_draws": [float(v) if np.isfinite(v) else None for v in arr],
            "r2_median": float(np.median(fin)) if fin.size else float("nan"),
            "r2_p25": float(np.percentile(fin, 25)) if fin.size else float("nan"),
            "r2_p75": float(np.percentile(fin, 75)) if fin.size else float("nan"),
            "frac_positive_r2": float(np.mean(arr > 0)),
        }
    return out
