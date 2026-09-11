"""
Split, diagnostics and readout scoring for step 6.

Two disjoint evaluation sets exist for one reason: 468 readouts are screened
and the best is reported, which is selection on the test set. eval_a ranks,
eval_b reports, and raw input goes through the identical select-then-confirm
procedure so any residual optimism is symmetric. See the design doc, sec. 5.
"""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from .protocol import r2, run_primary
from .regressors import make_regressor


def make_split(n: int, n_eval_a: int = 500, n_eval_b: int = 500,
               seed: int = 42) -> dict:
    """One permutation, sliced. Deterministic in `seed` so every readout,
    every probe and both families see the identical partition."""
    perm = np.random.default_rng(seed).permutation(n)
    return {
        "eval_a": perm[:n_eval_a],
        "eval_b": perm[n_eval_a:n_eval_a + n_eval_b],
        "pool": perm[n_eval_a + n_eval_b:],
    }


def draw_indices(pool: np.ndarray, n_train: int, n_draws: int,
                 seed: int) -> list:
    """The shared training draws. Generated once and passed to every readout
    so comparisons are PAIRED -- the same 20 spectra train the embedding
    probe and the raw probe. Measured in step 5's protocol work: pairing
    tightens the resolvable delta ~1.6x, and it costs nothing."""
    rng = np.random.default_rng(seed)
    return [rng.choice(pool, size=n_train, replace=False) for _ in range(n_draws)]


def readout_diagnostics(X_all: np.ndarray, y: np.ndarray,
                        seed: int = 42) -> dict:
    """
    Label-free geometry plus large-n label diagnostics for one readout.

    `whitened_topk_r2` is the load-bearing one: a large-n RidgeCV fit on the
    top-k whitened principal components bounds what ANY probe could extract
    from a k-dimensional unlabeled reduction. The step-5 asymmetry -- k=10
    retaining 77% of raw's label signal but 11% of the transformer's -- is
    exactly this quantity, and the study's secondary claim is that it
    predicts few-shot R2 across readouts.

    These use all labels at large n. They therefore rank readouts using
    information a 20-label deployment does not have; that is why they inform
    only the screen, and every reported number comes from eval_b.

    Only a SINGLE randomized-SVD PCA is fitted here (not two, as a naive
    reading might do) -- one whitened PCA at k_full = min(n, d, 128)
    components. `participation_ratio` and `effective_rank` are derived from
    its `explained_variance_`; whitening rescales the transformed data but
    not the eigenvalues themselves, so these are identical to what an
    un-whitened PCA would have given. `ridgecv_full_r2` is RidgeCV run on
    the top-`k_full` whitened components rather than on the raw
    full-dimensional matrix (the key name is kept for a later task's report
    code, which indexes it by name) -- this equals the true full-dimensional
    RidgeCV value whenever d <= 128 (the whitened top-k components then span
    the same subspace as the raw features), and is a fast ranking stand-in
    otherwise. The 128 cap (down from an earlier 200) matches the
    `WHITEN_MAX_RANK` fix in `geometry.fit_normalizer`: at n=4,716 a
    128-component PCA is comfortably well-conditioned (~37 samples per
    retained direction), avoiding the same trailing-eigenvalue amplification
    that motivated capping the readout-level whitening normalizer. This
    keeps the cost of this function to one big SVD plus five small RidgeCV
    fits per readout instead of two big SVDs plus a RidgeCV on up to 9,216
    raw dimensions, which is what pushed the naive version past this study's
    8-hour screen budget across ~456 readouts.
    """
    from sklearn.decomposition import PCA

    X = np.asarray(X_all, dtype=np.float32)
    n, d = X.shape

    k_full = min(n, d, 128)
    pca = PCA(n_components=k_full, whiten=True, svd_solver="randomized",
              random_state=seed)
    Xw = pca.fit_transform(X)

    ev = pca.explained_variance_
    ev_norm = ev / max(ev.sum(), 1e-12)
    participation_ratio = float(1.0 / np.sum(ev_norm ** 2))
    nz = ev_norm[ev_norm > 1e-12]
    effective_rank = float(np.exp(-np.sum(nz * np.log(nz))))

    with threadpool_limits(limits=1):
        full = run_primary(lambda: make_regressor("ridgecv", seed=seed),
                           Xw, y, n_repeats=1, seed0=seed)

    topk = {}
    for k in (5, 10, 20, 50):
        kk = min(k, Xw.shape[1])
        with threadpool_limits(limits=1):
            pr = run_primary(lambda: make_regressor("ridgecv", seed=seed),
                             Xw[:, :kk], y, n_repeats=1, seed0=seed)
        topk[k] = pr.r2_mean

    return {
        "participation_ratio": participation_ratio,
        "effective_rank": effective_rank,
        "ridgecv_full_r2": full.r2_mean,
        "whitened_topk_r2": topk,
        "n_features": int(d),
    }


def score_readout(X_all: np.ndarray, y: np.ndarray, split: dict,
                  probes, n_trains, draws_by_n: dict, eval_key: str,
                  X_for_graph: np.ndarray = None, seed: int = 42, probe_factory=None) -> dict:
    """
    Score one already-normalized readout on the given evaluation set.

    Unlike protocol.run_fewshot this takes the draws from OUTSIDE, so every
    readout and probe is fitted on identical training sets -- that is what
    makes the embedding-vs-raw delta paired. r2_per_draw is kept so the
    report can bootstrap the paired difference.

    DEVIATION from constructing a fresh probe inside the draw loop: each
    probe is instead constructed ONCE per (probe, n_train) cell, before the
    draw loop, and re-fit per draw. `graph_prop`/`graph_prop50` build a
    4,716-node kNN graph at CONSTRUCTION time (its own docstring promises
    this graph is built once and reused across draws); constructing fresh
    per draw would rebuild that graph up to 100x per cell -- hours instead
    of the cost of one graph build. Every estimator in this panel fully
    resets its state in `.fit()`, so sharing the constructed instance across
    draws is safe. If construction itself raises (e.g. a probe genuinely
    incompatible with this readout), every draw in that cell is counted as a
    failed draw rather than crashing the whole sweep -- the per-draw
    try/except below still applies to the fit/predict step for whichever
    failures happen there.

    probe_factory: optional callable (name, seed, X_all) -> estimator, used in
    place of geometry.make_probe. Step 7's n-ladder needs probes geometry does
    not build (ridgecv at large n), and injecting the factory keeps every line
    of the scoring/pairing/failure-counting logic shared between studies.
    """
    from .geometry import make_probe

    X = np.asarray(X_all, dtype=np.float32)
    ev = split[eval_key]
    X_ev, y_ev = X[ev], y[ev]
    graph_src = X if X_for_graph is None else X_for_graph

    from scipy.stats import spearmanr

    _factory = make_probe if probe_factory is None else probe_factory

    out = {}
    for probe in probes:
        out[probe] = {}
        for n_train in n_trains:
            draws = draws_by_n[n_train]
            scores = []
            maes = []
            rhos = []

            try:
                m = _factory(probe, seed=seed, X_all=graph_src)
                construction_failed = False
            except Exception as exc:      # noqa: BLE001
                print(f"[step6] probe construction failed: {probe} "
                      f"n={n_train}: {type(exc).__name__}: {exc}", flush=True)
                m = None
                construction_failed = True

            for tr in draws:
                if construction_failed:
                    scores.append(float("-inf"))
                    maes.append(float("nan"))
                    rhos.append(float("nan"))
                    continue
                try:
                    with threadpool_limits(limits=1):
                        m.fit(X[tr], y[tr])
                        pred = np.asarray(m.predict(X_ev)).ravel()
                    scores.append(r2(y_ev, pred))
                    maes.append(float(np.mean(np.abs(y_ev - pred))))
                    rho = spearmanr(y_ev, pred).statistic
                    rhos.append(float(rho) if np.isfinite(rho) else float("nan"))
                except Exception as exc:      # noqa: BLE001
                    # a probe can legitimately fail at n=20 (singular kernel,
                    # rank-deficient PLS). Count it as a failed draw rather
                    # than dropping it, so frac_positive_r2 stays honest.
                    print(f"[step6] draw failed: {probe} n={n_train}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    scores.append(float("-inf"))
                    maes.append(float("nan"))
                    rhos.append(float("nan"))

            arr = np.array(scores, dtype=float)
            fin = arr[np.isfinite(arr)]
            out[probe][n_train] = {
                "n_train": n_train,
                "n_draws": len(arr),
                "r2_median": float(np.median(fin)) if fin.size else float("nan"),
                "r2_p10": float(np.percentile(fin, 10)) if fin.size else float("nan"),
                "r2_p25": float(np.percentile(fin, 25)) if fin.size else float("nan"),
                "r2_p75": float(np.percentile(fin, 75)) if fin.size else float("nan"),
                "r2_p90": float(np.percentile(fin, 90)) if fin.size else float("nan"),
                "mae_median": float(np.nanmedian(maes)),
                "spearman_median": float(np.nanmedian(rhos)),
                "frac_positive_r2": float(np.mean(arr > 0)),
                "n_failed_draws": int(np.sum(~np.isfinite(arr))),
                "r2_per_draw": arr.tolist(),
            }
    return out
