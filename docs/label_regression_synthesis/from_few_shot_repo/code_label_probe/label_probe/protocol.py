"""
Split protocols and paired fold-wise aggregation.

Findings baked in (see plan.md "Split design"):
  - stratification on y-deciles and 10-fold were measured and REJECTED --
    they don't reduce repeat-to-repeat variance vs plain shuffled 5-fold.
  - a single holdout is unusable: 20 seeds gave R2 = 0.6596 +- 0.0115 (2-comp
    OLS), a 0.043 spread larger than the ~0.03 deltas this metric decides.
  - pooled K-fold OOF is ~23x tighter than a holdout; pairing on identical
    folds buys a further ~1.6x. Both are cheap, so both are done.
  - report TWO uncertainties: repeat_sd (split-assignment noise on this
    fixed dataset -- use for "does A beat B here") and bootstrap_sd
    (resampling spectra -- use for "does this generalize"). Never conflate.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning

# At high p/n (e.g. 12-comp transformer at 9216 dims), RidgeCV's small-alpha
# end is ill-conditioned by construction -- that's the point of sweeping
# alpha, and CV picks a well-conditioned one. Silence the resulting spam
# rather than the underlying (harmless, by design) numerics.
warnings.filterwarnings('ignore', category=LinAlgWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from threadpoolctl import threadpool_limits


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


@dataclass
class ProtocolResult:
    name: str
    r2_mean: float
    r2_repeat_sd: float          # split-assignment noise only
    r2_bootstrap_sd: float       # generalization uncertainty
    per_repeat_r2: list = field(default_factory=list)
    oof_predictions: np.ndarray = None   # from the first repeat, for scatter plots


def _oof(model_fn, X, y, splitter, groups=None):
    """
    Runs every fit under threadpool_limits(1). Measured on this (shared,
    heavily oversubscribed -- load avg ~68 on 40 cores) machine: an
    unconstrained LinearRegression.fit on a 160x512 matrix took ~6s from BLAS
    thread contention; pinned to 1 thread it takes ~0.02s. Since this module
    does many small independent fits rather than one big one, single-threaded
    BLAS is strictly better here -- there is no parallelism to lose.
    """
    pred = np.zeros(len(y))
    with threadpool_limits(limits=1):
        for tr, te in splitter.split(X, y, groups):
            m = model_fn()
            m.fit(X[tr], y[tr])
            pred[te] = np.asarray(m.predict(X[te])).ravel()
    return pred


def _bootstrap_sd(y_true, y_pred, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = r2(y_true[idx], y_pred[idx])
    return float(vals.std())


def run_primary(model_fn, X: np.ndarray, y: np.ndarray, n_repeats: int = 5,
                 n_folds: int = 5, seed0: int = 0) -> ProtocolResult:
    """All samples, repeated shuffled K-fold. The default protocol."""
    reps = []
    first_pred = None
    for r in range(n_repeats):
        cv = KFold(n_folds, shuffle=True, random_state=seed0 + r)
        pred = _oof(model_fn, X, y, cv)
        reps.append(r2(y, pred))
        if r == 0:
            first_pred = pred
    reps = np.array(reps)
    return ProtocolResult(
        name="primary", r2_mean=float(reps.mean()), r2_repeat_sd=float(reps.std()),
        r2_bootstrap_sd=_bootstrap_sd(y, first_pred), per_repeat_r2=reps.tolist(),
        oof_predictions=first_pred,
    )


def run_legacy(model_fn, X: np.ndarray, y: np.ndarray) -> ProtocolResult:
    """Exact historical protocol: n<=1000 subsample (caller truncates before
    calling), KFold(5, shuffle=False), single pass. One row, for lineage."""
    cv = KFold(5, shuffle=False)
    pred = _oof(model_fn, X, y, cv)
    val = r2(y, pred)
    return ProtocolResult(name="legacy", r2_mean=val, r2_repeat_sd=0.0,
                           r2_bootstrap_sd=_bootstrap_sd(y, pred),
                           per_repeat_r2=[val], oof_predictions=pred)


def run_label_group(model_fn, X: np.ndarray, y: np.ndarray,
                     n_folds: int = 5) -> ProtocolResult:
    """GroupKFold on the label VALUE (not spectrum id) -- guards against
    memorizing one of the 168 discrete levels. Measured clean this session
    (matches `primary` to 3 decimals); kept as a regression guard."""
    cv = GroupKFold(n_folds)
    pred = _oof(model_fn, X, y, cv, groups=y)
    val = r2(y, pred)
    return ProtocolResult(name="label_group", r2_mean=val, r2_repeat_sd=0.0,
                           r2_bootstrap_sd=_bootstrap_sd(y, pred),
                           per_repeat_r2=[val], oof_predictions=pred)


def paired_delta(model_fn_a, model_fn_b, X: np.ndarray, y: np.ndarray,
                  n_repeats: int = 5, n_folds: int = 5, seed0: int = 0):
    """
    Score two models on IDENTICAL folds per repeat, difference per-repeat.
    Returns (mean_delta, delta_sd, r2_a_mean, r2_b_mean) -- delta_sd is the
    tight, paired uncertainty (measured ~1.6x tighter than the marginal one).
    """
    deltas, a_vals, b_vals = [], [], []
    for r in range(n_repeats):
        cv = KFold(n_folds, shuffle=True, random_state=seed0 + r)
        pa = _oof(model_fn_a, X, y, cv)
        pb = _oof(model_fn_b, X, y, cv)
        ra, rb = r2(y, pa), r2(y, pb)
        a_vals.append(ra)
        b_vals.append(rb)
        deltas.append(ra - rb)
    deltas = np.array(deltas)
    return {
        "delta_mean": float(deltas.mean()),
        "delta_sd": float(deltas.std()),
        "r2_a_mean": float(np.mean(a_vals)),
        "r2_b_mean": float(np.mean(b_vals)),
        "is_tie": bool(abs(deltas.mean()) < 2 * deltas.std()) if deltas.std() > 0 else False,
    }


# ── Few-shot protocol (step 5) ────────────────────────────────────────────

def run_fewshot(model_fn, X: np.ndarray, y: np.ndarray, n_train: int,
                 eval_idx: np.ndarray, pool_idx: np.ndarray,
                 n_draws: int = 100, seed0: int = 0) -> dict:
    """
    Repeated random subsampling for the tiny-n regime. See plan.md "Step 5".

    k-fold is inappropriate here (5-fold on 20 samples trains on 16 and tests
    on 4), so instead: draw `n_train` labeled samples from `pool_idx`, fit,
    score on the FIXED held-out `eval_idx` (large, ~1000), repeat `n_draws`
    times, and report the DISTRIBUTION. A point estimate at this n is
    meaningless -- the estimator variance dominates.

    Returns median/IQR/p10/p90 for R2, plus MAE, Spearman, and
    frac_positive_r2 -- the fraction of draws that beat the mean predictor,
    which is the "does this work at all, and how reliably" number.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed0)
    X_ev, y_ev = X[eval_idx], y[eval_idx]

    r2s, maes, rhos = [], [], []
    with threadpool_limits(limits=1):
        for d in range(n_draws):
            tr = rng.choice(pool_idx, size=n_train, replace=False)
            m = model_fn()
            try:
                m.fit(X[tr], y[tr])
                pred = np.asarray(m.predict(X_ev)).ravel()
            except Exception:
                # a probe can legitimately fail at n=10 (e.g. PLS with
                # n_components > rank); count it as a failed draw rather
                # than dropping it silently, so frac_positive_r2 stays honest
                r2s.append(float("-inf"))
                maes.append(float("nan"))
                rhos.append(float("nan"))
                continue
            r2s.append(r2(y_ev, pred))
            maes.append(float(np.mean(np.abs(y_ev - pred))))
            rho = spearmanr(y_ev, pred).statistic
            rhos.append(float(rho) if np.isfinite(rho) else float("nan"))

    r2s = np.array(r2s, dtype=float)
    finite = r2s[np.isfinite(r2s)]
    return {
        "n_train": n_train,
        "n_draws": n_draws,
        "r2_median": float(np.median(finite)) if finite.size else float("nan"),
        "r2_p10": float(np.percentile(finite, 10)) if finite.size else float("nan"),
        "r2_p25": float(np.percentile(finite, 25)) if finite.size else float("nan"),
        "r2_p75": float(np.percentile(finite, 75)) if finite.size else float("nan"),
        "r2_p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
        "mae_median": float(np.nanmedian(maes)),
        "spearman_median": float(np.nanmedian(rhos)),
        # the client-facing reliability number: how often does it beat
        # predicting the mean?
        "frac_positive_r2": float(np.mean(r2s > 0)),
        "n_failed_draws": int(np.sum(~np.isfinite(r2s))),
    }
