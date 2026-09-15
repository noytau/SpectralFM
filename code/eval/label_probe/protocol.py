"""
The primary split protocol: repeated shuffled K-fold, out-of-fold.

Findings baked in:
  - stratification on y-deciles and 10-fold were measured and REJECTED --
    they don't reduce repeat-to-repeat variance vs plain shuffled 5-fold.
  - a single holdout is unusable: 20 seeds gave R2 = 0.6596 +- 0.0115 (2-comp
    OLS), a 0.043 spread larger than the ~0.03 deltas this metric decides.
  - pooled K-fold OOF is ~23x tighter than a holdout; pairing readouts on
    identical draws (used throughout ladder.py/panel.py) buys a further
    ~1.6x. Both are cheap, so both are done.
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
from sklearn.model_selection import KFold
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
