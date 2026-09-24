"""
Shuffled-label leak canary.

Permute y, then run the identical evaluation path. With labels destroyed, an
honest pipeline MUST score R2 ~ 0 (slightly negative out-of-fold). Anything
meaningfully above 0 means label information is reaching the model through a
path other than the training labels — e.g. a supervised transform (PLS, a
learned reducer) fitted on combined train+test data before cross-validation.

Run this on any new pipeline that adds a supervised transform, and on any
pipeline whose result looks surprisingly good.
"""
from __future__ import annotations

import numpy as np

from .protocol import r2


def shuffled_label_canary(fit_predict_oof, X: np.ndarray, y: np.ndarray,
                           seed: int = 42, threshold: float = 0.02) -> dict:
    """
    fit_predict_oof: callable (X, y) -> out-of-fold predictions [N], using
    whatever CV/draw protocol the real pipeline uses.

    Returns {"real_r2", "shuffled_r2", "passed"}. `passed` is True iff the
    shuffled-label run scores at or below `threshold` — i.e. it collapsed to
    noise as it should.
    """
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(np.asarray(y))

    real_pred = fit_predict_oof(X, y)
    shuf_pred = fit_predict_oof(X, y_shuffled)

    real_r2 = r2(np.asarray(y), real_pred)
    shuf_r2 = r2(y_shuffled, shuf_pred)
    return {
        "real_r2": real_r2,
        "shuffled_r2": shuf_r2,
        "passed": bool(shuf_r2 <= threshold),
    }
