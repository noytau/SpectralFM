"""
Unlabeled-corpus feature normalizers.

`whiten` is the default: PCA-whitening on the unlabeled corpus (rotate to
its principal axes, equalize their scales) removes a small number of
dominant-variance directions from monopolizing a small-n fit. It is the
single largest lever measured in this probe's development, on both raw
input and embeddings, and costs nothing at deployment time since it never
touches a label.

Every normalizer here is fit ONCE on the full unlabeled corpus and its
.transform() never re-fits — re-fitting on a training draw or on eval rows
would leak.
"""
from __future__ import annotations

import numpy as np

NORMALIZERS = ("none", "standardize", "l2", "whiten")

# "whiten" keeps every numerically-alive direction. "whitenK" keeps only the
# top K principal directions before equalising -- the amount of whitening is
# a HYPERPARAMETER, not a constant, because its optimum moves with n_train:
# full-rank whitening equalises ~235 near-zero-variance raw directions, which
# a full-pool fit can exploit and a 50-label fit cannot tell from noise.
DEFAULT_NORMALIZER = "whiten"

# How many directions PCA-whitening retains.
#
# This used to be a flat cap of 256, which was wrong and cost real signal. A
# 768-d Transformer readout kept only its top 256 PCs; those hold 100.00% of
# the variance (to 2dp) and yet discarding the rest cost 0.12-0.17 R²
# depending on the block. That is the same lesson the raw-input baseline
# teaches: on this task the label signal lives disproportionately in
# near-zero-variance directions, so "explains ~all the variance" is NOT a
# safe reason to drop a direction. Measured, 1 component, n=4,716:
#
#   block                  standardize   whiten k=256   whiten full-rank
#   Transformer layer 2         0.8531         0.7344             0.8536
#   Transformer layer 12        0.6819         0.5794             0.7457
#   FE (post-LN)                0.6088         0.6052             0.7631
#
# The real constraint is statistical, not a magic number: an eigenvalue
# estimated from too few samples is noise, and whitening divides by its
# square root. So keep at most one direction per 2 samples.
WHITEN_SAMPLES_PER_DIRECTION = 2

# Below this, take the exact SVD rather than the randomized approximation.
# 4,716 x 9,216 (the widest readout here, 3-comp segment4) is a few hundred MB
# and well under a minute, so every decomposition in this study is exact.
EXACT_SVD_MAX_DIM = 6000

# The only directions dropped beyond that are the NUMERICALLY dead ones --
# those below the standard matrix-rank tolerance (sigma_max * eps * max(n,d),
# squared because explained_variance_ is in sigma^2 units). Anything less
# strict re-commits this module's own cardinal sin: on this task variance is
# not information, so a "surely this direction is negligible" variance
# threshold silently deletes signal. A 1e-10 variance-ratio cut, which looks
# conservative, kept only 33 of raw input's 245 directions and cost the
# baseline 0.045 R² (0.833 -> 0.788).
#
# The decomposition is also done in float64 for the reason the baseline
# section documents: at cond ~1e9, float32 cannot resolve the small
# singular values, which are exactly the ones carrying the label signal.


class _Identity:
    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


class _Standardize:
    def __init__(self, mu, sd):
        self.mu, self.sd = mu, sd

    def transform(self, X):
        return ((np.asarray(X, dtype=np.float32) - self.mu) / self.sd).astype(np.float32)


class _L2:
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return (X / np.maximum(n, 1e-8)).astype(np.float32)


class _Whiten:
    def __init__(self, pca, k_keep):
        self.pca = pca
        self.k_keep = k_keep

    def transform(self, X):
        Z = self.pca.transform(np.asarray(X, dtype=np.float64))
        return Z[:, : self.k_keep].astype(np.float32)


def fit_normalizer(name: str, X_all: np.ndarray, seed: int = 42):
    """Fit on X_all (the whole unlabeled corpus, no labels)."""
    X_all = np.asarray(X_all)
    if name != "whiten":
        # whiten does its own float64 promotion below; don't downcast it here
        X_all = X_all.astype(np.float32, copy=False)
    if name == "none":
        return _Identity()
    if name == "standardize":
        mu = X_all.mean(axis=0)
        sd = X_all.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        return _Standardize(mu.astype(np.float32), sd.astype(np.float32))
    if name == "l2":
        return _L2()
    if name.startswith("whiten"):
        from sklearn.decomposition import PCA
        k_req = int(name[len("whiten"):]) if name != "whiten" else None
        X64 = np.asarray(X_all, dtype=np.float64)
        n, d = X64.shape
        k = min(d, max(1, n // WHITEN_SAMPLES_PER_DIRECTION))
        if k_req is not None:
            k = min(k, max(1, k_req))
        # Always take the exact SVD at the sizes this study works at. The
        # randomized solver is an approximation of exactly the small singular
        # values that carry the label signal here: on raw input it reported
        # 0.833 where the exact decomposition gives 0.824, and it is least
        # accurate when k is a large fraction of d -- which is the regime every
        # readout in this study lands in.
        solver = "full" if min(n, d) <= EXACT_SVD_MAX_DIM else "randomized"
        pca = PCA(n_components=k, whiten=True, svd_solver=solver,
                  random_state=seed).fit(X64)
        ev = pca.explained_variance_
        tol = (np.finfo(np.float64).eps * max(n, d)) ** 2
        k_keep = int(np.sum(ev > tol * ev[0])) if ev.size else 0
        return _Whiten(pca, max(1, k_keep))
    raise ValueError(f"unknown normalizer {name!r}; known: {NORMALIZERS} "
                     f"(plus 'whitenK' for a rank-K whitening)")
