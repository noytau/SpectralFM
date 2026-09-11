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
DEFAULT_NORMALIZER = "whiten"

# Cap on the number of directions PCA-whitening retains. Without a cap,
# k = min(n_samples, n_features): for a 3-comp embedding readout that's
# min(4716, 9216) = 4716, whitening 4,716 directions from 4,716 samples —
# the trailing eigenvalues of such a fit are ~0, and whitening divides by
# their square roots, amplifying noise-dominated directions to unit
# variance. 256 leaves comfortably many samples per retained direction at
# n≈4,716 while staying well-conditioned for smaller feature counts too.
WHITEN_MAX_RANK = 256


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
    def __init__(self, pca):
        self.pca = pca

    def transform(self, X):
        return self.pca.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)


def fit_normalizer(name: str, X_all: np.ndarray, seed: int = 42):
    """Fit on X_all (the whole unlabeled corpus, no labels)."""
    X_all = np.asarray(X_all, dtype=np.float32)
    if name == "none":
        return _Identity()
    if name == "standardize":
        mu = X_all.mean(axis=0)
        sd = X_all.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        return _Standardize(mu.astype(np.float32), sd.astype(np.float32))
    if name == "l2":
        return _L2()
    if name == "whiten":
        from sklearn.decomposition import PCA
        k = min(X_all.shape[0], X_all.shape[1], WHITEN_MAX_RANK)
        pca = PCA(n_components=k, whiten=True, svd_solver="randomized",
                  random_state=seed).fit(X_all)
        return _Whiten(pca)
    raise ValueError(f"unknown normalizer {name!r}; known: {NORMALIZERS}")
