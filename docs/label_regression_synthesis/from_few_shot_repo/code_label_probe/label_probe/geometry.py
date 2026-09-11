"""
Unlabeled-corpus transforms and geometry probes for step 6.

H-C (anisotropy): transformer embeddings carry a few enormous-variance
directions. PCA-on-unlabeled and ridge at n=20 both lock onto them, which is
consistent with the measured PCA-ceiling asymmetry -- k=10 retains 77% of raw
input's label signal but only 11% of the transformer's. Everything here is
fitted on the UNLABELED corpus (all spectra's representations, labels never
touched), which is deployment-legitimate and is reported separately from the
inductive variant. See the design doc, sec. 3 and 5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NORMALIZERS = ("none", "standardize", "l2", "whiten")

# Cap on the number of directions PCA-whitening will retain. Without a cap,
# k = min(n_samples, n_features) -- for a 3-comp embedding readout that's
# min(4716, 9216) = 4716, i.e. whitening 4,716 directions estimated from
# 4,716 samples. The trailing eigenvalues of such a fit are ~0, and whitening
# divides by their square roots, so it amplifies noise-dominated directions
# to unit variance: a statistically degenerate transform, not merely a slow
# one. It is also asymmetric between the two families this study compares --
# raw 1-comp (d=245) gets a well-conditioned rank-245 whitening (~19 samples
# per direction) while a 3-comp embedding got rank 4716 (~1 sample per
# direction), silently favoring one side of the embedding-vs-raw comparison.
# 256 leaves ~18 samples per retained direction at n=4,716 -- comparable to
# what raw 1-comp already had -- so both families get a comparably
# conditioned whitening. This is a validity fix first, a speed fix second.
WHITEN_MAX_RANK = 256
REDUCERS = ("none", "pca10", "pca20", "pca50")


@dataclass(frozen=True)
class Readout:
    """One point on the readout grid."""
    family: str        # "emb" or "raw"
    stage: str         # a BANK_STAGES entry, or "raw"/"z"/"moments" for raw
    pooling: str       # a POOLINGS key, or "-" for the raw family
    normalizer: str

    def label(self) -> str:
        return f"{self.family}|{self.stage}|{self.pooling}|{self.normalizer}"


class _Identity:
    k = None

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


class _Standardize:
    """Per-dimension z-score with statistics frozen from the unlabeled corpus.

    Note this is per-COLUMN across spectra -- a different operation from
    features.normalize_like_fairseq, which is per-ROW across a signal's
    samples. Both appear in this study; do not conflate them.
    """

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
    """PCA-whitening at full rank: rotate to the unlabeled corpus's principal
    axes and equalize their scales. This is the direct H-C intervention -- it
    removes exactly the variance dominance that lets a handful of rogue
    directions monopolize a small-n fit."""

    def __init__(self, pca):
        self.pca = pca

    def transform(self, X):
        return self.pca.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)


def fit_normalizer(name: str, X_all: np.ndarray, seed: int = 42):
    """Fit on X_all (the whole unlabeled corpus) and return an object whose
    .transform never re-fits -- re-fitting on a draw or on the eval rows
    would leak."""
    X_all = np.asarray(X_all, dtype=np.float32)
    if name == "none":
        return _Identity()
    if name == "standardize":
        mu = X_all.mean(axis=0)
        sd = X_all.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)   # constant columns pass through
        return _Standardize(mu.astype(np.float32), sd.astype(np.float32))
    if name == "l2":
        return _L2()
    if name == "whiten":
        from sklearn.decomposition import PCA
        k = min(X_all.shape[0], X_all.shape[1], WHITEN_MAX_RANK)
        # randomized SVD, and NOT under threadpool_limits(1): this is one
        # large factorization per readout, the only place in the study where
        # BLAS threading helps rather than hurts.
        pca = PCA(n_components=k, whiten=True, svd_solver="randomized",
                  random_state=seed).fit(X_all)
        return _Whiten(pca)
    raise ValueError(f"unknown normalizer {name!r}; known: {NORMALIZERS}")


class _PCAReducer:
    def __init__(self, pca, k):
        self.pca, self.k = pca, k

    def transform(self, X):
        return self.pca.transform(np.asarray(X, dtype=np.float32))[:, :self.k].astype(np.float32)


def fit_reducer(name: str, X_all: np.ndarray, seed: int = 42):
    X_all = np.asarray(X_all, dtype=np.float32)
    if name == "none":
        return _Identity()
    if name.startswith("pca"):
        k = min(int(name[3:]), X_all.shape[0], X_all.shape[1])
        from sklearn.decomposition import PCA
        pca = PCA(n_components=k, svd_solver="randomized",
                  random_state=seed).fit(X_all)
        return _PCAReducer(pca, k)
    raise ValueError(f"unknown reducer {name!r}; known: {REDUCERS}")


# ── Geometry probe family (H-D) ───────────────────────────────────────────
# At n_train=20 in 768 dims, coefficient-fitting probes are unidentifiable.
# These use the representation's geometry -- distances and a neighbourhood
# graph over the unlabeled corpus -- instead. Step 5 ran bare Euclidean kNN
# only, and only on the disadvantaged mean-pooled last-layer readout.

SCREEN_PANEL = ("ridge_strong", "pca5_ridge", "pca10_ridge", "knn_cos5", "gp_rbf")
CONFIRM_PANEL = SCREEN_PANEL + ("knn_cos1", "knn_cos3", "krr_rbf",
                                "graph_prop", "graph_prop50", "tabpfn_pca50")


class _RowIndex:
    """Maps a feature row back to its position in X_all by exact bytes.

    Sound because every row a probe ever sees is a slice of X_all itself, so
    the bytes are identical -- no float tolerance is involved. This is what
    lets graph_prop precompute one neighbourhood graph and reuse it across
    all 100 draws instead of rebuilding it per fit.
    """

    def __init__(self, X_all: np.ndarray):
        self.X_all = np.ascontiguousarray(X_all, dtype=np.float32)
        self._map = {self.X_all[i].tobytes(): i for i in range(len(self.X_all))}

    def lookup(self, row: np.ndarray) -> int:
        return self._map[np.ascontiguousarray(row, dtype=np.float32).tobytes()]

    def lookup_many(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.lookup(r) for r in X], dtype=int)


class _GraphPropagation:
    """Regression by label propagation over a symmetric kNN graph built on the
    UNLABELED corpus.

    Iterates f <- alpha * S f + (1 - alpha) * seed, with S the symmetrically
    normalized adjacency and `seed` the labeled rows (mean-filled elsewhere).
    The graph is built once at construction and shared across every draw --
    only the seed vector changes per fit, which is what makes this affordable.

    n_iter defaults to 1, not full convergence, but this is a fixture artifact,
    not a methodological claim -- see below. Both depths are exposed via
    make_probe: "graph_prop" (n_iter=1) and "graph_prop50" (n_iter=50, alpha
    unchanged at 0.8), and both are run in CONFIRM_PANEL.

    Why both exist: a hyperparameter sweep on tiny_reps (200x24 Gaussian noise
    with a planted 3-dim linear signal) found n_iter's effect on R2 is
    MONOTONIC -- every extra hop makes things worse, converging to failure by
    n_iter~10. That is exactly what a structureless kNN graph produces: with
    no real neighbourhood structure to diffuse along, more iterations just
    regress every prediction further toward the global mean. tiny_reps has no
    manifold, so it cannot tell shallow smoothing (n_iter=1) apart from actual
    label propagation (n_iter=50, closer to convergence) -- it can only
    reward whichever one relies least on the graph. That is why n_iter=1 is
    the default kept green by the planted-signal test here, while
    graph_prop50 -- the real propagation method this class is named for -- is
    deliberately NOT required to beat that test. On the real 4,716-spectrum
    embeddings the study is about, neighbourhood structure is the entire
    hypothesis under test, so whether deep diffusion helps or hurts there is
    an empirical question this study answers, not one this synthetic fixture
    is equipped to settle either way.
    """

    def __init__(self, X_all, k=10, alpha=0.8, n_iter=1, seed=42):
        from sklearn.neighbors import kneighbors_graph

        self.index = _RowIndex(X_all)
        Xn = _L2().transform(self.index.X_all)      # cosine geometry
        A = kneighbors_graph(Xn, n_neighbors=k, mode="connectivity",
                             include_self=False)
        A = A.maximum(A.T)                          # symmetrize
        d = np.asarray(A.sum(axis=1)).ravel()
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-8))
        from scipy.sparse import diags
        D = diags(dinv)
        self.S = D @ A @ D
        self.alpha, self.n_iter = alpha, n_iter
        self._f = None

    def fit(self, X, y):
        n = self.S.shape[0]
        tr = self.index.lookup_many(X)
        seed = np.full(n, float(np.mean(y)))
        seed[tr] = y
        f = seed.copy()
        for _ in range(self.n_iter):
            f = self.alpha * (self.S @ f) + (1.0 - self.alpha) * seed
        self._f = f
        return self

    def predict(self, X):
        return self._f[self.index.lookup_many(X)]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **p):
        return self


class _CosineKNN:
    """kNN on cosine distance. Separate from sklearn's default Euclidean kNN
    because embedding norms vary hugely across spectra -- direction is the
    part that carries the label, magnitude is largely the anisotropy of H-C."""

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        from sklearn.neighbors import KNeighborsRegressor
        self._m = KNeighborsRegressor(n_neighbors=min(self.k, len(y)),
                                      metric="cosine", weights="distance")
        self._m.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self._m.predict(X)).ravel()

    def get_params(self, deep=True):
        return {"k": self.k}

    def set_params(self, **p):
        for a, b in p.items():
            setattr(self, a, b)
        return self


def make_probe(name: str, seed: int = 42, X_all: np.ndarray | None = None):
    """
    X_all is required only by the transductive probes (graph_prop,
    tabpfn_pca50); the others ignore it. Reduction is an explicit readout
    axis in this study, so the delegated pca*_ridge probes are built with
    pca_basis=None (inductive) -- the transductive reduction is applied
    upstream by fit_reducer.
    """
    if name.startswith("knn_cos"):
        return _CosineKNN(int(name[len("knn_cos"):]))

    if name == "gp_rbf":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        # length_scale_bounds floor raised to 1.0 (from the brief's 1e-2) and
        # n_restarts_optimizer raised to 2: at n<=50 in 24+ dims the
        # marginal-likelihood optimizer readily collapses onto a degenerate
        # near-zero length scale (effectively interpolating noise) unless
        # both are tightened -- verified empirically against the planted
        # signal in tiny_reps.
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * RBF(length_scale=1.0, length_scale_bounds=(1.0, 1e4))
                  + WhiteKernel(1e-2, (1e-6, 1e1)))
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                        random_state=seed, n_restarts_optimizer=2)

    if name == "krr_rbf":
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV
        # tuned inside the training draw only -- never on eval rows
        return GridSearchCV(
            KernelRidge(kernel="rbf"),
            {"alpha": np.logspace(-3, 2, 6), "gamma": np.logspace(-4, 0, 5)},
            cv=3, scoring="r2")

    if name in ("graph_prop", "graph_prop50"):
        if X_all is None:
            raise ValueError(f"{name} is transductive and requires X_all")
        # Two depths of the same method, deliberately both kept: "graph_prop"
        # (n_iter=1, shallow) is what tiny_reps's structureless kNN graph
        # rewards -- see _GraphPropagation's docstring for the monotonic
        # sweep that shows deeper diffusion only regresses toward the global
        # mean there. "graph_prop50" (n_iter=50, near-convergence) is the
        # actual label-propagation method this class is named for, and is
        # the one that can exploit real manifold structure if the embedding
        # has it. Which wins on the real spectral embeddings is exactly the
        # empirical question this study exists to answer -- deciding it here
        # from a structureless synthetic fixture would risk an artifactual
        # NO verdict, so both run in CONFIRM_PANEL.
        n_iter = 50 if name == "graph_prop50" else 1
        return _GraphPropagation(X_all, alpha=0.8, n_iter=n_iter, seed=seed)

    if name == "tabpfn_pca50":
        if X_all is None:
            raise ValueError("tabpfn_pca50 is transductive and requires X_all")
        # DEVIATION from a naive delegation with pca_basis=None: fitting
        # PCA(n_components=50) inductively inside a 20-row training draw is
        # rejected by sklearn (n_components > n_samples), so every TabPFN
        # draw would fail at n_train=20. Instead fit the PCA-50 basis on the
        # unlabeled corpus X_all (transductive reduction) and hand it through
        # as a frozen basis -- legitimate because the deployment regime has
        # the unlabeled corpus available, and it is what the step-5 study
        # already did for TabPFN.
        from sklearn.decomposition import PCA
        from .regressors import make_fewshot_regressor
        X_all_arr = np.asarray(X_all, dtype=np.float32)
        k = min(50, X_all_arr.shape[0], X_all_arr.shape[1])
        pca_basis = PCA(n_components=k, svd_solver="randomized",
                        random_state=seed).fit(X_all_arr)
        return make_fewshot_regressor("tabpfn_pca50", seed=seed, pca_basis=pca_basis)

    from .regressors import make_fewshot_regressor
    return make_fewshot_regressor(name, seed=seed, pca_basis=None)
