"""
Regressor registry for the label probe.

Includes a plain OLS branch (missing from metrics.compute_linear_probing_metrics)
because the plan's evidence shows probe choice alone moves the input R2
baseline by up to 0.27 -- OLS is not optional, it's part of the panel that
makes "best linear baseline" honest.

xgb/hgb: xgboost preferred, HistGradientBoostingRegressor as a zero-dep
fallback so this runs under either the labelprobe venv or plain spectralfm_env.
"""
from __future__ import annotations

import numpy as np


def _xgb_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def make_regressor(name: str, seed: int = 42):
    """
    name in: 'ols', 'ridge', 'ridgecv', 'xgb', 'hgb', 'knn', 'mlp', 'pls', 'dummy'
    """
    if name == "ols":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    if name == "ridgecv":
        from sklearn.linear_model import RidgeCV
        return RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    if name == "xgb":
        if _xgb_available():
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                random_state=seed, n_jobs=-1,
            )
        # fall back silently but visibly
        name = "hgb"
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=seed)
    if name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=5)
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(256,), max_iter=500,
                             random_state=seed, early_stopping=True)
    if name == "pls":
        from sklearn.cross_decomposition import PLSRegression
        return PLSRegression(n_components=64)
    if name == "dummy":
        from sklearn.dummy import DummyRegressor
        return DummyRegressor(strategy="mean")
    raise ValueError(f"unknown regressor {name!r}")


REGRESSOR_PANEL = ("ols", "ridge", "ridgecv")   # the default "best-linear-baseline" panel


# ── Few-shot probe family (step 5) ────────────────────────────────────────
# At n_train ~ 10-50, the step 1-4 probes are unidentifiable (p >> n). These
# are deliberately low-capacity. See plan.md "Step 5".

FEWSHOT_PANEL = (
    "dummy",              # mean predictor -- the floor, a real contender at n=10
    "ridge_strong",       # RidgeCV, alpha grid biased high, LOO inside the draw
    "pca1_ridge", "pca2_ridge", "pca3_ridge", "pca5_ridge",
    "pls1", "pls2",
    "knn1", "knn3", "knn5",
    # TabPFN on PCA-reduced features. Added because the deployment regime is
    # 1-3 components (not 12), which is exactly where a better probe can earn
    # its keep: at 1-comp the best linear probe leaves ~37% of the variance
    # unexplained, whereas at 12-comp linear already reaches 0.989 and there
    # is no headroom. TabPFN needs no tuning (one forward pass over a learned
    # prior), which matters at n=20 where CV-based alpha selection is noise.
    # Skipped automatically if tabpfn isn't installed.
    "tabpfn_pca10", "tabpfn_pca20", "tabpfn_pca50",
)


def tabpfn_available() -> bool:
    try:
        import tabpfn  # noqa: F401
        return True
    except ImportError:
        return False


def make_fewshot_regressor(name: str, seed: int = 42, pca_basis=None):
    """
    Low-capacity probes for the few-shot regime.

    pca_basis: a *pre-fitted* sklearn PCA (fitted on UNLABELED data, i.e. all
    spectra's representations, no labels used). When supplied, the pca*_ridge
    probes project with it instead of fitting PCA on the tiny training draw.
    That is the transductive/"we have unlabeled spectra in deployment"
    variant; pass None for the strictly inductive variant. Both are reported
    -- see plan.md, never silently mix them.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import RidgeCV

    if name == "dummy":
        from sklearn.dummy import DummyRegressor
        return DummyRegressor(strategy="mean")

    if name == "ridge_strong":
        # alphas biased high: at n_train~20 the useful range is large-shrinkage
        return RidgeCV(alphas=np.logspace(0, 6, 25))

    if name.startswith("pca") and name.endswith("_ridge"):
        k = int(name[len("pca"):-len("_ridge")])
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 20))
        if pca_basis is not None:
            return Pipeline([("pca", _FrozenPCA(pca_basis, k)), ("ridge", ridge)])
        from sklearn.decomposition import PCA
        return Pipeline([("pca", PCA(n_components=k, random_state=seed)), ("ridge", ridge)])

    if name.startswith("pls"):
        from sklearn.cross_decomposition import PLSRegression
        return _PLSWrapper(int(name[len("pls"):]))

    if name.startswith("knn"):
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=int(name[len("knn"):]))

    if name.startswith("tabpfn_pca"):
        # TabPFN caps out around a few hundred features and its prior is over
        # low-dimensional tabular data, so the PCA reduction is mandatory, not
        # optional. NOTE the reduction is unsupervised: it maximizes variance,
        # not label-relevance, so a poor result here is ambiguous between "bad
        # probe" and "PCA discarded the label direction" -- study.py runs a
        # large-n linear probe on the same PCA-k features to disambiguate.
        from tabpfn import TabPFNRegressor
        k = int(name[len("tabpfn_pca"):])
        reg = TabPFNRegressor(device=_tabpfn_device(), ignore_pretraining_limits=True)
        if pca_basis is not None:
            return Pipeline([("pca", _FrozenPCA(pca_basis, k)), ("tabpfn", reg)])
        from sklearn.decomposition import PCA
        return Pipeline([("pca", PCA(n_components=k, random_state=seed)), ("tabpfn", reg)])

    raise ValueError(f"unknown few-shot probe {name!r}")


def _tabpfn_device() -> str:
    """TabPFN is a single forward pass, so the GPU is nearly free here -- and
    steps 2-5 are otherwise CPU-bound, so the GPUs sit idle."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class _FrozenPCA:
    """Applies a PCA basis fitted elsewhere (on unlabeled data); fit() is a
    no-op so the tiny labeled draw never re-fits the projection."""

    def __init__(self, pca, k):
        self.pca, self.k = pca, k

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.pca.transform(X)[:, : self.k]

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def get_params(self, deep=True):
        return {"pca": self.pca, "k": self.k}

    def set_params(self, **p):
        for key, val in p.items():
            setattr(self, key, val)
        return self


class _PLSWrapper:
    """PLSRegression with a 1-D predict(), and a guard for n_components > n_train."""

    def __init__(self, n_components):
        self.n_components = n_components
        self._m = None

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression
        k = min(self.n_components, X.shape[0] - 1, X.shape[1])
        k = max(k, 1)
        self._m = PLSRegression(n_components=k).fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self._m.predict(X)).ravel()

    def get_params(self, deep=True):
        return {"n_components": self.n_components}

    def set_params(self, **p):
        for key, val in p.items():
            setattr(self, key, val)
        return self
