"""
Regressor registry for the label probe.

Two panels: the large-n panel (`make_regressor`) for the full-pool asymptote,
and the few-shot panel (`make_fewshot_regressor`) for n_train in the tens —
deliberately low-capacity, since ordinary coefficient-fitting probes are
unidentifiable at p >> n.
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
    """name in: 'ols', 'ridge', 'ridgecv', 'xgb', 'hgb', 'knn', 'pls', 'dummy'"""
    if name == "ols":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)
    if name == "ridgecv":
        from sklearn.linear_model import RidgeCV
        # cv=None (the default) uses sklearn's efficient generalized/leave-
        # one-out CV via a single SVD, not a naive grid search — an explicit
        # cv=5 instead refits Ridge from scratch for every (alpha, fold)
        # pair, ~100x more solves for no accuracy benefit at this n/d.
        return RidgeCV(alphas=np.logspace(-3, 3, 20))
    if name == "xgb":
        if _xgb_available():
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                random_state=seed, n_jobs=-1,
            )
        name = "hgb"  # zero-dep fallback
    if name == "hgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=seed)
    if name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=5)
    if name == "pls":
        return _PLSWrapper(64)
    if name == "dummy":
        from sklearn.dummy import DummyRegressor
        return DummyRegressor(strategy="mean")
    raise ValueError(f"unknown regressor {name!r}")


REGRESSOR_PANEL = ("ols", "ridge", "ridgecv", "pls", "hgb")

# Few-shot panel: dummy is the floor and a real contender at n<=20.
FEWSHOT_PANEL = ("dummy", "ridge_strong", "pca2_ridge", "pca5_ridge",
                 "pls1", "pls2", "knn3")


def make_fewshot_regressor(name: str, seed: int = 42, pca_basis=None):
    """
    pca_basis: an optional *pre-fitted* sklearn PCA (fitted on the UNLABELED
    corpus). When supplied, the pca*_ridge probes project with it instead of
    fitting PCA on the tiny training draw (the transductive variant —
    legitimate since deployment has the unlabeled corpus available). Pass
    None for the strictly inductive variant.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline

    if name == "dummy":
        from sklearn.dummy import DummyRegressor
        return DummyRegressor(strategy="mean")

    if name == "ridge_strong":
        # alphas biased high: at n_train~20 the useful shrinkage range is large.
        return RidgeCV(alphas=np.logspace(0, 6, 25))

    if name.startswith("pca") and name.endswith("_ridge"):
        k = int(name[len("pca"):-len("_ridge")])
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 20))
        if pca_basis is not None:
            return Pipeline([("pca", _FrozenPCA(pca_basis, k)), ("ridge", ridge)])
        return Pipeline([("pca", _SafePCA(k, seed=seed)), ("ridge", ridge)])

    if name.startswith("pls"):
        return _PLSWrapper(int(name[len("pls"):]))

    if name.startswith("knn"):
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=int(name[len("knn"):]))

    raise ValueError(f"unknown few-shot probe {name!r}")


class _SafePCA:
    """PCA that caps n_components to what the training draw can actually
    support (min(requested, n_samples, n_features)) at fit time, instead of
    sklearn's PCA -- which fixes n_components at construction and raises if
    a later, smaller training draw can't support it. Needed here because the
    label-efficiency ladder reuses the same probe across n_train from 10 up
    to the full pool; a probe picked by a large-n screen (e.g. pca64_ridge)
    would otherwise crash the moment the ladder reaches an n_train below the
    requested component count."""

    def __init__(self, n_components, seed=42):
        self.n_components = n_components
        self.seed = seed
        self._pca = None

    def fit(self, X, y=None):
        from sklearn.decomposition import PCA
        k = max(1, min(self.n_components, X.shape[0], X.shape[1]))
        self._pca = PCA(n_components=k, random_state=self.seed).fit(X)
        return self

    def transform(self, X):
        return self._pca.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "seed": self.seed}

    def set_params(self, **p):
        for key, val in p.items():
            setattr(self, key, val)
        return self


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
    """PLSRegression with a 1-D predict() and a guard for
    n_components > n_train. Fits strictly inside .fit() on whatever X, y it
    is given — never on combined train+test data. A caller that fits this on
    the full dataset before cross-validating is the one thing that can leak;
    this wrapper itself cannot."""

    def __init__(self, n_components):
        self.n_components = n_components
        self._m = None

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression
        k = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
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
