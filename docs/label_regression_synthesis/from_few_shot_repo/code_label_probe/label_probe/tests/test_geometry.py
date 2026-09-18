import numpy as np
import pytest

from eval.label_probe import geometry as geo


def test_normalizer_none_is_identity(tiny_reps):
    X, _ = tiny_reps
    out = geo.fit_normalizer("none", X).transform(X)
    np.testing.assert_allclose(out, X, rtol=1e-6)


def test_standardize_gives_unit_variance_columns(tiny_reps):
    X, _ = tiny_reps
    out = geo.fit_normalizer("standardize", X).transform(X)
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-5)


def test_standardize_survives_a_constant_column():
    X = np.hstack([np.random.default_rng(0).normal(size=(50, 3)),
                   np.full((50, 1), 7.0)])
    out = geo.fit_normalizer("standardize", X).transform(X)
    assert np.all(np.isfinite(out)), "constant column must not produce NaN/inf"


def test_l2_gives_unit_norm_rows(tiny_reps):
    X, _ = tiny_reps
    out = geo.fit_normalizer("l2", X).transform(X)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_whiten_decorrelates(tiny_reps):
    X, _ = tiny_reps
    out = geo.fit_normalizer("whiten", X).transform(X)
    cov = np.cov(out, rowvar=False)
    np.testing.assert_allclose(np.diag(cov), 1.0, atol=1e-2)
    off = cov - np.diag(np.diag(cov))
    assert np.abs(off).max() < 1e-2


def test_whiten_caps_rank_when_features_exceed_max_rank():
    """Without a cap, k = min(n, d) -- for d > WHITEN_MAX_RANK that fits as
    many directions as samples, whitening trailing near-zero eigenvalues
    into unit-variance noise. The transform must retain at most
    WHITEN_MAX_RANK columns."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 400)).astype(np.float32)
    out = geo.fit_normalizer("whiten", X).transform(X)
    assert out.shape[1] <= geo.WHITEN_MAX_RANK


def test_normalizer_is_fitted_on_all_and_applied_unchanged(tiny_reps):
    """The transform must NOT re-fit on the rows it is given -- that would
    leak the eval set's statistics into a few-shot draw."""
    X, _ = tiny_reps
    nrm = geo.fit_normalizer("standardize", X)
    sub = X[:5]
    a = nrm.transform(sub)
    b = nrm.transform(X)[:5]
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_reducer_shapes_and_k(tiny_reps):
    X, _ = tiny_reps          # 24 features
    r10 = geo.fit_reducer("pca10", X)
    assert r10.k == 10 and r10.transform(X).shape == (200, 10)
    # k larger than the feature count must clamp, not crash
    r50 = geo.fit_reducer("pca50", X)
    assert r50.k == min(50, 24) and r50.transform(X).shape[1] == r50.k
    assert geo.fit_reducer("none", X).transform(X).shape == (200, 24)


def test_unknown_names_raise(tiny_reps):
    X, _ = tiny_reps
    with pytest.raises(ValueError, match="unknown normalizer"):
        geo.fit_normalizer("nope", X)
    with pytest.raises(ValueError, match="unknown reducer"):
        geo.fit_reducer("nope", X)


def test_readout_label():
    r = geo.Readout(family="emb", stage="layer8", pooling="mean_std",
                    normalizer="whiten")
    assert r.label() == "emb|layer8|mean_std|whiten"


def test_row_index_round_trips(tiny_reps):
    X, _ = tiny_reps
    idx = geo._RowIndex(X)
    assert idx.lookup(X[7]) == 7
    assert idx.lookup(X[199]) == 199
    with pytest.raises(KeyError):
        idx.lookup(np.zeros(X.shape[1], dtype=X.dtype))


@pytest.mark.parametrize("name", ["knn_cos1", "knn_cos5", "gp_rbf", "krr_rbf",
                                  "graph_prop", "graph_prop50",
                                  "ridge_strong", "pca5_ridge"])
def test_probe_fits_and_predicts_at_n20(tiny_reps, name):
    """Every probe must survive n_train=20 in 24 dims and return one finite
    prediction per row -- the regime the whole study lives in."""
    X, y = tiny_reps
    rng = np.random.default_rng(3)
    tr = rng.choice(200, 20, replace=False)
    m = geo.make_probe(name, seed=42, X_all=X)
    m.fit(X[tr], y[tr])
    pred = np.asarray(m.predict(X[:50])).ravel()
    assert pred.shape == (50,)
    assert np.all(np.isfinite(pred))


def test_geometry_probes_beat_the_mean_on_a_planted_signal(tiny_reps):
    """Sanity floor: with a strong planted linear signal, a geometry probe at
    n=50 must beat predicting the mean. Catches a probe that silently
    predicts a constant."""
    from eval.label_probe.protocol import r2

    X, y = tiny_reps
    rng = np.random.default_rng(4)
    tr = rng.choice(200, 50, replace=False)
    te = np.setdiff1d(np.arange(200), tr)
    for name in ("knn_cos5", "gp_rbf", "graph_prop"):
        m = geo.make_probe(name, seed=42, X_all=X)
        m.fit(X[tr], y[tr])
        score = r2(y[te], np.asarray(m.predict(X[te])).ravel())
        assert score > 0.0, f"{name} failed to beat the mean: R2={score:.3f}"


def test_graph_prop_is_transductive_over_x_all(tiny_reps):
    """graph_prop must use X_all's geometry: it should still predict sensibly
    for rows it never saw in training, via the precomputed graph."""
    from eval.label_probe.protocol import r2

    X, y = tiny_reps
    m = geo.make_probe("graph_prop", seed=42, X_all=X)
    m.fit(X[:20], y[:20])
    assert r2(y[100:150], np.asarray(m.predict(X[100:150])).ravel()) > -1.0


def test_panels_are_disjointly_defined():
    assert set(geo.SCREEN_PANEL).issubset(set(geo.CONFIRM_PANEL))
    assert "tabpfn_pca50" in geo.CONFIRM_PANEL
    assert "tabpfn_pca50" not in geo.SCREEN_PANEL   # too slow for 468 readouts


def test_graph_prop50_is_confirm_only():
    """graph_prop50 (n_iter=50, near-convergence) is the deep-propagation
    variant added by controller ruling alongside graph_prop (n_iter=1) --
    whether deep diffusion helps is an empirical question for the real
    embeddings, not something tiny_reps's structureless graph can settle, so
    it is confirm-only rather than dropped or forced into the screen panel."""
    assert "graph_prop50" in geo.CONFIRM_PANEL
    assert "graph_prop50" not in geo.SCREEN_PANEL
