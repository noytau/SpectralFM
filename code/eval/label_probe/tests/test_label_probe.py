import numpy as np
import pytest

from .. import features as feat
from .. import ladder as laddermod
from .. import readouts as ro
from .. import study
from ..canary import shuffled_label_canary
from ..normalize import fit_normalizer
from ..protocol import r2
from ..regressors import _PLSWrapper, make_fewshot_regressor, make_regressor


def test_r2_perfect_and_mean_baseline():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == pytest.approx(1.0)
    assert r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-9)


def test_make_wide_concatenates_components():
    X = np.arange(2 * 3 * 4).reshape(2, 3, 4)  # [N=2, K=3, D=4]
    out = feat.make_wide(X, [0, 2])
    assert out.shape == (2, 8)
    np.testing.assert_array_equal(out[0], np.concatenate([X[0, 0], X[0, 2]]))


def test_whiten_normalizer_fits_only_on_given_data():
    rng = np.random.default_rng(0)
    X_all = rng.normal(size=(200, 10))
    norm = fit_normalizer("whiten", X_all, seed=0)
    # transform on a disjoint set must not error or refit
    X_other = rng.normal(size=(5, 10))
    out = norm.transform(X_other)
    assert out.shape == (5, min(10, 200))


def test_pls_wrapper_is_fold_internal_and_never_sees_test_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20))
    y = X[:, 0] * 2.0
    m = _PLSWrapper(n_components=4)
    m.fit(X[:40], y[:40])
    pred = m.predict(X[40:])
    # fold-internal PLS on a clean linear signal should generalize well
    assert r2(y[40:], pred) > 0.5


def test_shuffled_label_canary_catches_a_leaky_pipeline():
    """A pipeline that fits its (supervised) transform on ALL labels before
    scoring must fail the canary; the same pipeline done fold-internally
    must pass."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 30))
    y = rng.normal(size=120)

    def leaky_fit_predict_oof(Xc, yc):
        from sklearn.cross_decomposition import PLSRegression
        # BUG: fits on the whole dataset (including what will be "held out")
        pls = PLSRegression(n_components=5).fit(Xc, yc)
        return np.asarray(pls.predict(Xc)).ravel()

    def honest_fit_predict_oof(Xc, yc):
        from sklearn.model_selection import KFold
        cv = KFold(5, shuffle=True, random_state=0)
        pred = np.zeros(len(yc))
        for tr, te in cv.split(Xc):
            m = _PLSWrapper(5)
            m.fit(Xc[tr], yc[tr])
            pred[te] = m.predict(Xc[te])
        return pred

    leaky = shuffled_label_canary(leaky_fit_predict_oof, X, y, seed=1)
    honest = shuffled_label_canary(honest_fit_predict_oof, X, y, seed=1)
    assert leaky["passed"] is False
    assert honest["passed"] is True


def test_regressor_registry_builds_every_named_regressor():
    for name in ("ols", "ridge", "ridgecv", "hgb", "knn", "pls", "dummy"):
        m = make_regressor(name)
        assert hasattr(m, "fit") and hasattr(m, "predict")


def test_fewshot_regressor_registry_builds_every_named_probe():
    rng = np.random.default_rng(0)
    X_all = rng.normal(size=(50, 12))
    from sklearn.decomposition import PCA
    basis = PCA(n_components=5).fit(X_all)
    for name in ("dummy", "ridge_strong", "pca2_ridge", "pls1", "knn3"):
        m = make_fewshot_regressor(name, pca_basis=basis if "pca" in name else None)
        assert hasattr(m, "fit") and hasattr(m, "predict")


def test_ladder_score_readout_returns_expected_shape():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    eval_idx = np.arange(200, 300)
    out = laddermod.score_readout(X, y, probe_fn=lambda: make_regressor("ridgecv"),
                                   eval_idx=eval_idx, n_trains=[20, 50])
    assert set(out.keys()) == {20, 50}
    assert out[50]["r2_median"] > out[20]["r2_median"] - 0.5  # sanity, not a tight bound


# ---------------------------------------------------------------------------
# Backbone-generality: readouts.py must never assume THIS backbone's shape
# (13 hidden_states layers, feature_extractor/feature_projection submodules)
# -- a different Transformer, tried by a different user, has neither.
# ---------------------------------------------------------------------------

class _FakeOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeBackboneNoFE:
    """A minimal stand-in for a Transformer encoder that has NEITHER
    `feature_extractor` nor `feature_projection` (unlike this repo's
    data2vec-audio backbone) and a layer count that is deliberately not 13,
    so a hardcoded assumption about either would fail loudly."""

    def __init__(self, n_layers=4, seq_len=6, d=8):
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.d = d

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_values, output_hidden_states=True):
        base = input_values[:, :self.d].unsqueeze(1).repeat(1, self.seq_len, 1)
        hs = tuple(base + 0.1 * i for i in range(self.n_layers))
        return _FakeOutput(hs)


def test_extract_bank_has_no_fe_taps_without_the_submodules():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    model = _FakeBackboneNoFE(n_layers=4)
    signals = rng.normal(size=(20, 245)).astype(np.float32)
    bank = ro.extract_bank(model, signals, device="cpu", batch_size=8)
    assert "fe" not in bank and "extract_features" not in bank
    assert set(bank) == {f"layer{i}" for i in range(4)}


def test_extract_bank_layer_count_is_whatever_the_backbone_has():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    model = _FakeBackboneNoFE(n_layers=7)
    signals = rng.normal(size=(20, 245)).astype(np.float32)
    bank = ro.extract_bank(model, signals, device="cpu", batch_size=8)
    assert set(bank) == {f"layer{i}" for i in range(7)}


def test_final_layer_stage_picks_the_actual_final_layer():
    """The conventional-tap reference must never hardcode a layer index --
    a different Transformer has a different depth."""
    bank = {"layer0": None, "layer1": None, "layer2": None}
    assert study._final_layer_stage(bank) == "layer2"

    bank_with_fe = {"fe": None, "extract_features": None, "layer0": None,
                    "layer1": None, "layer2": None, "layer3": None, "layer4": None}
    assert study._final_layer_stage(bank_with_fe) == "layer4"


def test_stage_display_name_is_backbone_general():
    # "layer0" is special-cased to "Projector" for THIS backbone (it's the
    # Transformer's input, not a block output -- see KNOWN_STAGE_DISPLAY_NAMES).
    # Every other layer index, including ones this backbone doesn't have (a
    # different Transformer might), must render generically, not KeyError.
    assert ro.stage_display_name("layer0") == "Projector"
    assert ro.stage_display_name("layer6") == "Transformer layer 6"
    assert ro.stage_display_name("layer99") == "Transformer layer 99"
    assert ro.stage_display_name("fe") == "FE (pre-LN)"
    assert ro.stage_display_name("some_unknown_stage") == "some_unknown_stage"
