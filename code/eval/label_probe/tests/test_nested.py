import numpy as np
import pytest

from .. import nested
from ..normalize import fit_normalizer


def _arms(n=80, d=12, n_arms=6, seed=0):
    rng = np.random.default_rng(seed)
    return {f"a{i}": rng.normal(size=(n, d)).astype(np.float32) for i in range(n_arms)}


def test_fold_transforms_match_fit_normalizer():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 40)) @ rng.normal(size=(40, 40))
    tr, te = X[:70], X[70:]
    out = nested.fold_transforms(tr, te, seed=0)
    assert set(out) == set(nested.NORMALIZERS)
    for name, (a, b) in out.items():
        ref = fit_normalizer(name, tr, seed=0)
        np.testing.assert_allclose(a, ref.transform(tr), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(b, ref.transform(te), rtol=1e-4, atol=1e-4)


def test_selection_over_many_recipes_does_not_inflate_noise():
    # Pure-noise labels, 6 arms x 12 recipes = 72 candidates. A best-of-72
    # picked on the scoring folds would come out positive; nested CV must not.
    arms = _arms()
    y = np.random.default_rng(1).normal(size=80)
    res = nested.nested_cv(arms, {"all": list(arms)}, y, n_repeats=2, seed=0)
    assert res["all"]["r2_mean"] < 0.05


def test_selection_finds_the_informative_arm():
    arms = _arms(n=120)
    y = arms["a3"] @ np.linspace(1, 2, arms["a3"].shape[1])
    res = nested.nested_cv(arms, {"all": list(arms)}, y, n_repeats=1, seed=0)
    assert res["all"]["r2_mean"] > 0.9
    assert all(c["arm"] == "a3" for c in res["all"]["chosen"])


def test_normalizer_and_probe_never_see_outer_test_rows(monkeypatch):
    arms = _arms(n=50, n_arms=2)
    y = np.random.default_rng(2).normal(size=50)
    row_id = {float(v): i for i, v in enumerate(arms["a0"][:, 0])}
    seen = []
    real = nested.fold_transforms

    def spy(X_tr, X_te, **kw):
        if X_tr.shape[1] == arms["a0"].shape[1] and float(X_tr[0, 0]) in row_id:
            seen.append(({row_id[float(v)] for v in X_tr[:, 0]},
                         {row_id[float(v)] for v in X_te[:, 0]}))
        return real(X_tr, X_te, **kw)

    monkeypatch.setattr(nested, "fold_transforms", spy)
    res = nested.nested_cv(arms, {"all": ["a0", "a1"]}, y, n_repeats=1, seed=0)
    outer_tests = [set(t) for t in res["test_folds"][0]]
    assert seen
    for tr, te in seen:
        assert not tr & te
        # every fit's training rows exclude one whole outer test fold
        assert any(not tr & ot for ot in outer_tests)


def test_paired_delta_of_identical_arms_is_zero():
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    p = y + rng.normal(size=(2, 100))
    d = nested.paired_delta(y, p, p.copy(), n_boot=50)
    assert d["delta"] == 0.0 and d["sd"] == 0.0


def test_paired_delta_is_tighter_than_combined_sd():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    base = y + rng.normal(scale=0.8, size=(2, 300))
    better = base - 0.1 * (base - y)          # a small, consistent improvement
    d = nested.paired_delta(y, better, base, n_boot=200)
    sa = nested.bootstrap_sd(y, better, n_boot=200)
    sb = nested.bootstrap_sd(y, base, n_boot=200)
    assert d["delta"] > 0
    assert d["sd"] < 0.5 * np.hypot(sa, sb)


def test_run_nested_on_a_bank():
    rng = np.random.default_rng(0)
    n = 60
    bank = {s: rng.normal(size=(n, 1, 10, 8)).astype(np.float32) for s in ("layer0", "layer1")}
    raw = rng.normal(size=(n, 1, 20)).astype(np.float32)
    y = bank["layer1"][:, 0].mean(axis=1) @ np.ones(8)
    res, oof = nested.run_nested(bank, raw, y, n_repeats=1, seed=0)
    assert set(res["families"]) == {"raw", "embedding", "layer0", "layer1"}
    assert res["families"]["embedding"]["r2_mean"] > res["families"]["raw"]["r2_mean"]
    assert len(res["raw_fixed_recipes"]) == len(nested.RECIPES)
    assert oof["embedding"].shape == (1, n)
