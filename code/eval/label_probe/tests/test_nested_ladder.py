import numpy as np

from .. import nested, nested_ladder as nl


def _arms(n=120, d=10, seed=0):
    rng = np.random.default_rng(seed)
    return {"raw": rng.normal(size=(n, d)).astype(np.float32),
            "block": rng.normal(size=(n, d)).astype(np.float32)}


def test_ladder_never_fits_on_held_out_rows(monkeypatch):
    arms = _arms(n=60)
    y = np.random.default_rng(1).normal(size=60)
    row_id = {float(v): i for i, v in enumerate(arms["raw"][:, 0])}
    seen = []
    real = nl.fold_transforms

    def spy(X_tr, X_te, **kw):
        if float(X_tr[0, 0]) in row_id:
            seen.append(({row_id[float(v)] for v in X_tr[:, 0]}, {row_id[float(v)] for v in X_te[:, 0]}))
        return real(X_tr, X_te, **kw)

    # both the inner selection (nested._inner_scores) and the refit go through fold_transforms
    monkeypatch.setattr(nl, "fold_transforms", spy)
    monkeypatch.setattr(nested, "fold_transforms", spy)
    nl.nested_ladder(arms, y, n_trains=(10, 20), seed=0, max_draws=2)
    from sklearn.model_selection import KFold
    tests = [set(te) for _, te in KFold(5, shuffle=True, random_state=0).split(y)]
    assert seen
    for tr, te in seen:
        assert not tr & te
        assert any(not tr & t for t in tests)


def test_ladder_on_shuffled_labels_stays_near_zero():
    arms = _arms()
    y = np.random.default_rng(2).normal(size=120)
    out = nl.nested_ladder(arms, y, n_trains=(10, 50), seed=0, max_draws=6)
    for a in ("raw", "block"):
        for n, v in out["arms"][a].items():
            assert v["median"] < 0.05, (a, n, v)


def test_ladder_rungs_and_paired_gaps():
    arms = _arms()
    y = arms["block"][:, 0] * 2.0
    out = nl.nested_ladder(arms, y, n_trains=(10, 50), seed=0, max_draws=2)
    assert out["rungs"] == [10, 50, 96]
    assert out["draws"] == {"10": 10, "50": 10, "96": 5}
    assert out["gaps"]["block"]["96"]["median"] > 0.5
