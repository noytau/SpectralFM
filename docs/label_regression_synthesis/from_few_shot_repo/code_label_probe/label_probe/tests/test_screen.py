import numpy as np
import pytest

from eval.label_probe import screen as scr


def test_split_is_disjoint_and_complete():
    s = scr.make_split(4716, n_eval_a=500, n_eval_b=500, seed=42)
    a, b, p = s["eval_a"], s["eval_b"], s["pool"]
    assert len(a) == 500 and len(b) == 500 and len(p) == 4716 - 1000
    allidx = np.concatenate([a, b, p])
    assert len(np.unique(allidx)) == 4716
    assert not set(a) & set(b)
    assert not set(a) & set(p)


def test_split_is_deterministic():
    a = scr.make_split(4716, seed=42)["eval_a"]
    b = scr.make_split(4716, seed=42)["eval_a"]
    np.testing.assert_array_equal(a, b)


def test_draws_are_shared_and_reproducible():
    pool = np.arange(1000)
    d1 = scr.draw_indices(pool, n_train=20, n_draws=5, seed=7)
    d2 = scr.draw_indices(pool, n_train=20, n_draws=5, seed=7)
    assert len(d1) == 5 and all(len(d) == 20 for d in d1)
    for x, y in zip(d1, d2):
        np.testing.assert_array_equal(x, y)
    # no repeats inside a draw, and draws differ from each other
    assert len(np.unique(d1[0])) == 20
    assert not np.array_equal(d1[0], d1[1])


def test_participation_ratio_detects_anisotropy():
    rng = np.random.default_rng(0)
    iso = rng.normal(size=(500, 20))
    aniso = iso.copy()
    aniso[:, 0] *= 100.0            # one rogue direction
    d_iso = scr.readout_diagnostics(iso, rng.normal(size=500))
    d_ani = scr.readout_diagnostics(aniso, rng.normal(size=500))
    assert d_ani["participation_ratio"] < d_iso["participation_ratio"] / 2
    assert d_iso["effective_rank"] > d_ani["effective_rank"]


def test_diagnostics_report_a_topk_ceiling(tiny_reps):
    X, y = tiny_reps
    d = scr.readout_diagnostics(X, y)
    assert set(d["whitened_topk_r2"]) == {5, 10, 20, 50}
    assert d["ridgecv_full_r2"] > 0.8          # planted linear signal
    assert all(np.isfinite(v) for v in d["whitened_topk_r2"].values())


def test_score_readout_returns_paired_per_draw_scores(tiny_reps):
    X, y = tiny_reps
    split = scr.make_split(200, n_eval_a=40, n_eval_b=40, seed=1)
    draws = {20: scr.draw_indices(split["pool"], 20, 4, seed=1)}
    out = scr.score_readout(X, y, split, probes=("ridge_strong", "knn_cos5"),
                            n_trains=(20,), draws_by_n=draws, eval_key="eval_a")
    assert set(out) == {"ridge_strong", "knn_cos5"}
    cell = out["ridge_strong"][20]
    assert len(cell["r2_per_draw"]) == 4
    assert "r2_median" in cell and "frac_positive_r2" in cell
    for key in ("r2_p10", "r2_p90", "mae_median", "spearman_median"):
        assert key in cell
        assert np.isfinite(cell[key])


def test_score_readout_degrades_gracefully_on_unknown_probe(tiny_reps):
    X, y = tiny_reps
    split = scr.make_split(200, n_eval_a=40, n_eval_b=40, seed=1)
    draws = {20: scr.draw_indices(split["pool"], 20, 4, seed=1)}
    out = scr.score_readout(X, y, split, probes=("not_a_real_probe",),
                            n_trains=(20,), draws_by_n=draws, eval_key="eval_a")
    cell = out["not_a_real_probe"][20]
    assert cell["n_failed_draws"] == 4
    assert np.isnan(cell["r2_median"])
    assert cell["frac_positive_r2"] == 0.0
    assert np.isnan(cell["mae_median"])
    assert np.isnan(cell["spearman_median"])


def test_score_readout_uses_the_requested_eval_set(tiny_reps):
    """eval_a and eval_b are different spectra, so the same probe on the same
    draws must not give byte-identical scores -- a guard against silently
    scoring everything on one set."""
    X, y = tiny_reps
    split = scr.make_split(200, n_eval_a=40, n_eval_b=40, seed=1)
    draws = {20: scr.draw_indices(split["pool"], 20, 4, seed=1)}
    a = scr.score_readout(X, y, split, ("ridge_strong",), (20,), draws, "eval_a")
    b = scr.score_readout(X, y, split, ("ridge_strong",), (20,), draws, "eval_b")
    assert a["ridge_strong"][20]["r2_per_draw"] != b["ridge_strong"][20]["r2_per_draw"]


def test_score_readout_uses_injected_probe_factory(tiny_reps):
    """A custom factory must be used for every probe construction, so a study
    can supply probes geometry.make_probe does not know how to build."""
    from sklearn.dummy import DummyRegressor

    X, y = tiny_reps
    split = scr.make_split(200, n_eval_a=40, n_eval_b=40, seed=1)
    draws = {20: scr.draw_indices(split["pool"], 20, 3, seed=1)}
    seen = []

    def factory(name, seed=42, X_all=None):
        seen.append(name)
        return DummyRegressor(strategy="mean")

    out = scr.score_readout(X, y, split, ("not_a_real_probe",), (20,), draws,
                            "eval_a", probe_factory=factory)
    assert seen == ["not_a_real_probe"], "factory must be called once per (probe, n_train) cell"
    cell = out["not_a_real_probe"][20]
    assert cell["n_failed_draws"] == 0, "the injected DummyRegressor must fit fine"
    assert len(cell["r2_per_draw"]) == 3


def test_score_readout_default_factory_unchanged(tiny_reps):
    """Omitting probe_factory must keep the existing geometry.make_probe path."""
    X, y = tiny_reps
    split = scr.make_split(200, n_eval_a=40, n_eval_b=40, seed=1)
    draws = {20: scr.draw_indices(split["pool"], 20, 3, seed=1)}
    a = scr.score_readout(X, y, split, ("ridge_strong",), (20,), draws, "eval_a")
    b = scr.score_readout(X, y, split, ("ridge_strong",), (20,), draws, "eval_a",
                          probe_factory=None)
    assert a["ridge_strong"][20]["r2_per_draw"] == b["ridge_strong"][20]["r2_per_draw"]
