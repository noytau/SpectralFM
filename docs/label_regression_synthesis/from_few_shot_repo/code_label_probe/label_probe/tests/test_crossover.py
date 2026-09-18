import numpy as np
import pytest

from eval.label_probe import crossover as cx
from eval.label_probe import geometry as geo


def test_ladder_and_draw_budget_are_consistent():
    assert cx.N_LADDER == (20, 50, 100, 200, 500, 1000, 2000)
    assert set(cx.DRAWS_BY_N) == set(cx.N_LADDER)
    # draws must not increase with n -- cost rises and variance falls
    counts = [cx.DRAWS_BY_N[n] for n in cx.N_LADDER]
    assert counts == sorted(counts, reverse=True)
    # every rung must be drawable without replacement from the 3716-spectrum pool
    assert max(cx.N_LADDER) <= 3716


def test_low_rungs_reproduce_step6_panel():
    """n<=50 must use step 6's CONFIRM_PANEL verbatim so the bottom of the
    ladder is comparable with the published step-6 cells."""
    assert cx.panel_for_n(20) == geo.CONFIRM_PANEL
    assert cx.panel_for_n(50) == geo.CONFIRM_PANEL


def test_mid_rungs_drop_tabpfn_and_add_large_n_probes():
    p = cx.panel_for_n(200)
    assert "tabpfn_pca50" not in p
    assert "ridgecv" in p and "pls64" in p
    assert "gp_rbf" in p, "exact GP is still affordable at n=200"


def test_high_rungs_drop_cubic_probes():
    p = cx.panel_for_n(2000)
    for cubic in ("gp_rbf", "krr_rbf", "tabpfn_pca50"):
        assert cubic not in p, f"{cubic} is infeasible at n=2000"
    assert "ridgecv" in p and "pls64" in p
    assert "ridge_strong" not in p, (
        "ridge_strong's alpha grid is biased high for n=20 and would "
        "over-regularize at n=2000, understating both arms")


def test_every_panel_is_nonempty_and_buildable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    for n in cx.N_LADDER:
        p = cx.panel_for_n(n)
        assert len(p) >= 3, f"panel at n={n} is too thin to pick a winner from"
        for name in p:
            # tabpfn* is skipped here: constructing it can trigger a model-
            # weights download, which would make the unit suite network-
            # dependent and slow. Its panel MEMBERSHIP (present at n<=50,
            # absent above) is still checked by the other tests in this file.
            if name.startswith("tabpfn"):
                continue
            probe = cx.make_probe_n(name, seed=42, X_all=X)
            assert hasattr(probe, "fit") and hasattr(probe, "predict"), name


def test_large_n_extras_route_to_make_regressor():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 6))
    y = X[:, 0] * 2.0 + 0.1 * rng.normal(size=80)
    for name in cx.LARGE_N_EXTRAS:
        m = cx.make_probe_n(name, seed=42, X_all=X)
        m.fit(X[:60], y[:60])
        pred = np.asarray(m.predict(X[60:])).ravel()
        assert pred.shape == (20,) and np.all(np.isfinite(pred))


def test_candidates_cover_both_arms_and_the_historical_reference():
    for n_comp in (1, 2, 3):
        cands = cx.CANDIDATES[n_comp]
        labels = [r.label() for r in cands]
        assert len(labels) == len(set(labels)), f"duplicate candidate at {n_comp}-comp"
        emb = [r for r in cands if r.family == "emb"]
        raw = [r for r in cands if r.family == "raw"]
        assert len(raw) == 4
        # 6 step-6 winners plus the historical reference arm
        assert len(emb) == 7
        assert cx.HISTORICAL in cands
        # the historical arm is the tap all prior work used
        assert cx.HISTORICAL.stage == "layer12" and cx.HISTORICAL.pooling == "mean"


def test_candidate_stages_are_real_bank_stages():
    from eval.label_probe import readouts as ro
    for n_comp in (1, 2, 3):
        for r in cx.CANDIDATES[n_comp]:
            if r.family == "emb":
                assert r.stage in ro.BANK_STAGES, r.stage
                assert r.pooling in ro.POOLINGS, r.pooling
            else:
                assert r.stage in ("raw", "z", "moments"), r.stage
            assert r.normalizer in geo.NORMALIZERS, r.normalizer


def _rungs(medians, spread=0.02, n=60, seed=0):
    """Build a per_n dict whose paired deltas have the requested medians."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, m in zip(cx.N_LADDER, medians):
        raw = rng.normal(0.30, spread, n)
        emb = raw + m
        out[k] = {"emb_per_draw": list(emb), "raw_per_draw": list(raw)}
    return out


def test_crossover_found_and_bracketed():
    # deltas go negative -> positive between n=200 and n=500 and stay positive
    r = cx.estimate_crossover(_rungs([-0.30, -0.20, -0.10, -0.04, +0.05, +0.12, +0.18]))
    assert r["crossed"] is True
    assert r["n_cross"] == 500, "smallest won rung whose successors are all won"
    assert 200 < r["n_cross_interp"] < 500
    lo, hi = r["ci"]
    assert 200 <= lo <= r["n_cross_interp"] <= hi <= 500


def test_no_crossover_reports_best_delta_instead():
    r = cx.estimate_crossover(_rungs([-0.40, -0.35, -0.30, -0.25, -0.20, -0.16, -0.12]))
    assert r["crossed"] is False
    assert r["n_cross"] is None and r["n_cross_interp"] is None and r["ci"] is None
    assert r["best_delta"] == pytest.approx(-0.12, abs=0.02)
    assert r["best_delta_n"] == 2000


def test_transient_win_is_not_a_crossing():
    """A single won rung followed by a lost one must not be reported as the
    crossing -- persistence is what separates a crossing from rung noise."""
    r = cx.estimate_crossover(_rungs([-0.30, -0.20, +0.06, -0.08, +0.05, +0.12, +0.18]))
    assert r["crossed"] is True
    assert r["n_cross"] == 500, "the transient win at n=100 must be skipped"
    assert r["rungs"][100]["won"] is True and r["rungs"][200]["won"] is False


def test_win_requires_bootstrap_confidence_not_just_a_positive_median():
    """A hair-positive median with deltas straddling zero must not count."""
    rng = np.random.default_rng(3)
    per_n = {}
    for k in cx.N_LADDER:
        raw = rng.normal(0.30, 0.02, 60)
        emb = raw + rng.normal(0.001, 0.40, 60)   # median ~0, huge spread
        per_n[k] = {"emb_per_draw": list(emb), "raw_per_draw": list(raw)}
    r = cx.estimate_crossover(per_n)
    for k in cx.N_LADDER:
        assert r["rungs"][k]["boot_frac_positive"] < 0.90


def test_all_rungs_won_crosses_at_the_first_rung():
    r = cx.estimate_crossover(_rungs([0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]))
    assert r["crossed"] is True and r["n_cross"] == cx.N_LADDER[0]
    assert r["n_cross_interp"] == float(cx.N_LADDER[0]), (
        "with no losing rung to interpolate from, the estimate is the first rung")
    assert r["ci"] == [float(cx.N_LADDER[0]), float(cx.N_LADDER[0])]


def test_failed_draws_are_counted_not_silently_dropped():
    per_n = _rungs([-0.30, -0.20, -0.10, -0.04, +0.05, +0.12, +0.18])
    e = list(per_n[500]["emb_per_draw"]); e[0] = float("-inf"); e[1] = float("nan")
    per_n[500]["emb_per_draw"] = e
    r = cx.estimate_crossover(per_n)
    assert r["rungs"][500]["n_dropped"] == 2
    assert r["rungs"][500]["n_pairs"] == 58


def test_estimator_survives_an_all_failed_rung():
    per_n = _rungs([-0.30, -0.20, -0.10, -0.04, +0.05, +0.12, +0.18])
    per_n[1000]["emb_per_draw"] = [float("-inf")] * 60
    r = cx.estimate_crossover(per_n)
    assert r["rungs"][1000]["n_pairs"] == 0
    assert r["rungs"][1000]["won"] is False
    assert np.isnan(r["rungs"][1000]["delta_median"])


def _write_tiny_bank(tmp_path, n=1200, k=3, d=4):
    """A synthetic bank in exactly load_bank_cache's layout: no GPU, no checkpoint.

    n=1200, not step 6's smaller convention: screen.make_split defaults to
    n_eval_a=500, n_eval_b=500, so a smaller n would leave the training pool
    empty (or too small for the largest monkeypatched rung below) and
    draw_indices would raise immediately.
    """
    import numpy as np
    from eval.label_probe import readouts as ro
    rng = np.random.default_rng(7)
    y = rng.normal(size=n)
    payload = {}
    for st in ("layer12", "layer1", "layer2"):
        arr = rng.normal(size=(n, k, len(ro.POOL_STATS), d)).astype(np.float32)
        arr[:, 0, 0, 0] += 1.2 * y          # a recoverable signal
        payload[f"bank__{st}"] = arr
    raw = rng.normal(size=(n, k, 245)).astype(np.float32)
    raw[:, 0, 0] += 2.0 * y
    payload["input_raw"] = raw
    payload["input_z"] = raw
    payload["y"] = y
    payload["_meta"] = np.array([repr({"checkpoint": "tiny", "comps": (0, 1, 2),
                                       "n": int(n), "pool_stats": ro.POOL_STATS})])
    p = tmp_path / "bank.npz"
    np.savez(str(p), **payload)
    return str(p)


def test_run_crossover_end_to_end_on_a_synthetic_bank(tmp_path, monkeypatch):
    """Orchestration only: the grid, per-rung selection, both arms scored on
    eval_b, the historical arm present, and the JSON written."""
    from eval.label_probe import crossover as cxm

    bank = _write_tiny_bank(tmp_path)
    # keep it fast: two rungs, two cheap probes, three candidates
    monkeypatch.setattr(cxm, "N_LADDER", (20, 50))
    monkeypatch.setattr(cxm, "DRAWS_BY_N", {20: 3, 50: 3})
    monkeypatch.setattr(cxm, "panel_for_n", lambda n: ("ridge_strong", "knn_cos5"))
    monkeypatch.setattr(cxm, "CANDIDATES", {1: [
        geo.Readout("emb", "layer2", "mean", "none"),
        cxm.HISTORICAL,
        geo.Readout("raw", "raw", "-", "none"),
    ]})

    res = cxm.run_crossover(bank, str(tmp_path), comp_counts=(1,), seed=42)

    assert res["ladder"] == [20, 50]
    per_n = res["per_comp"]["1"]["per_n"] if "1" in res["per_comp"] else res["per_comp"][1]["per_n"]
    for n in (20, 50):
        cell = per_n[n] if n in per_n else per_n[str(n)]
        for arm in ("emb", "raw", "historical"):
            assert arm in cell, f"{arm} missing at n={n}"
            assert "r2_per_draw" in cell[arm] and len(cell[arm]["r2_per_draw"]) == 3
            assert "readout" in cell[arm] and "probe" in cell[arm]
    assert (tmp_path / "step7_results.json").exists()
    assert "crossings" in res


def test_run_crossover_scores_reportables_on_eval_b_only(tmp_path, monkeypatch):
    """Selection may read eval_a; every reported cell must come from eval_b."""
    from eval.label_probe import crossover as cxm
    from eval.label_probe import screen as scrm

    bank = _write_tiny_bank(tmp_path)
    monkeypatch.setattr(cxm, "N_LADDER", (20,))
    monkeypatch.setattr(cxm, "DRAWS_BY_N", {20: 2})
    monkeypatch.setattr(cxm, "panel_for_n", lambda n: ("ridge_strong",))
    monkeypatch.setattr(cxm, "CANDIDATES", {1: [
        geo.Readout("emb", "layer2", "mean", "none"),
        cxm.HISTORICAL,
        geo.Readout("raw", "raw", "-", "none"),
    ]})

    keys = []
    orig = scrm.score_readout

    def spy(*a, **kw):
        keys.append(kw.get("eval_key", a[6] if len(a) > 6 else None))
        return orig(*a, **kw)

    monkeypatch.setattr(scrm, "score_readout", spy)
    cxm.run_crossover(bank, str(tmp_path), comp_counts=(1,), seed=42)
    assert "eval_a" in keys, "selection pass must run on eval_a"
    assert "eval_b" in keys, "reporting pass must run on eval_b"


def test_run_crossover_raises_on_an_empty_candidate_pool(tmp_path, monkeypatch):
    """An arm with no candidates in CANDIDATES must fail loudly and name the
    arm, not IndexError into nothing (pool_of[0] on an empty list)."""
    from eval.label_probe import crossover as cxm

    bank = _write_tiny_bank(tmp_path)
    monkeypatch.setattr(cxm, "N_LADDER", (20,))
    monkeypatch.setattr(cxm, "DRAWS_BY_N", {20: 2})
    monkeypatch.setattr(cxm, "panel_for_n", lambda n: ("ridge_strong",))
    # No raw-family readout at all for this component count.
    monkeypatch.setattr(cxm, "CANDIDATES", {1: [
        geo.Readout("emb", "layer2", "mean", "none"),
        cxm.HISTORICAL,
    ]})

    with pytest.raises(ValueError, match="raw"):
        cxm.run_crossover(bank, str(tmp_path), comp_counts=(1,), seed=42)


def test_run_crossover_sel_draws_overrides_selection_only(tmp_path, monkeypatch):
    """sel_draws must shrink only the eval_a (selection) draw count; the
    eval_b (reporting) cells must still carry DRAWS_BY_N-many entries."""
    from eval.label_probe import crossover as cxm
    from eval.label_probe import screen as scrm

    bank = _write_tiny_bank(tmp_path)
    monkeypatch.setattr(cxm, "N_LADDER", (20,))
    monkeypatch.setattr(cxm, "DRAWS_BY_N", {20: 5})
    monkeypatch.setattr(cxm, "panel_for_n", lambda n: ("ridge_strong",))
    monkeypatch.setattr(cxm, "CANDIDATES", {1: [
        geo.Readout("emb", "layer2", "mean", "none"),
        cxm.HISTORICAL,
        geo.Readout("raw", "raw", "-", "none"),
    ]})

    seen = []
    orig = scrm.score_readout

    def spy(*a, **kw):
        eval_key = kw.get("eval_key", a[6] if len(a) > 6 else None)
        draws_by_n = kw.get("draws_by_n", a[5] if len(a) > 5 else None)
        seen.append((eval_key, len(draws_by_n[20])))
        return orig(*a, **kw)

    monkeypatch.setattr(scrm, "score_readout", spy)
    res = cxm.run_crossover(bank, str(tmp_path), comp_counts=(1,), seed=42,
                            sel_draws=2)

    sel_lens = {n for ek, n in seen if ek == "eval_a"}
    rep_lens = {n for ek, n in seen if ek == "eval_b"}
    assert sel_lens == {2}, f"selection draws must be sel_draws=2: {sel_lens}"
    assert rep_lens == {5}, f"reporting draws must stay at DRAWS_BY_N: {rep_lens}"

    per_n = res["per_comp"]["1"]["per_n"]
    cell = per_n["20"] if "20" in per_n else per_n[20]
    for arm in ("emb", "raw", "historical"):
        assert len(cell[arm]["r2_per_draw"]) == 5, (
            f"{arm} reporting draws must not be shrunk by sel_draws")


def test_run_crossover_sel_top_restricts_selection_pool(tmp_path, monkeypatch):
    """sel_top_emb/sel_top_raw must restrict which candidates are scored
    during selection, while HISTORICAL is still carried as its own arm."""
    from eval.label_probe import crossover as cxm
    from eval.label_probe import screen as scrm

    bank = _write_tiny_bank(tmp_path)
    monkeypatch.setattr(cxm, "N_LADDER", (20,))
    monkeypatch.setattr(cxm, "DRAWS_BY_N", {20: 2})
    monkeypatch.setattr(cxm, "panel_for_n", lambda n: ("ridge_strong",))
    monkeypatch.setattr(cxm, "CANDIDATES", {1: [
        geo.Readout("emb", "layer2", "mean", "none"),
        geo.Readout("emb", "layer1", "mean", "none"),
        geo.Readout("emb", "layer0", "mean", "none"),
        cxm.HISTORICAL,
        geo.Readout("raw", "raw", "-", "none"),
        geo.Readout("raw", "z", "-", "none"),
        geo.Readout("raw", "raw", "-", "standardize"),
    ]})

    calls = {"eval_a": 0, "eval_b": 0}
    orig = scrm.score_readout

    def spy(*a, **kw):
        eval_key = kw.get("eval_key", a[6] if len(a) > 6 else None)
        calls[eval_key] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(scrm, "score_readout", spy)
    res = cxm.run_crossover(bank, str(tmp_path), comp_counts=(1,), seed=42,
                            sel_top_emb=1, sel_top_raw=1)

    # one rung: selection scores exactly 1 emb + 1 raw + HISTORICAL = 3
    assert calls["eval_a"] == 3, calls
    # reporting always scores exactly the 3 chosen arms
    assert calls["eval_b"] == 3, calls

    per_n = res["per_comp"]["1"]["per_n"]
    cell = per_n["20"] if "20" in per_n else per_n[20]
    assert "historical" in cell, "HISTORICAL must always be carried as an arm"
    assert cell["historical"]["readout"] == cxm.HISTORICAL.label()


def test_run_crossover_persists_sel_scores(tmp_path, monkeypatch):
    """sel_scores must be persisted per rung, keyed by candidate label, for
    every candidate considered during selection -- the audit trail for
    diagnosing why a given readout won a rung, not just which one won."""
    from eval.label_probe import crossover as cxm

    bank = _write_tiny_bank(tmp_path)
    monkeypatch.setattr(cxm, "N_LADDER", (20, 50))
    monkeypatch.setattr(cxm, "DRAWS_BY_N", {20: 2, 50: 2})
    monkeypatch.setattr(cxm, "panel_for_n", lambda n: ("ridge_strong",))
    cands = [
        geo.Readout("emb", "layer2", "mean", "none"),
        cxm.HISTORICAL,
        geo.Readout("raw", "raw", "-", "none"),
    ]
    monkeypatch.setattr(cxm, "CANDIDATES", {1: cands})

    res = cxm.run_crossover(bank, str(tmp_path), comp_counts=(1,), seed=42)

    sel_scores = res["per_comp"]["1"]["sel_scores"]
    assert set(sel_scores) == {"20", "50"}
    expected_labels = {r.label() for r in cands}
    for n in ("20", "50"):
        assert set(sel_scores[n]) == expected_labels
        for v in sel_scores[n].values():
            assert isinstance(v, float)


def test_run_crossover_new_params_default_to_todays_call_count(tmp_path, monkeypatch):
    """With sel_draws/sel_top_emb/sel_top_raw all None, every candidate is
    still scored during selection, exactly as before these parameters
    existed -- the number of score_readout calls per rung is unchanged."""
    from eval.label_probe import crossover as cxm
    from eval.label_probe import screen as scrm

    bank = _write_tiny_bank(tmp_path)
    monkeypatch.setattr(cxm, "N_LADDER", (20,))
    monkeypatch.setattr(cxm, "DRAWS_BY_N", {20: 2})
    monkeypatch.setattr(cxm, "panel_for_n", lambda n: ("ridge_strong",))
    cands = [
        geo.Readout("emb", "layer2", "mean", "none"),
        geo.Readout("emb", "layer1", "mean", "none"),
        cxm.HISTORICAL,
        geo.Readout("raw", "raw", "-", "none"),
        geo.Readout("raw", "z", "-", "none"),
    ]
    monkeypatch.setattr(cxm, "CANDIDATES", {1: cands})

    calls = {"eval_a": 0, "eval_b": 0}
    orig = scrm.score_readout

    def spy(*a, **kw):
        eval_key = kw.get("eval_key", a[6] if len(a) > 6 else None)
        calls[eval_key] += 1
        return orig(*a, **kw)

    monkeypatch.setattr(scrm, "score_readout", spy)
    cxm.run_crossover(bank, str(tmp_path), comp_counts=(1,), seed=42)

    assert calls["eval_a"] == len(cands), (
        "default behaviour must score every candidate during selection")
