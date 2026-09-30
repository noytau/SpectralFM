import numpy as np
import pytest

from eval.label_probe import geometry as geo
from eval.label_probe import readouts as ro
from eval.label_probe import study6


def _write_synthetic_bank(path, n=1060, k=1, d=4, seed=7):
    """
    A tiny, from-scratch .npz in exactly the layout `readouts.load_bank_cache`
    expects, so `run_screen` can be exercised end-to-end without a checkpoint,
    a GPU, or the real 3 GB bank cache.

    n is larger than the trivial "N=60" a unit test would prefer because
    `run_screen` calls `screen.make_split` with its default n_eval_a=500 /
    n_eval_b=500; a smaller corpus would make eval_b and/or the training pool
    degenerate (empty, or too small for the n_train=50 draws). n=1060 leaves
    a pool of 60 -- enough for real (if repetitive) n_train=50 draws -- while
    keeping k and d tiny so every fit is cheap.

    y carries a real planted linear signal (not just noise) so the probes
    inside run_screen have something to find rather than degenerating to the
    dummy/near-zero-R2 case throughout.
    """
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=n)

    payload = {}
    for stage in ro.BANK_STAGES:
        arr = rng.normal(scale=0.1, size=(n, k, 10, d)).astype(np.float32)
        arr[:, 0, 0, 0] += factor       # plant the signal in the "mean" stat
        payload[f"bank__{stage}"] = arr.astype(np.float16)

    input_raw = rng.normal(size=(n, k, 245)).astype(np.float32)
    input_raw[:, 0, 0] += factor
    input_z = input_raw.copy()
    y = (2.0 * factor + 0.1 * rng.normal(size=n)).astype(np.float64)

    meta = {"checkpoint": "synthetic", "comps": (0,), "n": int(n),
            "max_samples": int(n), "seed": seed, "pool_stats": ro.POOL_STATS,
            "stages": ro.BANK_STAGES}
    payload["input_raw"] = input_raw
    payload["input_z"] = input_z
    payload["y"] = y
    payload["_meta"] = np.array([repr(meta)])

    with open(path, "wb") as f:
        np.savez(f, **payload)


def test_run_screen_dedups_wave2_and_writes_output(tmp_path, monkeypatch):
    # Keep this test's wall-clock cost low without touching production code:
    # the SCREEN_PANEL's gp_rbf (hyperparameter-optimized, multiple restarts)
    # and readout_diagnostics's per-readout PCA + 5 RidgeCV fits dominate
    # run_screen's real runtime and are irrelevant to what this test checks
    # (the dedup bookkeeping and the JSON write). `study6` looks these up as
    # module attributes (`geo.SCREEN_PANEL`, `scr.readout_diagnostics`) at
    # call time, so patching the attribute on the already-imported module
    # objects is enough -- no production file is edited.
    monkeypatch.setattr(geo, "SCREEN_PANEL", ("ridge_strong", "knn_cos5"))

    def _fast_diagnostics(X, y, seed=42):
        d = np.asarray(X).shape[1]
        return {"participation_ratio": 1.0, "effective_rank": 1.0,
                "ridgecv_full_r2": 0.0,
                "whitened_topk_r2": {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0},
                "n_features": int(d)}

    import eval.label_probe.screen as scr_module
    monkeypatch.setattr(scr_module, "readout_diagnostics", _fast_diagnostics)

    bank_path = tmp_path / "bank.npz"
    _write_synthetic_bank(bank_path)
    output_dir = tmp_path / "out"

    results = study6.run_screen(str(bank_path), str(output_dir),
                                 comp_counts=(1,), n_draws=2, n_top_stages=1)

    # n_top_stages=1 -> exactly one finalist stage for n_comp=1, so wave 2's
    # grid (that stage x every pooling x every normalizer) recreates exactly
    # `len(geo.NORMALIZERS)` cells wave 1 already scored (the pooling="mean"
    # ones) -- those, and only those, must be skipped rather than recomputed.
    assert results["wave2_skipped"][1] == len(geo.NORMALIZERS)

    n_wave0 = len(study6.RAW_STAGES) * len(geo.NORMALIZERS)
    n_wave1 = len(ro.BANK_STAGES) * len(geo.NORMALIZERS)
    n_wave2_new = len(ro.POOLINGS) * len(geo.NORMALIZERS) - len(geo.NORMALIZERS)
    assert len(results["readouts"]) == n_wave0 + n_wave1 + n_wave2_new

    output_path = output_dir / "step6_screen.json"
    assert output_path.exists()


def test_readout_grid_sizes():
    w1 = study6.iter_readouts(wave=1)
    # wave 1: 15 stages x 4 normalizers, pooling fixed to mean
    assert len(w1) == 15 * 4
    assert all(r.pooling == "mean" for r in w1)
    assert len({r.label() for r in w1}) == len(w1)

    w2 = study6.iter_readouts(wave=2, stages=("layer6", "layer8"))
    # wave 2: 2 stages x 6 poolings x 4 normalizers
    assert len(w2) == 2 * 6 * 4
    assert len({r.label() for r in w2}) == len(w2)


def test_raw_family_readouts_exist():
    raws = study6.iter_readouts(wave=0)
    stages = {r.stage for r in raws}
    assert stages == {"raw", "z", "moments"}
    assert all(r.family == "raw" for r in raws)
    assert len(raws) == 3 * 4        # 3 raw stages x 4 normalizers


def test_materialize_shapes(tiny_bank):
    from eval.label_probe import geometry as geo

    n = tiny_bank.shape[0]
    bank = {"layer0": tiny_bank}
    raw = np.random.default_rng(0).normal(size=(n, 3, 245)).astype(np.float32)
    cache = {}
    r = geo.Readout("emb", "layer0", "mean_std", "standardize")
    X = study6.materialize(r, bank, raw, raw, [0, 1], cache)
    assert X.shape == (n, 2 * 2 * 8)     # 2 comps x 2 stats x 8 channels
    assert np.all(np.isfinite(X))

    r2_ = geo.Readout("raw", "raw", "-", "none")
    X2 = study6.materialize(r2_, bank, raw, raw, [0, 1], cache)
    assert X2.shape == (n, 2 * 245)

    r3 = geo.Readout("raw", "moments", "-", "none")
    X3 = study6.materialize(r3, bank, raw, raw, [0, 1], cache)
    assert X3.shape == (n, 2 * 10)


def test_materialize_memoizes_the_normalizer(tiny_bank):
    from eval.label_probe import geometry as geo

    n = tiny_bank.shape[0]
    raw = np.zeros((n, 3, 245), dtype=np.float32)
    cache = {}
    r = geo.Readout("emb", "layer0", "mean", "whiten")
    study6.materialize(r, {"layer0": tiny_bank}, raw, raw, [0], cache)
    assert len(cache) == 1
    study6.materialize(r, {"layer0": tiny_bank}, raw, raw, [0], cache)
    assert len(cache) == 1, "normalizer must be fitted once, not per call"


def _cell(median, per_draw):
    return {"r2_median": median, "frac_positive_r2": 1.0,
            "r2_per_draw": per_draw}


def test_verdict_win_when_embedding_leads_in_five_cells():
    rng = np.random.default_rng(0)
    cells = {}
    for n_comp in (1, 2, 3):
        for n_train in (20, 50):
            emb = list(rng.normal(0.30, 0.02, 100))
            raw = list(rng.normal(0.10, 0.02, 100))
            if (n_comp, n_train) == (1, 20):        # one losing cell
                emb, raw = raw, emb
            cells[f"{n_comp}|{n_train}"] = {
                "emb": _cell(float(np.median(emb)), emb),
                "raw": _cell(float(np.median(raw)), raw),
            }
    v = study6.verdict({"cells": cells})
    assert v["verdict"] == "WIN"
    assert v["n_cells_won"] == 5


def test_verdict_no_when_raw_leads_everywhere():
    rng = np.random.default_rng(1)
    cells = {}
    for n_comp in (1, 2, 3):
        for n_train in (20, 50):
            emb = list(rng.normal(0.05, 0.02, 100))
            raw = list(rng.normal(0.40, 0.02, 100))
            cells[f"{n_comp}|{n_train}"] = {
                "emb": _cell(float(np.median(emb)), emb),
                "raw": _cell(float(np.median(raw)), raw),
            }
    v = study6.verdict({"cells": cells})
    assert v["verdict"] == "NO"
    assert v["n_cells_won"] == 0


def test_verdict_partial_between_the_thresholds():
    rng = np.random.default_rng(2)
    cells = {}
    for i, (n_comp, n_train) in enumerate(
            [(c, n) for c in (1, 2, 3) for n in (20, 50)]):
        win = i < 2
        emb = list(rng.normal(0.30 if win else 0.05, 0.02, 100))
        raw = list(rng.normal(0.10 if win else 0.40, 0.02, 100))
        cells[f"{n_comp}|{n_train}"] = {
            "emb": _cell(float(np.median(emb)), emb),
            "raw": _cell(float(np.median(raw)), raw),
        }
    assert study6.verdict({"cells": cells})["verdict"] == "PARTIAL"


def test_verdict_requires_reliability_not_just_a_higher_median():
    """An embedding that edges raw on the median but works less than 80% of
    the time must NOT count as a win -- that is the client-facing guard."""
    rng = np.random.default_rng(3)
    cells = {}
    for n_comp in (1, 2, 3):
        for n_train in (20, 50):
            emb = list(rng.normal(0.30, 0.02, 100))
            raw = list(rng.normal(0.10, 0.02, 100))
            cells[f"{n_comp}|{n_train}"] = {
                "emb": {"r2_median": float(np.median(emb)),
                        "frac_positive_r2": 0.5, "r2_per_draw": emb},
                "raw": _cell(float(np.median(raw)), raw),
            }
    assert study6.verdict({"cells": cells})["verdict"] == "NO"


def test_verdict_records_n_pairs_and_n_dropped_when_draws_fail():
    """T12 finding 3: a cell that lost some of its paired draws to a failed
    probe (-inf, excluded from the delta) must record how many pairs
    survived and how many were dropped -- a lossy cell must not look
    identical to a clean one."""
    n = 100
    n_bad = 30
    emb = [0.3] * (n - n_bad) + [float("-inf")] * n_bad
    raw = [0.1] * n
    cells = {"1|20": {
        "emb": _cell(0.3, emb),
        "raw": _cell(0.1, raw),
    }}
    v = study6.verdict({"cells": cells})
    cell = v["cells"]["1|20"]
    assert cell["n_pairs"] == n - n_bad
    assert cell["n_dropped"] == n_bad


def test_verdict_n_dropped_zero_when_no_draws_fail():
    emb = [0.3] * 100
    raw = [0.1] * 100
    cells = {"1|20": {"emb": _cell(0.3, emb), "raw": _cell(0.1, raw)}}
    v = study6.verdict({"cells": cells})
    cell = v["cells"]["1|20"]
    assert cell["n_pairs"] == 100
    assert cell["n_dropped"] == 0


def test_confirm_does_not_report_an_all_failed_probe_as_the_winner(
        tmp_path, monkeypatch):
    """T12 finding 2: `max()` compares via `>`, and comparisons against NaN
    are always False, so a naive `max(..., key=lambda c: c['r2_median'])`
    can get permanently stuck on the FIRST candidate if it happens to be a
    probe whose every draw failed (r2_median NaN) -- silently reporting a
    failed probe as the cell's best instead of the genuinely-scoring one.
    Order the fake panel so the failing probe comes first."""
    monkeypatch.setattr(geo, "CONFIRM_PANEL", ("failing_probe", "good_probe"))

    def _fake_score_readout(X, y, split, panel, n_trains, draws, eval_key,
                            seed=42):
        out = {}
        for p in panel:
            out[p] = {}
            for nt in n_trains:
                if p == "failing_probe":
                    out[p][nt] = {
                        "n_train": nt, "n_draws": 100, "r2_median": float("nan"),
                        "r2_p25": float("nan"), "r2_p75": float("nan"),
                        "r2_p10": float("nan"), "r2_p90": float("nan"),
                        "mae_median": float("nan"), "spearman_median": float("nan"),
                        "frac_positive_r2": 0.0, "n_failed_draws": 100,
                        "r2_per_draw": [float("-inf")] * 100,
                    }
                else:
                    out[p][nt] = {
                        "n_train": nt, "n_draws": 100, "r2_median": 0.4,
                        "r2_p25": 0.3, "r2_p75": 0.5, "r2_p10": 0.2, "r2_p90": 0.6,
                        "mae_median": 0.1, "spearman_median": 0.5,
                        "frac_positive_r2": 1.0, "n_failed_draws": 0,
                        "r2_per_draw": [0.4] * 100,
                    }
        return out

    import eval.label_probe.screen as scr_module
    monkeypatch.setattr(scr_module, "score_readout", _fake_score_readout)

    bank_path = tmp_path / "bank.npz"
    _write_synthetic_bank(bank_path)
    screen_results = {"readouts": {
        "1|emb|layer0|mean|none": {
            "n_comp": 1, "family": "emb", "stage": "layer0", "pooling": "mean",
            "normalizer": "none", "best_r2": 0.4},
        "1|raw|raw|-|none": {
            "n_comp": 1, "family": "raw", "stage": "raw", "pooling": "-",
            "normalizer": "none", "best_r2": 0.3},
    }}

    out = study6.run_confirm(str(bank_path), screen_results,
                             str(tmp_path / "out"), n_draws=2, top_emb=1,
                             top_raw=1, use_tabpfn=False)

    cell = out["cells"]["1|20"]
    assert cell["emb"]["probe"] == "good_probe", (
        "a failed (NaN-median) probe must never be reported as the winner "
        "just because it sorts first")
    assert np.isfinite(cell["emb"]["r2_median"])
