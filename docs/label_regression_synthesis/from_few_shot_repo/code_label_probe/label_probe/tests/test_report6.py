import json
import os

import numpy as np
import pytest

from eval.label_probe import report6


def _fake_screen():
    rng = np.random.default_rng(0)
    ro_, diag = {}, {}
    for i, stage in enumerate(["layer0", "layer6", "layer12", "fe"]):
        for nrm in ("none", "whiten"):
            key = f"1|emb|{stage}|mean|{nrm}"
            # planted: higher topk10 goes with higher best_r2, so the
            # correlation test has a signal to find
            topk = 0.05 * i + (0.2 if nrm == "whiten" else 0.0)
            ro_[key] = {"n_comp": 1, "family": "emb", "stage": stage,
                        "pooling": "mean", "normalizer": nrm,
                        "best_probe": "ridge_strong", "best_n_train": 20,
                        "best_r2": topk + 0.01 * rng.normal(), "scores": {}}
            diag[key] = {"participation_ratio": 10.0 + i,
                         "effective_rank": 12.0 + i,
                         "ridgecv_full_r2": 0.5,
                         "whitened_topk_r2": {5: topk / 2, 10: topk,
                                              20: topk, 50: topk},
                         "n_features": 768}
    return {"readouts": ro_, "diagnostics": diag, "n_draws": 30,
            "finalist_stages": {1: ["layer12"]},
            "split_sizes": {"eval_a": 500, "eval_b": 500, "pool": 3716}}


def _fake_confirm():
    rng = np.random.default_rng(1)
    cells = {}
    for n_comp in (1, 2, 3):
        for nt in (20, 50):
            emb = list(rng.normal(0.2, 0.05, 100))
            raw = list(rng.normal(0.3, 0.05, 100))
            cells[f"{n_comp}|{nt}"] = {
                "emb": {"r2_median": float(np.median(emb)), "r2_p25": 0.1,
                        "r2_p75": 0.3, "frac_positive_r2": 0.9,
                        "r2_per_draw": emb, "readout": "emb|layer12|mean|whiten",
                        "probe": "tabpfn_pca50", "n_failed_draws": 0},
                "raw": {"r2_median": float(np.median(raw)), "r2_p25": 0.2,
                        "r2_p75": 0.4, "frac_positive_r2": 1.0,
                        "r2_per_draw": raw, "readout": "raw|raw|-|none",
                        "probe": "tabpfn_pca50", "n_failed_draws": 0},
            }
    return {"cells": cells, "n_draws": 100, "panel": ["tabpfn_pca50"],
            "finalists": {1: ["1|emb|layer12|mean|whiten"]}, "detail": {}}


def test_diagnostic_correlation_finds_the_planted_relationship():
    out = report6.diagnostic_correlation(_fake_screen())
    assert "whitened_topk_r2_10" in out
    assert out["whitened_topk_r2_10"]["rho"] > 0.8
    assert np.isfinite(out["whitened_topk_r2_10"]["p"])


def test_report_states_the_verdict_and_the_losing_cells(tmp_path):
    from eval.label_probe import study6

    screen, confirm = _fake_screen(), _fake_confirm()
    v = study6.verdict(confirm)
    assert v["verdict"] == "NO"          # raw leads in the fake data
    path = report6.write_step6_report(screen, confirm, v, str(tmp_path))
    text = open(path).read()
    assert "# Step 6" in text
    assert "NO" in text
    assert "eval_b" in text              # must name which set produced the numbers
    assert "raw|raw|-|none" in text      # must name the winning raw readout
    assert "tabpfn_pca50" in text        # must name the probe


def test_report_does_not_claim_a_win_it_did_not_measure(tmp_path):
    """Guard against a template that hardcodes celebratory language."""
    from eval.label_probe import study6

    confirm = _fake_confirm()
    v = study6.verdict(confirm)
    text = open(report6.write_step6_report(_fake_screen(), confirm, v,
                                           str(tmp_path))).read().lower()
    assert "beats raw input" not in text


def test_diagnostic_correlation_handles_json_roundtripped_string_keys():
    """Regression test: json.load() turns whitened_topk_r2's int sub-keys
    into strings ('5', '10', '20', '50'). _diag_value used to look them up
    as int only, so diagnostic_correlation's `except KeyError: continue`
    silently dropped all four whitened_topk_r2_* rows when diagnostics came
    from a loaded step6_screen.json rather than an in-memory dict."""
    rng = np.random.default_rng(2)
    ro_, diag = {}, {}
    for i, stage in enumerate(["layer0", "layer6", "layer12", "fe"]):
        key = f"1|emb|{stage}|mean|whiten"
        topk = 0.1 * i
        ro_[key] = {"n_comp": 1, "family": "emb", "stage": stage,
                    "pooling": "mean", "normalizer": "whiten",
                    "best_probe": "ridge_strong", "best_n_train": 20,
                    "best_r2": topk + 0.01 * rng.normal(), "scores": {}}
        # simulate the JSON round-trip: sub-keys are strings, not ints
        diag[key] = {"participation_ratio": 10.0 + i,
                     "effective_rank": 12.0 + i,
                     "ridgecv_full_r2": 0.5,
                     "whitened_topk_r2": {"5": topk / 2, "10": topk,
                                          "20": topk, "50": topk},
                     "n_features": 768}
    screen = {"readouts": ro_, "diagnostics": diag, "n_draws": 30,
              "finalist_stages": {1: ["layer12"]},
              "split_sizes": {"eval_a": 500, "eval_b": 500, "pool": 3716}}

    out = report6.diagnostic_correlation(screen)

    for n in (5, 10, 20, 50):
        key = f"whitened_topk_r2_{n}"
        assert key in out, f"{key} missing -- string sub-keys silently dropped"
        assert out[key]["rho"] > 0


def test_diagnostic_correlation_reports_missing_key_as_unavailable_not_dropped(tmp_path):
    """A diagnostic entirely absent from `diagnostics` (KeyError path) must
    still show up in the returned mapping, marked unavailable -- not vanish
    with only a print statement as its trace. This is the same defect class
    (T12 finding 1) as the silent row-drop that once inverted a reader's
    conclusion."""
    from eval.label_probe import study6

    screen = _fake_screen()
    for d in screen["diagnostics"].values():
        del d["ridgecv_full_r2"]

    out = report6.diagnostic_correlation(screen)

    assert "ridgecv_full_r2" in out, "missing diagnostic must not vanish silently"
    assert out["ridgecv_full_r2"]["rho"] is None
    assert out["ridgecv_full_r2"]["n"] == 0
    assert "unavailable" in out["ridgecv_full_r2"]["status"]

    confirm = _fake_confirm()
    path = report6.write_step6_report(screen, confirm, study6.verdict(confirm),
                                      str(tmp_path))
    text = open(path).read()
    assert "ridgecv_full_r2" in text
    assert "unavailable" in text.lower()


def test_diagnostic_correlation_reports_too_few_points_as_unavailable_not_dropped(tmp_path):
    """The `ok.sum() < 3` path must also surface as a visible unavailable
    row, not a table that just loses the diagnostic with nothing printed."""
    from eval.label_probe import study6

    screen = _fake_screen()
    # Poison all but two readouts' ridgecv_full_r2 so only 2 finite pairs
    # remain -- below the n>=3 threshold.
    keys = list(screen["diagnostics"])
    for k in keys[2:]:
        screen["diagnostics"][k]["ridgecv_full_r2"] = float("nan")

    out = report6.diagnostic_correlation(screen)

    assert "ridgecv_full_r2" in out, "low-n diagnostic must not vanish silently"
    assert out["ridgecv_full_r2"]["rho"] is None
    assert out["ridgecv_full_r2"]["n"] < 3
    assert "unavailable" in out["ridgecv_full_r2"]["status"]

    confirm = _fake_confirm()
    path = report6.write_step6_report(screen, confirm, study6.verdict(confirm),
                                      str(tmp_path))
    text = open(path).read()
    assert "unavailable" in text.lower()


def test_plots_write_files(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from eval.label_probe import study6

    screen, confirm = _fake_screen(), _fake_confirm()
    p1 = report6.plot_screen(screen, str(tmp_path / "screen.png"), n_comp=1)
    p2 = report6.plot_confirm(confirm, study6.verdict(confirm),
                              str(tmp_path / "confirm.png"))
    assert os.path.getsize(p1) > 1000
    assert os.path.getsize(p2) > 1000
