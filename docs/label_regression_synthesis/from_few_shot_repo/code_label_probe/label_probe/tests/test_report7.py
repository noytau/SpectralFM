import os

import numpy as np

from eval.label_probe import crossover as cx
from eval.label_probe import report7


def _fake_results(cross_at=500):
    rng = np.random.default_rng(0)
    per_n, selected = {}, {}
    for n in cx.N_LADDER:
        d = -0.30 if n < cross_at else +0.10
        raw = rng.normal(0.30, 0.02, 40)
        emb = raw + d
        hist = raw - 0.35
        mk = lambda arr, ro_, pr: {
            "r2_median": float(np.median(arr)), "r2_p25": 0.1, "r2_p75": 0.4,
            "frac_positive_r2": 0.9, "n_failed_draws": 0,
            "r2_per_draw": list(arr), "readout": ro_, "probe": pr}
        per_n[str(n)] = {"emb": mk(emb, "emb|layer2|segment4|whiten", "ridgecv"),
                         "raw": mk(raw, "raw|raw|-|whiten", "ridgecv"),
                         "historical": mk(hist, "emb|layer12|mean|none", "ridgecv")}
        selected[str(n)] = {"emb": "emb|layer2|segment4|whiten",
                            "raw": "raw|raw|-|whiten",
                            "historical": "emb|layer12|mean|none"}
    crossings = {"1": cx.estimate_crossover(
        {n: {"emb_per_draw": per_n[str(n)]["emb"]["r2_per_draw"],
             "raw_per_draw": per_n[str(n)]["raw"]["r2_per_draw"]}
         for n in cx.N_LADDER})}
    return {"meta": {"checkpoint": "feb25"}, "ladder": list(cx.N_LADDER),
            "draws_by_n": {str(n): cx.DRAWS_BY_N[n] for n in cx.N_LADDER},
            "panels": {str(n): list(cx.panel_for_n(n)) for n in cx.N_LADDER},
            "split_sizes": {"eval_a": 500, "eval_b": 500, "pool": 3716},
            "per_comp": {"1": {"per_n": per_n, "selected": selected}},
            "crossings": crossings}


def test_report_states_the_crossing_and_names_the_eval_set(tmp_path):
    p = report7.write_step7_report(_fake_results(), str(tmp_path))
    t = open(p).read()
    assert "# Step 7" in t
    assert "eval_b" in t, "must name the set the numbers came from"
    assert "500" in t, "the crossing rung must appear"
    assert "layer12" in t, "the historical reference arm must be reported"


def test_report_discloses_the_overlapping_draws_caveat(tmp_path):
    t = open(report7.write_step7_report(_fake_results(), str(tmp_path))).read().lower()
    assert "overlap" in t, (
        "draws at n>=1000 come from a 3716 pool and understate variance; "
        "the report must say so")


def test_report_does_not_claim_a_crossing_it_did_not_measure(tmp_path):
    """Crossing-specific language must appear only when a crossing was found,
    and must be absent when raw leads at every rung -- and vice versa for the
    no-crossing language. A vacuous check for a phrase that appears nowhere in
    the template would pass trivially and catch nothing; this anchors on the
    actual wording each branch emits."""
    crossed_dir = tmp_path / "crossed"
    not_crossed_dir = tmp_path / "not_crossed"
    t_crossed = open(report7.write_step7_report(
        _fake_results(cross_at=500), str(crossed_dir))).read().lower()
    t_not_crossed = open(report7.write_step7_report(
        _fake_results(cross_at=10**9), str(not_crossed_dir))).read().lower()

    assert "crossing at n =" in t_crossed
    assert "no crossing" in t_not_crossed or "did not cross" in t_not_crossed

    # Each branch's language must not leak into the other.
    assert "crossing at n =" not in t_not_crossed
    assert "no crossing on this ladder" not in t_crossed
    assert "overtakes raw" not in t_crossed
    assert "overtakes raw" not in t_not_crossed


def test_missing_rung_renders_as_explicit_marker(tmp_path):
    """A rung absent from crossings['rungs'] must render as a visible marker
    (e.g. '--') in the derived columns, not as 0.00/no -- values that are
    indistinguishable from a real computed near-zero win probability."""
    r = _fake_results(cross_at=500)
    # Drop one rung entirely from the crossing's per-rung detail, simulating
    # the int/str key hazard or any other reason a rung's detail goes missing.
    del r["crossings"]["1"]["rungs"][100]
    t = open(report7.write_step7_report(r, str(tmp_path))).read()
    lines = [ln for ln in t.splitlines() if ln.startswith("| 100 |")]
    assert len(lines) == 1, "the row for the missing rung must still be present"
    row = lines[0]
    cols = [c.strip() for c in row.strip("|").split("|")]
    # columns: n_train, draws, emb R2, raw R2, historical R2, dR2, boot P(D>0), won, readout, probe
    delta_col, boot_col, won_col = cols[5], cols[6], cols[7]
    assert delta_col == "—"
    assert boot_col == "—"
    assert won_col == "—"
    assert boot_col != "0.00"
    assert won_col != "no"


def test_plot_writes_a_real_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    p = report7.plot_crossover(_fake_results(), str(tmp_path / "c.png"), n_comp=1)
    assert os.path.getsize(p) > 1000
