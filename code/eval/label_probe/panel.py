"""
Per-rung label-efficiency curves, raw input and named embedding layers
treated identically: every arm scores every recipe in RECIPE_PANEL at every
label budget, and the SELECTED best recipe is allowed to vary by budget.

Fixing one recipe across the ladder guarantees an unfair comparison at one
end of it, because the best normalizer is itself n_train-dependent -- and
which normalizer wins also differs between raw input and an embedding, and
between datasets. Measured on raw input, 1 component, labeled_data:

    n_train      whitened (full rank)     z-scored
    50                          0.096        0.126
    200                         0.457        0.609
    4,716                       0.825        0.762

Full-rank whitening equalises ~235 near-zero-variance directions. With 4,716
labels a probe can work out which of them carry signal; with 50 it cannot, and
overfits amplified noise. Giving only one side of the comparison this
per-budget choice would hand that side a structural advantage, so both sides
get it.

Each embedding arm is a plain mean-pooled single layer -- never the recipe
search's pooling-squeezed readout (that squeeze is a one-off finding
reported once, in the search itself, not a generalizing representation to
carry across every figure).

Per arm, `_select_per_rung` produces:
  * `panel`     -- every recipe at every rung (report it all, hide nothing)
  * `selected`  -- the best recipe per rung, chosen on a SELECTION eval
                   split and scored on a disjoint REPORT split, so the
                   per-rung choice cannot inflate the number it reports.
Every arm is scored on the same two splits with the same training draws, so
any two arms can be paired draw-by-draw at a given rung.
"""
from __future__ import annotations

import numpy as np

from . import features as feat
from . import ladder as laddermod
from .normalize import fit_normalizer

# Recipes scored for every arm, raw input and embedding layers alike. The
# whitenK rungs are the point: they trace how much whitening is too much as
# the label budget shrinks.
RECIPE_PANEL = [
    ("standardize", "ridgecv"),
    ("none", "ridgecv"),
    ("whiten", "ridgecv"),
    ("whiten8", "ridgecv"),
    ("whiten32", "ridgecv"),
    ("whiten128", "ridgecv"),
    # Unregularised OLS, at three normalizers -- traces the float32/
    # ill-conditioning trap documented in "Setting the baseline" (cond ~1e9,
    # OLS ~0.40 vs the true ~0.82 ceiling). "none" and "standardize" are both
    # ill-conditioned (per-feature scaling doesn't decorrelate); "whiten"
    # removes the collinearity a rotation can fix, so OLS should recover there.
    ("none", "ols"),
    ("standardize", "ols"),
    ("whiten", "ols"),
    # No PLS anywhere in this panel: it is a SUPERVISED preprocessing step
    # (it rotates using the labels, not just the features), a leak risk that
    # needs careful fold-internal discipline to use safely at all (see the
    # leak demonstration in "Setting the baseline"). Whitening is unsupervised
    # by construction and carries no such risk -- every recipe here is
    # unsupervised-preprocessing + a probe that sees labels only at fit time.
]


def _probe(name, seed=42):
    from .study import _make_any_regressor
    return _make_any_regressor(name, seed=seed)


def _recipes(X, seed=42):
    """recipe_label -> (normalized feature matrix, probe), for one arm's
    un-normalized features."""
    return {f"{norm} + {probe}": (fit_normalizer(norm, X, seed=seed).transform(X), probe)
            for norm, probe in RECIPE_PANEL}


def _eval_split(y, seed=42, n_eval=500):
    """Two disjoint held-out sets: one to CHOOSE a per-rung recipe, one to
    REPORT it. Shared by run_panel and run_layer_curves so every curve in a
    run is scored against the identical held-out rows."""
    rng = np.random.default_rng(seed)
    held = rng.choice(len(y), size=min(2 * n_eval, len(y) // 2), replace=False)
    return held[:len(held) // 2], held[len(held) // 2:]


def _select_per_rung(X, y, n_trains, eval_select, eval_report, seed, tag):
    """Score every RECIPE_PANEL recipe on one arm's un-normalized features X
    at every rung, on both eval splits; pick each rung's best on the
    selection split and report it on the report split.
    Returns (panel {recipe: {select, report}}, selected {rung: result})."""
    # A small-n rung for the progress print, picked from whatever rungs THIS
    # run actually has (n_trains is trimmed to the real pool size by the
    # caller, so a fixed 50 is absent on a dataset with under 50 labels).
    small_n = min((n for n in n_trains if n >= 50), default=min(n_trains))

    panel = {}
    for label, (Xn, probe) in _recipes(X, seed=seed).items():
        res_sel = laddermod.score_readout(
            Xn, y, probe_fn=lambda p=probe: _probe(p, seed), eval_idx=eval_select,
            n_trains=list(n_trains), seed=seed)
        res_rep = laddermod.score_readout(
            Xn, y, probe_fn=lambda p=probe: _probe(p, seed), eval_idx=eval_report,
            n_trains=list(n_trains), seed=seed)
        panel[label] = {
            "recipe": label,
            "select": {str(k): v["r2_median"] for k, v in res_sel.items()},
            "report": {str(k): {"r2_median": v["r2_median"],
                                 "r2_p25": v["r2_p25"], "r2_p75": v["r2_p75"],
                                 "n_draws": v["n_draws"],
                                 "r2_draws": v["r2_draws"]}
                        for k, v in res_rep.items()},
        }
        print(f"[panel] {tag:<34} {label:<22} "
              f"n={small_n}:{res_rep[small_n]['r2_median']:+.3f}  "
              f"full:{res_rep[max(n_trains)]['r2_median']:+.3f}", flush=True)

    selected = {}
    for n in n_trains:
        key = max(panel, key=lambda k: panel[k]["select"][str(n)])
        rep = panel[key]["report"][str(n)]
        selected[str(n)] = {"recipe": panel[key]["recipe"],
                             "r2_median": rep["r2_median"],
                             "r2_p25": rep["r2_p25"], "r2_p75": rep["r2_p75"],
                             "n_draws": rep["n_draws"],
                             "r2_draws": rep["r2_draws"]}
    return panel, selected


def run_panel(bank, input_raw, y, n_comp, n_trains, seed=42, n_eval=500):
    """Raw-input arm: per-rung recipe selection over RECIPE_PANEL."""
    ci = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
    eval_select, eval_report = _eval_split(y, seed=seed, n_eval=n_eval)
    panel, selected = _select_per_rung(
        feat.make_wide(input_raw, ci), y, n_trains, eval_select, eval_report, seed,
        tag=f"{n_comp}-comp raw")
    return {"panel": panel, "selected": selected,
            "n_eval_select": len(eval_select), "n_eval_report": len(eval_report)}


def run_layer_curves(bank, input_raw, y, n_comp, layer_stages, n_trains, seed=42,
                      n_eval=500):
    """One curve per named layer in `layer_stages` ({display_name: stage_key}),
    plain mean-pooled, with the SAME per-rung recipe selection over
    RECIPE_PANEL, on the same two splits and draws, as run_panel's raw arm.
    Returns (curves, panels): curves = {name: {rung: selected result incl.
    its "recipe"}}; panels = {name: {recipe: {rung: report-split median}}},
    every recipe kept so nothing chosen is hidden."""
    from . import readouts as ro

    ci = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
    eval_select, eval_report = _eval_split(y, seed=seed, n_eval=n_eval)

    curves, panels = {}, {}
    for name, stage in layer_stages.items():
        panel, selected = _select_per_rung(
            ro.build_readout(bank[stage], ci, "mean"), y, n_trains,
            eval_select, eval_report, seed, tag=f"{n_comp}-comp {name}")
        curves[name] = selected
        panels[name] = {recipe: {k: v["r2_median"] for k, v in p["report"].items()}
                        for recipe, p in panel.items()}
    return curves, panels


def _full_pool_references(results_path: str) -> dict:
    """{n_comp: {"raw": ref, "embedding": ref}}, ref = (r2_mean,
    r2_bootstrap_sd, normalizer_label, arm_name): the best full-pool CV score
    of raw input, and of any named embedding layer, each over both
    normalizers -- read from the sibling label_probe_results.json, if one was
    written alongside this recipe_panel.json. Returns {} if it isn't there
    (e.g. a bare recipe_panel.json redrawn on its own)."""
    import json
    import os

    if not os.path.isfile(results_path):
        return {}
    with open(results_path) as f:
        results = json.load(f)
    out = {}
    for n_comp, by_comp in results.get("by_n_comp", {}).items():
        refs = {}
        for label, v in by_comp.get("full_pool", {}).items():
            # "<arm> (<normalizer>), RidgeCV"; runs written before embeddings
            # got both normalizers have "<arm>, RidgeCV" (whitened) instead.
            arm, sep, rest = label.rpartition(" (")
            if not sep:
                arm, rest = label.rsplit(",", 1)[0], "whitened)"
            side = "raw" if arm == "raw input" else "embedding"
            name = arm.replace("embedding: ", "").replace("/mean", "")
            if side not in refs or v["r2_mean"] > refs[side][0]:
                refs[side] = (v["r2_mean"], v.get("r2_bootstrap_sd"), rest.split(")")[0], name)
        if refs:
            out[n_comp] = refs
    return out


def write_panel_figures(recipe_panel_path: str, out_dir: str = None) -> str:
    """Redraw the panel-derived figure (crossover_panel.png) from a finished
    recipe_panel.json. Cheap: seconds, no recompute -- mirrors
    study.replot_from_results for the main run. probe_comparison.png is a
    full-pool figure now, drawn by study._write_figures instead -- see that
    function's docstring."""
    import json
    import os

    from . import panel_plots as pp

    with open(recipe_panel_path) as f:
        d = json.load(f)
    out_dir = out_dir or os.path.dirname(recipe_panel_path)
    n_pool = d["meta"].get("n")
    n_eval_report = next(iter(d["by_n_comp"].values())).get("n_eval_report")
    refs = _full_pool_references(
        os.path.join(os.path.dirname(recipe_panel_path), "label_probe_results.json"))
    pp.plot_crossover_panel(d["by_n_comp"], os.path.join(out_dir, "crossover_panel.png"),
                            n_eval_report=n_eval_report, n_pool=n_pool,
                            full_pool_refs=refs)
    print(f"[panel] redrew figures in {out_dir}", flush=True)
    return out_dir
