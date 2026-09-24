"""
Per-rung label-efficiency curves: raw input via an honest recipe search,
named embedding layers via one fixed recipe.

Raw input (`run_panel`): score every recipe in RAW_PANEL at every label
budget, then let the SELECTED best recipe vary by budget -- fixing one
recipe across the whole ladder guarantees an unfair comparison at one end of
it, because the best normalizer is itself n_train-dependent. Measured on raw
input, 1 component:

    n_train      whitened (full rank)     z-scored
    50                          0.096        0.126
    200                         0.457        0.609
    4,716                       0.825        0.762

Full-rank whitening equalises ~235 near-zero-variance directions. With 4,716
labels a probe can work out which of them carry signal; with 50 it cannot, and
overfits amplified noise.

Embedding layers (`run_layer_curves`): one fixed recipe (LAYER_RECIPE) per
named, plain mean-pooled layer -- no per-layer recipe search, and never the
recipe search's pooling-squeezed readout (that squeeze is a one-off finding
reported once, in the search itself, not a generalizing representation to
carry across every figure).

`run_panel` returns two things:
  * `panel`     -- every raw recipe at every rung (report it all, hide nothing)
  * `selected`  -- the best raw recipe per rung, chosen on a SELECTION eval
                   split and scored on a disjoint REPORT split, so the
                   per-rung choice cannot inflate the number it reports.
"""
from __future__ import annotations

import numpy as np

from . import features as feat
from . import ladder as laddermod
from .normalize import fit_normalizer

# Recipes for the raw-input arm. The whitenK rungs are the point: they trace
# how much whitening is too much as the label budget shrinks.
RAW_PANEL = [
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

# Single fixed recipe for the per-layer embedding curves below: the
# normalizer/probe adopted everywhere else in the report (whitened,
# RidgeCV). Deliberately NOT the recipe search's pooling-squeezed readout
# (four-segment pooling etc.) -- that squeeze is a one-off finding reported
# once, in the recipe search itself, not a generalizing representation to
# repeat across every figure. Every embedding curve here is a plain
# mean-pooled single layer, so "the embedding" is always a specific,
# nameable block.
LAYER_RECIPE = ("whiten", "ridgecv")


def _probe(name, seed=42):
    from .study import _make_any_regressor
    return _make_any_regressor(name, seed=seed)


def build_raw_arms(input_raw, ci, seed=42):
    """recipe_label -> feature matrix, for every raw-input recipe."""
    X_raw = feat.make_wide(input_raw, ci)
    return {f"{norm} + {probe}": (fit_normalizer(norm, X_raw, seed=seed).transform(X_raw), probe)
            for norm, probe in RAW_PANEL}


def _eval_split(y, seed=42, n_eval=500):
    """Two disjoint held-out sets: one to CHOOSE a per-rung recipe, one to
    REPORT it. Shared by run_panel and run_layer_curves so every curve in a
    run is scored against the identical held-out rows."""
    rng = np.random.default_rng(seed)
    held = rng.choice(len(y), size=min(2 * n_eval, len(y) // 2), replace=False)
    return held[:len(held) // 2], held[len(held) // 2:]


def run_panel(bank, input_raw, y, n_comp, n_trains, seed=42, n_eval=500):
    """Honest per-rung selection, raw input only -- score every recipe in
    RAW_PANEL at every rung, choose the best per rung on a selection split,
    report it on a disjoint split. See run_layer_curves for the embedding
    side, which uses one fixed recipe per named layer instead: the best
    normalizer for raw input is itself n_train-dependent (why this function
    exists at all), but this report does not carry that same search onto
    every embedding layer -- see this module's docstring."""
    ci = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
    arms = build_raw_arms(input_raw, ci, seed=seed)
    eval_select, eval_report = _eval_split(y, seed=seed, n_eval=n_eval)

    # A small-n rung for the progress print below, picked from whatever
    # rungs THIS run actually has (n_trains is trimmed to the real pool size
    # by the caller, so a fixed 50 isn't always present -- e.g. it's absent
    # entirely on a dataset with under 50 labels).
    small_n = min((n for n in n_trains if n >= 50), default=min(n_trains))

    panel = {}
    for label, (X, probe) in arms.items():
        res_sel = laddermod.score_readout(
            X, y, probe_fn=lambda p=probe: _probe(p, seed), eval_idx=eval_select,
            n_trains=list(n_trains), seed=seed)
        res_rep = laddermod.score_readout(
            X, y, probe_fn=lambda p=probe: _probe(p, seed), eval_idx=eval_report,
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
        print(f"[panel] {n_comp}-comp  raw  {label:<26} "
              f"n={small_n}:{res_rep[small_n]['r2_median']:+.3f}  "
              f"full:{res_rep[max(n_trains)]['r2_median']:+.3f}", flush=True)

    selected = {}
    for n in n_trains:
        key = max(panel, key=lambda k: panel[k]["select"][str(n)])
        rep = panel[key]["report"][str(n)]
        selected[str(n)] = {"recipe": panel[key]["recipe"],
                             "r2_median": rep["r2_median"],
                             "r2_p25": rep["r2_p25"], "r2_p75": rep["r2_p75"],
                             "r2_draws": rep["r2_draws"]}
    return {"panel": panel, "selected": selected,
            "n_eval_select": len(eval_select), "n_eval_report": len(eval_report)}


def run_layer_curves(bank, input_raw, y, n_comp, layer_stages, n_trains, seed=42,
                      n_eval=500):
    """One curve per named layer in `layer_stages` ({display_name: stage_key}),
    plain mean-pooled, LAYER_RECIPE only -- no per-layer recipe search. Scored
    on the same report split `run_panel` uses, so every curve in a run sits on
    the same held-out rows and can be paired draw-by-draw with the raw curve.
    """
    from . import readouts as ro

    ci = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
    _, eval_report = _eval_split(y, seed=seed, n_eval=n_eval)
    norm, probe = LAYER_RECIPE
    small_n = min((n for n in n_trains if n >= 50), default=min(n_trains))

    curves = {}
    for name, stage in layer_stages.items():
        X_raw_layer = ro.build_readout(bank[stage], ci, "mean")
        X = fit_normalizer(norm, X_raw_layer, seed=seed).transform(X_raw_layer)
        res = laddermod.score_readout(
            X, y, probe_fn=lambda p=probe: _probe(p, seed), eval_idx=eval_report,
            n_trains=list(n_trains), seed=seed)
        curves[name] = {str(k): {"r2_median": v["r2_median"],
                                  "r2_p25": v["r2_p25"], "r2_p75": v["r2_p75"],
                                  "n_draws": v["n_draws"], "r2_draws": v["r2_draws"]}
                        for k, v in res.items()}
        print(f"[panel] {n_comp}-comp  layer  {name:<24} "
              f"n={small_n}:{res[small_n]['r2_median']:+.3f}  "
              f"full:{res[max(n_trains)]['r2_median']:+.3f}", flush=True)
    return curves


def _raw_full_pool_reference(results_path: str) -> dict:
    """{n_comp: (r2_mean, r2_bootstrap_sd, normalizer_label)} for the best of
    (whitened, z-scored) raw input at the full pool, per n_comp -- read from
    the sibling label_probe_results.json, if one was written alongside this
    recipe_panel.json. Returns {} if it isn't there (e.g. a bare
    recipe_panel.json redrawn on its own with no results file beside it)."""
    import json
    import os

    if not os.path.isfile(results_path):
        return {}
    with open(results_path) as f:
        results = json.load(f)
    out = {}
    for n_comp, by_comp in results.get("by_n_comp", {}).items():
        best = None
        for label, v in by_comp.get("full_pool", {}).items():
            if label.startswith("raw input") and (best is None or v["r2_mean"] > best[0]):
                norm = "z-scored" if "z-scored" in label else "whitened"
                best = (v["r2_mean"], v.get("r2_bootstrap_sd"), norm)
        if best:
            out[n_comp] = best
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
    raw_full_pool = _raw_full_pool_reference(
        os.path.join(os.path.dirname(recipe_panel_path), "label_probe_results.json"))
    pp.plot_crossover_panel(d["by_n_comp"], os.path.join(out_dir, "crossover_panel.png"),
                            n_eval_report=n_eval_report, n_pool=n_pool,
                            raw_full_pool=raw_full_pool)
    print(f"[panel] redrew figures in {out_dir}", flush=True)
    return out_dir
