"""
Per-rung recipe panel: score EVERY recipe at EVERY label budget, then let each
arm pick its own best recipe per budget.

Why this exists. Fixing one recipe per arm across the whole ladder guarantees
an unfair comparison at one end of it, because the best recipe is
n_train-dependent. Measured on raw input, 1 component:

    n_train      whitened (full rank)     z-scored
    50                          0.096        0.126
    200                         0.457        0.609
    4,716                       0.825        0.762

Full-rank whitening equalises ~235 near-zero-variance directions. With 4,716
labels a probe can work out which of them carry signal; with 50 it cannot, and
overfits amplified noise. So "whiten vs standardize" has no single winner, and
a study that picks one and holds it fixed is measuring its own choice as much
as the representation.

Two outputs, from one pass:
  * `panel`     -- every recipe at every rung (report it all, hide nothing)
  * `selected`  -- the best recipe per arm per rung, chosen on a SELECTION
                   eval split and scored on a disjoint REPORT split, so the
                   per-rung choice cannot inflate the number it reports.
"""
from __future__ import annotations

import numpy as np

from . import features as feat
from . import ladder as laddermod
from . import readouts as ro
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
    ("standardize", "pls64"),
]

# Same normalizer ladder on the embedding side, on the block the full-pool
# search selected, so neither arm is handed an advantage the other is denied.
EMB_PANEL = [
    ("standardize", "ridgecv"),
    ("whiten", "ridgecv"),
    ("whiten8", "ridgecv"),
    ("whiten32", "ridgecv"),
    ("whiten128", "ridgecv"),
]


def _probe(name, seed=42):
    from .study import _make_any_regressor
    return _make_any_regressor(name, seed=seed)


def build_arms(bank, input_raw, ci, emb_spec, seed=42):
    """(arm, recipe_label) -> feature matrix, for every recipe in both panels."""
    from .study import _build_from_spec

    X_raw = feat.make_wide(input_raw, ci)
    kind, spec = emb_spec
    X_emb = _build_from_spec(bank, kind, spec, ci, seed=seed)

    arms = {}
    for norm, probe in RAW_PANEL:
        X = fit_normalizer(norm, X_raw, seed=seed).transform(X_raw)
        arms[("raw", f"{norm} + {probe}")] = (X, probe)
    for norm, probe in EMB_PANEL:
        X = fit_normalizer(norm, X_emb, seed=seed).transform(X_emb)
        arms[("embedding", f"{norm} + {probe}")] = (X, probe)
    return arms


def run_panel(bank, input_raw, y, n_comp, emb_spec, n_trains, seed=42,
              n_eval=500):
    ci = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
    arms = build_arms(bank, input_raw, ci, emb_spec, seed=seed)

    # Two disjoint held-out sets: one to CHOOSE the per-rung recipe, one to
    # REPORT it. Selecting and reporting on the same rows would let the choice
    # chase that split's noise and hand back an inflated number.
    rng = np.random.default_rng(seed)
    held = rng.choice(len(y), size=min(2 * n_eval, len(y) // 2), replace=False)
    eval_select, eval_report = held[:len(held) // 2], held[len(held) // 2:]

    panel = {}
    for (arm, label), (X, probe) in arms.items():
        res_sel = laddermod.score_readout(
            X, y, probe_fn=lambda p=probe: _probe(p, seed), eval_idx=eval_select,
            n_trains=list(n_trains), seed=seed)
        res_rep = laddermod.score_readout(
            X, y, probe_fn=lambda p=probe: _probe(p, seed), eval_idx=eval_report,
            n_trains=list(n_trains), seed=seed)
        panel[f"{arm} | {label}"] = {
            "arm": arm, "recipe": label,
            "select": {str(k): v["r2_median"] for k, v in res_sel.items()},
            "report": {str(k): {"r2_median": v["r2_median"],
                                 "r2_p25": v["r2_p25"], "r2_p75": v["r2_p75"],
                                 "n_draws": v["n_draws"],
                                 "r2_draws": v["r2_draws"]}
                        for k, v in res_rep.items()},
        }
        print(f"[panel] {n_comp}-comp  {arm:<9} {label:<26} "
              f"n=50:{res_rep[50]['r2_median']:+.3f}  "
              f"full:{res_rep[max(n_trains)]['r2_median']:+.3f}", flush=True)

    selected = {}
    for arm in ("raw", "embedding"):
        here = {k: v for k, v in panel.items() if v["arm"] == arm}
        per_n = {}
        for n in n_trains:
            key = max(here, key=lambda k: here[k]["select"][str(n)])
            rep = here[key]["report"][str(n)]
            per_n[str(n)] = {"recipe": here[key]["recipe"],
                              "r2_median": rep["r2_median"],
                              "r2_p25": rep["r2_p25"], "r2_p75": rep["r2_p75"],
                              "r2_draws": rep["r2_draws"]}
        selected[arm] = per_n
    return {"panel": panel, "selected": selected,
            "n_eval_select": len(eval_select), "n_eval_report": len(eval_report)}
