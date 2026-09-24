"""
Orchestration: does raw input or a frozen embedding tap support few-shot
regression of a scalar label better? Backbone-general -- see readouts.py for
what makes that true (hidden_states-only extraction works on any Transformer
encoder).

`run_study` is the one entry point and covers the whole pipeline:
  1. Extract a moment bank once (every hidden_states layer this backbone
     has, plus any FE-specific taps it happens to expose).
  2. SEARCH for the strongest embedding tap+pooling+probe combination
     (fold-internal, no label leak) -- writes label_probe_results.json,
     depth_profile.png, recipe_search.png.
  3. Full-pool-only diagnostics: a shuffled-label canary and a
     true-vs-predicted grid, for raw input and the winning embedding recipe
     -- also in label_probe_results.json, plus true_vs_pred_grid.png.
  4. The honest per-n_train PANEL (see panel.py for why this is a separate
     step from #2: the best recipe is n_train-dependent, so a single fixed
     recipe held across every label budget is not a fair comparison) --
     writes recipe_panel.json, crossover_panel.png, probe_comparison.png.

Both JSONs carry `meta.backbone` (the model's class name, auto-derived) so
runs against different backbones can be told apart later -- see compare.py
to line several runs up side by side.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import features as feat
from . import readouts as ro
from .canary import shuffled_label_canary
from .normalize import fit_normalizer
from .protocol import run_primary
from .regressors import make_fewshot_regressor, make_regressor


def _make_any_regressor(name: str, seed: int = 42):
    """Dispatches to the large-n panel (`make_regressor`) first, falling
    back to the few-shot panel (`make_fewshot_regressor` -- covers
    'plsK'/'pcaK_ridge'/'knnK' names) so both panels' probes are usable
    anywhere in this module without duplicating them."""
    try:
        return make_regressor(name, seed=seed)
    except ValueError:
        return make_fewshot_regressor(name, seed=seed)

# Default label-efficiency ladder for the per-rung panel (see panel.py). The
# full pool (len(y)) is appended at call time -- it isn't known until the
# bank is loaded, since it depends on how many labeled rows exist.
PANEL_N_TRAINS = (10, 20, 50, 100, 200, 500, 1000, 2000)


def _final_layer_stage(bank: dict) -> str:
    """The final Transformer block's hidden-states key, whatever this
    backbone's depth is -- NEVER hardcode a layer count; a different
    Transformer has a different number of blocks."""
    layer_stages = [s for s in bank if s.startswith("layer") and s[5:].isdigit()]
    return max(layer_stages, key=lambda s: int(s[5:]))


def _ranked_stages(embedding_search: dict) -> list:
    """Every block from phase A, ranked by the better of its two normalizers
    -- the same ranking search_best_embedding_recipe uses to pick its
    top-2 stages for phase C."""
    sc = embedding_search["stage_scores"]
    return sorted(
        {k.split("|")[0] for k in sc},
        key=lambda s: max(_r2_of(v) for k, v in sc.items() if k.split("|")[0] == s),
        reverse=True)

# Phase B: pooling schemes tried only on the single winning stage from phase
# A (not swept across every stage -- segment4/mean_max_min quadruple/triple
# feature width, and RidgeCV's LOOCV path costs O(n*d^2), so sweeping them
# across all 15 stages would be the single most expensive thing this search
# could do for the least information; concentrating them on the one stage
# that already won phase A is where they can actually change the answer).
POOLINGS_TO_SWEEP = ("mean", "mean_std", "mean_max_min", "segment4", "first_last")

# Phase D: alternative probes tried on the winning readout from A-C, looking
# for anything that beats RidgeCV on the embedding side. No PLS candidates:
# it is a supervised preprocessing step (fits on labels, not just features),
# and every recipe this module searches over is deliberately
# unsupervised-preprocessing + a probe -- see panel.py's RAW_PANEL comment
# for the full reasoning.
EMBEDDING_PROBE_CANDIDATES = (
    "ridgecv", "pca10_ridge", "pca32_ridge", "pca64_ridge", "hgb", "knn5",
)


def _comp_idx(n_comp: int) -> list:
    return [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]


def _build_matrix(bank_stage: np.ndarray, comp_idx: list, pooling: str,
                   normalizer: str, seed: int = 42):
    X = ro.build_readout(bank_stage, comp_idx, pooling)
    norm = fit_normalizer(normalizer, X, seed=seed)
    return norm.transform(X)


def _build_concat_matrix(bank: dict, stages: tuple, comp_idx: list, pooling: str,
                          normalizer: str, seed: int = 42):
    """Concatenates several stages' readouts, each normalized on its OWN
    scale first (concatenating raw then normalizing jointly would let
    whichever stage happens to have larger raw magnitude dominate the
    combined feature's variance/whitening basis)."""
    blocks = [_build_matrix(bank[s], comp_idx, pooling, normalizer, seed=seed)
              for s in stages]
    return np.concatenate(blocks, axis=1)


# Repeats for every search-phase score. A single 5-fold pass gives a bare
# point estimate with no error bar, which is not enough to say one recipe
# beats another; 3 repeated shuffled splits give a usable spread at 3x the
# cost, and the bootstrap SD on top costs nothing (it resamples predictions
# that have already been computed).
SCREEN_REPEATS = 3


def _screen_one(X: np.ndarray, y: np.ndarray, probe: str, seed: int,
                 n_repeats: int = SCREEN_REPEATS) -> dict:
    """R² with BOTH uncertainties, never a bare number:
      repeat_sd    -- split-assignment noise, for "does A beat B on this data"
      bootstrap_sd -- resampling spectra, for "does this generalize"
    They answer different questions and must not be conflated."""
    res = run_primary(lambda: _make_any_regressor(probe, seed=seed), X, y,
                       n_repeats=n_repeats, n_folds=5, seed0=seed)
    return {"r2": res.r2_mean,
            "repeat_sd": res.r2_repeat_sd,
            "bootstrap_sd": res.r2_bootstrap_sd,
            "n_repeats": n_repeats}


def _r2_of(score) -> float:
    """Search tables hold {r2, repeat_sd, ...} dicts; older runs held a bare
    float. Read either."""
    return score["r2"] if isinstance(score, dict) else float(score)


def search_best_embedding_recipe(bank: dict, y: np.ndarray, n_comp: int = 1,
                                  seed: int = 42) -> dict:
    """
    Staged search for the strongest embedding readout+probe, deliberately
    deeper than the raw-input side (which whitening/z-score already cover
    well): (A) which stage, mean-pooled; (B) which pooling, for the
    phase-A winner only; (C) does concatenating the top-2 phase-A stages
    beat any single stage; (D) which probe, for the phase A-C winner.
    Every phase's full score table is kept (for transparency/reproducing),
    but callers presenting results should report only the winner and, at
    most, the runner-up per phase -- most candidates here are expected to
    lose, by design of a search this wide.
    """
    ci = _comp_idx(n_comp)

    # Phase A: which stage (mean pooling, both normalizers). Stages come from
    # whatever this particular bank actually has -- backbone-dependent (a
    # different Transformer has a different layer count, and may lack the
    # FE-specific taps entirely), never a hardcoded list.
    stage_scores = {}
    for stage in sorted(bank):
        for normalizer in ("whiten", "standardize"):
            X = _build_matrix(bank[stage], ci, "mean", normalizer, seed=seed)
            sc = _screen_one(X, y, "ridgecv", seed)
            key = f"{stage}|mean|{normalizer}"
            stage_scores[key] = sc
            display = f"{ro.stage_display_name(stage)}|mean|{normalizer}"
            print(f"[label_probe] search-A {display:<36} "
                  f"R2={sc['r2']:+.4f} ±{sc['repeat_sd']:.4f}", flush=True)
    ranked_stages = sorted(
        {k.split("|")[0] for k in stage_scores},
        key=lambda s: max(_r2_of(v) for k, v in stage_scores.items()
                          if k.split("|")[0] == s),
        reverse=True)
    best_stage = ranked_stages[0]
    top2_stages = tuple(ranked_stages[:2])

    # Phase B: which pooling, for best_stage only.
    pooling_scores = {}
    for pooling in POOLINGS_TO_SWEEP:
        for normalizer in ("whiten", "standardize"):
            X = _build_matrix(bank[best_stage], ci, pooling, normalizer, seed=seed)
            sc = _screen_one(X, y, "ridgecv", seed)
            key = f"{best_stage}|{pooling}|{normalizer}"
            pooling_scores[key] = sc
            display = f"{ro.stage_display_name(best_stage)}|{pooling}|{normalizer}"
            print(f"[label_probe] search-B {display:<36} "
                  f"R2={sc['r2']:+.4f} ±{sc['repeat_sd']:.4f}", flush=True)

    # Phase C: does concatenating the top-2 stages (mean pooling) beat the
    # best single stage found so far?
    concat_scores = {}
    if len(top2_stages) == 2:
        concat_display_name = "+".join(ro.stage_display_name(s) for s in top2_stages)
        for normalizer in ("whiten", "standardize"):
            X = _build_concat_matrix(bank, top2_stages, ci, "mean", normalizer, seed=seed)
            sc = _screen_one(X, y, "ridgecv", seed)
            key = f"concat({'+'.join(top2_stages)})|mean|{normalizer}"
            concat_scores[key] = sc
            display = f"concat({concat_display_name})|mean|{normalizer}"
            print(f"[label_probe] search-C {display:<36} "
                  f"R2={sc['r2']:+.4f} ±{sc['repeat_sd']:.4f}", flush=True)

    all_readout_scores = {**stage_scores, **pooling_scores, **concat_scores}
    best_readout_key = max(all_readout_scores, key=lambda k: _r2_of(all_readout_scores[k]))

    if best_readout_key.startswith("concat("):
        readout_kind = "concat"
        readout_spec = (top2_stages, "mean", best_readout_key.split("|")[-1])
        X_best = _build_concat_matrix(bank, top2_stages, ci, "mean",
                                       readout_spec[2], seed=seed)
    else:
        readout_kind = "single"
        stage, pooling, normalizer = best_readout_key.split("|")
        readout_spec = (stage, pooling, normalizer)
        X_best = _build_matrix(bank[stage], ci, pooling, normalizer, seed=seed)

    # Phase D: which probe, for the winning readout.
    probe_scores = {}
    for probe in EMBEDDING_PROBE_CANDIDATES:
        sc = _screen_one(X_best, y, probe, seed)
        probe_scores[probe] = sc
        print(f"[label_probe] search-D probe={probe:<14} "
              f"R2={sc['r2']:+.4f} ±{sc['repeat_sd']:.4f}", flush=True)
    best_probe = max(probe_scores, key=lambda k: _r2_of(probe_scores[k]))

    return {
        "stage_scores": stage_scores, "pooling_scores": pooling_scores,
        "concat_scores": concat_scores, "probe_scores": probe_scores,
        "best_readout_kind": readout_kind, "best_readout_spec": readout_spec,
        "best_probe": best_probe,
        "best_r2": _r2_of(probe_scores[best_probe]),
        "best_score": probe_scores[best_probe],
        "n_comp_searched": n_comp,
        "n_samples": int(len(y)),
    }


def _build_from_spec(bank: dict, kind: str, spec: tuple, ci: list, seed: int):
    """spec is either (stage, pooling, normalizer) for kind='single' or
    (stages_tuple, pooling, normalizer) for kind='concat'."""
    if kind == "concat":
        stages, pooling, normalizer = spec
        return _build_concat_matrix(bank, tuple(stages), ci, pooling, normalizer, seed=seed)
    stage, pooling, normalizer = spec
    return _build_matrix(bank[stage], ci, pooling, normalizer, seed=seed)


def _readout_label(kind: str, spec: tuple) -> str:
    if kind == "concat":
        stages, pooling, normalizer = spec
        names = "+".join(ro.stage_display_name(s) for s in stages)
        return f"embedding: concat({names})/{pooling} ({normalizer})"
    stage, pooling, normalizer = spec
    return f"embedding: {ro.stage_display_name(stage)}/{pooling} ({normalizer})"


def _full_pool_readouts(bank: dict, input_raw: np.ndarray, ci: list,
                         embedding_search: dict, seed: int) -> dict:
    """label -> (X, probe_name), for the full-pool-only diagnostics (canary,
    true-vs-predicted, probe grid). Three embedding columns, each a plain
    mean-pooled single-layer readout -- never the search's pooling-squeezed
    recipe, so a block is never conflated with a pooling choice here: the
    winning block, the runner-up, and the conventional final layer, all
    whitened (the adopted normalizer) with RidgeCV. NOT used for any
    per-n_train comparison -- see panel.py's module docstring for why a
    fixed recipe is only fair at a single point, not across a ladder.
    """
    X_raw = feat.make_wide(input_raw, ci)
    X_raw_white = fit_normalizer("whiten", X_raw, seed=seed).transform(X_raw)
    X_raw_z = fit_normalizer("standardize", X_raw, seed=seed).transform(X_raw)

    ranked = _ranked_stages(embedding_search)
    stages = list(dict.fromkeys([ranked[0], ranked[1], _final_layer_stage(bank)]))

    out = {
        "raw input (whitened), RidgeCV": (X_raw_white, "ridgecv"),
        "raw input (z-scored), RidgeCV": (X_raw_z, "ridgecv"),
    }
    for stage in stages:
        X = _build_matrix(bank[stage], ci, "mean", "whiten", seed=seed)
        out[f"embedding: {ro.stage_display_name(stage)}/mean, RidgeCV"] = (X, "ridgecv")
    return out


def run_full_pool_comparison(bank: dict, input_raw: np.ndarray, y: np.ndarray,
                              n_comp: int, embedding_search: dict,
                              seed: int = 42) -> dict:
    ci = _comp_idx(n_comp)
    readouts = _full_pool_readouts(bank, input_raw, ci, embedding_search, seed)

    full_pool = {}
    for label, (X, probe) in readouts.items():
        res = run_primary(lambda probe=probe: _make_any_regressor(probe, seed=seed),
                           X, y, n_repeats=2, seed0=seed)
        full_pool[label] = {"r2_mean": res.r2_mean,
                            "r2_repeat_sd": res.r2_repeat_sd,
                            "r2_bootstrap_sd": res.r2_bootstrap_sd,
                            "n_repeats": 2, "n_samples": int(len(y)),
                            "oof_predictions": res.oof_predictions.tolist()}
        print(f"[label_probe] full-pool {label}: R2={res.r2_mean:+.4f} "
              f"±{res.r2_bootstrap_sd:.4f} (boot)", flush=True)

    return {"full_pool": full_pool, "readout_matrices": readouts}


# Normalizers/probes for the probe-choice grid. Every arm x normalizer here
# is tried with BOTH probes -- unlike the search phases above (which fix
# probe=ridgecv while comparing blocks/pooling) this grid exists specifically
# to answer "does OLS behave differently from RidgeCV", so it must never
# leave a normalizer covered by only one of them.
PROBE_GRID_NORMALIZERS = ("none", "standardize", "whiten", "whiten8", "whiten32", "whiten128")
PROBE_GRID_PROBES = ("ridgecv", "ols")


def run_probe_grid(bank: dict, input_raw: np.ndarray, y: np.ndarray, n_comp: int,
                    embedding_search: dict, seed: int = 42) -> dict:
    """
    Every normalizer x {RidgeCV, OLS}, full pool, on raw input and on three
    plain mean-pooled single-layer embedding readouts -- the winning block,
    the runner-up, and the conventional final layer (same three as
    `_full_pool_readouts`) -- never the search's pooling-squeezed recipe, so
    probe behavior is never confounded with pooling choice. "none" is
    raw-input-only (see the baseline section this feeds: it exists to show
    the naive failure mode, not as a real embedding candidate).
    """
    ci = _comp_idx(n_comp)
    X_raw = feat.make_wide(input_raw, ci)

    ranked = _ranked_stages(embedding_search)
    stages = list(dict.fromkeys([ranked[0], ranked[1], _final_layer_stage(bank)]))

    arms = {"raw input": X_raw}
    for stage in stages:
        label = f"embedding: {ro.stage_display_name(stage)}/mean"
        arms[label] = ro.build_readout(bank[stage], ci, "mean")

    grid = {}
    for arm_label, X_arm in arms.items():
        for norm in PROBE_GRID_NORMALIZERS:
            if norm == "none" and arm_label != "raw input":
                continue
            X = fit_normalizer(norm, X_arm, seed=seed).transform(X_arm)
            for probe in PROBE_GRID_PROBES:
                res = run_primary(lambda probe=probe: _make_any_regressor(probe, seed=seed),
                                   X, y, n_repeats=2, seed0=seed)
                grid[f"{arm_label} | {norm} + {probe}"] = {
                    "arm": arm_label, "normalizer": norm, "probe": probe,
                    "r2_mean": res.r2_mean, "r2_bootstrap_sd": res.r2_bootstrap_sd,
                }
                print(f"[label_probe] probe-grid {arm_label:<55} {norm:<12} {probe:<8} "
                      f"R2={res.r2_mean:+.4f}", flush=True)
    return grid


def run_canary_checks(readouts: dict, y: np.ndarray, seed: int = 42) -> dict:
    """Shuffled-label check on a couple of rungs, for the readouts actually
    used in the report."""
    out = {}
    for label, (X, probe) in readouts.items():
        def fit_predict_oof(Xc, yc, probe=probe):
            from sklearn.model_selection import KFold
            cv = KFold(5, shuffle=True, random_state=seed)
            pred = np.zeros(len(yc))
            for tr, te in cv.split(Xc):
                m = _make_any_regressor(probe, seed=seed)
                m.fit(Xc[tr], yc[tr])
                pred[te] = np.asarray(m.predict(Xc[te])).ravel()
            return pred

        out[label] = shuffled_label_canary(fit_predict_oof, X, y, seed=seed)
        print(f"[label_probe] canary {label}: {out[label]}", flush=True)
    return out


def build_true_vs_pred_cells(readouts: dict, y: np.ndarray, seed: int = 42) -> dict:
    from sklearn.model_selection import KFold
    cells = {}
    for label, (X, probe) in readouts.items():
        cv = KFold(5, shuffle=True, random_state=seed)
        pred = np.zeros(len(y))
        for tr, te in cv.split(X):
            m = _make_any_regressor(probe, seed=seed)
            m.fit(X[tr], y[tr])
            pred[te] = np.asarray(m.predict(X[te])).ravel()
        cells[label] = {"y_true": y, "y_pred": pred}
    return cells


def _write_figures(all_results: dict, out_dir: str, cells_all: dict = None) -> None:
    """Search-derived figures only (depth profile, recipe search, probe
    choice, true-vs-predicted) -- everything drawable from
    `label_probe_results.json` alone. The per-n_train honest crossover lives
    in panel.py/panel_plots.py, over `recipe_panel.json`, and is written
    separately by `write_panel_figures`."""
    from . import plots

    by_n_comp = all_results["by_n_comp"]
    first_comp = sorted(by_n_comp, key=lambda k: int(k))[0]
    n_samples = (all_results.get("meta") or {}).get("n")

    search = all_results.get("embedding_search", {})
    if search.get("stage_scores"):
        raw_ref = None
        for label, v in by_n_comp[first_comp]["full_pool"].items():
            if "raw input (whitened)" in label:
                raw_ref = v["r2_mean"]
        searched_comp = search.get("n_comp_searched", int(first_comp))
        plots.plot_depth_profile(search["stage_scores"],
                                  os.path.join(out_dir, "depth_profile.png"),
                                  raw_reference=raw_ref,
                                  display_name=ro.stage_display_name,
                                  n_comp=searched_comp, n_samples=n_samples,
                                  n_repeats=SCREEN_REPEATS)
        plots.plot_search_bars(search, os.path.join(out_dir, "recipe_search.png"),
                                n_comp=searched_comp, n_samples=n_samples)

    if all_results.get("probe_grid"):
        plots.plot_probe_comparison(all_results["probe_grid"],
                                     os.path.join(out_dir, "probe_comparison.png"),
                                     n_comp=search.get("n_comp_searched", 1),
                                     n_samples=n_samples)

    # The true-vs-predicted grid needs per-sample predictions, which are too
    # bulky to keep in the results JSON -- it can only be drawn on a full run.
    if cells_all:
        plots.plot_true_vs_pred_grid(cells_all,
                                      os.path.join(out_dir, "true_vs_pred_grid.png"))


def replot_from_results(results_path: str, out_dir: str = None) -> str:
    """Redraw the search-derived figures from a finished run's JSON. Cheap
    (seconds); doesn't cover the panel figures -- see
    `panel.write_panel_figures` for those. cells_all (needed for the
    true-vs-predicted grid) is not persisted in the JSON, so that one figure
    is only redrawn if it already exists; this never regenerates it."""
    out_dir = out_dir or os.path.dirname(results_path)
    with open(results_path) as f:
        all_results = json.load(f)
    _write_figures(all_results, out_dir)
    print(f"[label_probe] redrew figures in {out_dir}", flush=True)
    return out_dir


def run_study(checkpoint_path: str, labeled_data_dir: str, out_dir: str,
              device: str = "cpu", comps_for_ladder=(1, 2, 3), seed: int = 42,
              model=None) -> dict:
    """The whole pipeline, one call, for any backbone: extract -> search for
    the best embedding recipe -> a full-pool probe-choice grid (every
    normalizer x {RidgeCV, OLS}, on raw input and on two named embedding
    arms) -> full-pool diagnostics (canary, true-vs-predicted) -> the honest
    per-n_train panel (panel.py) across every component count. Writes
    label_probe_results.json + recipe_panel.json and every figure; both
    JSONs carry `meta.backbone` (auto-derived from the model class) so runs
    from different backbones can be told apart and compared later (see
    compare.py).

    `model`, when given, is used instead of loading `checkpoint_path` from
    disk -- lets a caller that already holds a loaded model (e.g. eval.runner,
    any checkpoint_mode including `hf`) reuse it. `checkpoint_path` is still
    recorded for provenance; pass "" or a descriptive label when there is no
    backing file.

    Extraction pulls exactly as many raw components as the largest requested
    n_comp needs (comps_for_ladder=(1,) only ever touches component 0) --
    NOT a fixed 3, since `load_labeled_data` keeps only spectra that have
    EVERY requested component, and requiring components a dataset doesn't
    consistently have silently shrinks N (or empties it to zero on a
    dataset with no multi-component spectra at all)."""
    from . import panel as pnl

    os.makedirs(out_dir, exist_ok=True)
    extract_comps = tuple(range(max(comps_for_ladder)))
    bank_path = ro.build_bank_cache(checkpoint_path, labeled_data_dir, out_dir,
                                     comps=extract_comps, device=device, seed=seed,
                                     model=model)
    bank, input_raw, input_z, y, meta = ro.load_bank_cache(bank_path)
    print(f"[label_probe] bank loaded: n={len(y)}, "
          f"backbone={meta.get('backbone')}, stages={list(bank)}", flush=True)

    embedding_search = search_best_embedding_recipe(bank, y, n_comp=1, seed=seed)
    best_label = _readout_label(embedding_search["best_readout_kind"],
                                 embedding_search["best_readout_spec"])
    print(f"[label_probe] best embedding recipe: {best_label} "
          f"+ {embedding_search['best_probe']} (R2={embedding_search['best_r2']:+.4f})",
          flush=True)

    probe_grid = run_probe_grid(bank, input_raw, y, 1, embedding_search, seed=seed)

    # Same three plain layers as run_probe_grid / _full_pool_readouts, named
    # once here and reused everywhere below -- winning block, runner-up,
    # conventional final layer.
    ranked = _ranked_stages(embedding_search)
    layer_keys = list(dict.fromkeys([ranked[0], ranked[1], _final_layer_stage(bank)]))
    layer_stages = {ro.stage_display_name(s): s for s in layer_keys}

    all_results = {"meta": meta, "embedding_search": embedding_search,
                   "probe_grid": probe_grid, "by_n_comp": {}}
    canary_all = {}
    cells_all = {}
    panel_results = {"meta": meta, "raw_panel": pnl.RAW_PANEL,
                     "layer_recipe": list(pnl.LAYER_RECIPE),
                     "layer_stages": layer_stages, "by_n_comp": {}}
    # Rungs at or above the real pool size are dropped, not just left in --
    # ladder.draw_indices clips each draw to min(n_train, pool) anyway, so an
    # unfiltered rung would silently draw the SAME actual sample and report
    # it under a bigger, misleading nominal n_train (e.g. a rung literally
    # labeled "n_train=2,000" on a 25-row dataset). The real pool size is
    # always kept as the final rung.
    n_trains = sorted(set(n for n in PANEL_N_TRAINS if n < len(y)) | {len(y)})

    for n_comp in comps_for_ladder:
        fp = run_full_pool_comparison(bank, input_raw, y, n_comp, embedding_search, seed=seed)
        all_results["by_n_comp"][n_comp] = {"full_pool": {
            k: {kk: vv for kk, vv in v.items() if kk != "oof_predictions"}
            for k, v in fp["full_pool"].items()}}
        canary_all[n_comp] = run_canary_checks(fp["readout_matrices"], y, seed=seed)
        if n_comp <= 3:
            cells = build_true_vs_pred_cells(fp["readout_matrices"], y, seed=seed)
            for label, cell in cells.items():
                cells_all[(f"{n_comp}-comp", label)] = cell

        rung = pnl.run_panel(bank, input_raw, y, n_comp, n_trains, seed=seed)
        rung["layer_curves"] = pnl.run_layer_curves(
            bank, input_raw, y, n_comp, layer_stages, n_trains, seed=seed)
        panel_results["by_n_comp"][str(n_comp)] = rung

    all_results["canary"] = canary_all

    results_path = os.path.join(out_dir, "label_probe_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"[label_probe] wrote {results_path}", flush=True)

    panel_path = os.path.join(out_dir, "recipe_panel.json")
    with open(panel_path, "w") as f:
        json.dump(panel_results, f, indent=2, default=str)
    print(f"[label_probe] wrote {panel_path}", flush=True)

    _write_figures(all_results, out_dir, cells_all=cells_all)
    pnl.write_panel_figures(panel_path, out_dir)
    print(f"[label_probe] wrote plots to {out_dir}", flush=True)

    return {"results": all_results, "panel": panel_results}
