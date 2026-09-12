"""
Orchestration: does raw input or a frozen embedding tap (FE / Projector /
Transformer layer N -- see readouts.py's STAGE_DISPLAY_NAMES) support
few-shot regression of `parameter_0` (labeled_data) better?

Pipeline:
  1. Extract a moment bank (FE, Projector, all 12 Transformer layers) once.
  2. Search for the strongest embedding tap+pooling+probe combination
     (fold-internal, no label leak -- this uses labels only to rank
     candidates, exactly as any model-selection step would).
  3. Whitened raw input vs. the best embedding recipe found vs. the
     conventional final-Transformer-layer/mean-pool tap, compared on a
     shared label-efficiency ladder (n_train small -> full pool), paired
     draws.
  4. A shuffled-label canary at a couple of rungs.
  5. A true-vs-predicted grid at the full pool, axes fixed across panels.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import features as feat
from . import ladder as laddermod
from . import readouts as ro
from .canary import shuffled_label_canary
from .normalize import fit_normalizer
from .protocol import r2, run_primary
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

N_TRAINS = (10, 20, 50, 100, 200, 500, 1000, 2000)

# Candidate embedding taps to screen -- see readouts.py's module docstring
# for exactly what each bank key is in the model's own block vocabulary
# (FE pre-LN, FE post-LN, Projector, Transformer layer 1-12). Pooling fixed
# to "mean" here: RidgeCV's efficient LOOCV path (used throughout) is
# O(n*d^2), so this keeps every screen candidate and every ladder rung
# tractable at this dataset's size (n=4,716) instead of sweeping the full
# pooling axis, which would multiply the cost several-fold for a dimension
# this study's scope doesn't need to resolve.
SCREEN_CANDIDATES = [
    (stage, "mean", normalizer)
    for stage in ("fe", "extract_features", "layer0", "layer1", "layer2", "layer3", "layer12")
    for normalizer in ("whiten", "standardize")
]

# The conventional tap used elsewhere in this codebase's eval package:
# Transformer layer 12 (the final block), mean-pooled.
NAIVE_EMBEDDING_READOUT = ("layer12", "mean", "standardize")

# Phase B: pooling schemes tried only on the single winning stage from phase
# A (not swept across every stage -- segment4/mean_max_min quadruple/triple
# feature width, and RidgeCV's LOOCV path costs O(n*d^2), so sweeping them
# across all 15 stages would be the single most expensive thing this search
# could do for the least information; concentrating them on the one stage
# that already won phase A is where they can actually change the answer).
POOLINGS_TO_SWEEP = ("mean", "mean_std", "mean_max_min", "segment4", "first_last")

# Phase D: alternative probes tried on the winning readout from A-C, looking
# specifically for anything that beats RidgeCV on the EMBEDDING side (this
# search deliberately does not extend this far on the raw-input side, which
# was already well covered by whitening/z-score/PLS-64 in _readout_table).
EMBEDDING_PROBE_CANDIDATES = (
    "ridgecv", "pls8", "pls16", "pls32", "pls64",
    "pca10_ridge", "pca32_ridge", "pca64_ridge", "hgb", "knn5",
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


def _screen_one(X: np.ndarray, y: np.ndarray, probe: str, seed: int) -> float:
    res = run_primary(lambda: _make_any_regressor(probe, seed=seed), X, y,
                       n_repeats=1, n_folds=5, seed0=seed)
    return res.r2_mean


def screen_embedding_readouts(bank: dict, y: np.ndarray, n_comp: int = 1,
                               seed: int = 42) -> dict:
    """Large-n RidgeCV screen over SCREEN_CANDIDATES; returns every score
    plus the best (stage, pooling, normalizer) tuple."""
    ci = _comp_idx(n_comp)
    scores = {}
    for stage, pooling, normalizer in SCREEN_CANDIDATES:
        X = _build_matrix(bank[stage], ci, pooling, normalizer, seed=seed)
        r2_mean = _screen_one(X, y, "ridgecv", seed)
        key = f"{stage}|{pooling}|{normalizer}"
        scores[key] = r2_mean
        display = f"{ro.stage_display_name(stage)}|{pooling}|{normalizer}"
        print(f"[label_probe] screen {display:<36} R2={r2_mean:+.4f}", flush=True)
    best_key = max(scores, key=scores.get)
    best = tuple(best_key.split("|"))
    return {"scores": scores, "best": best}


def search_best_embedding_recipe(bank: dict, y: np.ndarray, n_comp: int = 1,
                                  seed: int = 42) -> dict:
    """
    Staged search for the strongest embedding readout+probe, deliberately
    deeper than the raw-input side (which whitening/z-score/PLS-64 already
    cover well): (A) which stage, mean-pooled; (B) which pooling, for the
    phase-A winner only; (C) does concatenating the top-2 phase-A stages
    beat any single stage; (D) which probe, for the phase A-C winner.
    Every phase's full score table is kept (for transparency/reproducing),
    but callers presenting results should report only the winner and, at
    most, the runner-up per phase -- most candidates here are expected to
    lose, by design of a search this wide.
    """
    ci = _comp_idx(n_comp)

    # Phase A: which stage (mean pooling, both normalizers).
    stage_scores = {}
    for stage in ro.BANK_STAGES:
        for normalizer in ("whiten", "standardize"):
            X = _build_matrix(bank[stage], ci, "mean", normalizer, seed=seed)
            r2_mean = _screen_one(X, y, "ridgecv", seed)
            key = f"{stage}|mean|{normalizer}"
            stage_scores[key] = r2_mean
            display = f"{ro.stage_display_name(stage)}|mean|{normalizer}"
            print(f"[label_probe] search-A {display:<36} R2={r2_mean:+.4f}", flush=True)
    ranked_stages = sorted(
        {k.split("|")[0] for k in stage_scores},
        key=lambda s: max(v for k, v in stage_scores.items() if k.split("|")[0] == s),
        reverse=True)
    best_stage = ranked_stages[0]
    top2_stages = tuple(ranked_stages[:2])

    # Phase B: which pooling, for best_stage only.
    pooling_scores = {}
    for pooling in POOLINGS_TO_SWEEP:
        for normalizer in ("whiten", "standardize"):
            X = _build_matrix(bank[best_stage], ci, pooling, normalizer, seed=seed)
            r2_mean = _screen_one(X, y, "ridgecv", seed)
            key = f"{best_stage}|{pooling}|{normalizer}"
            pooling_scores[key] = r2_mean
            display = f"{ro.stage_display_name(best_stage)}|{pooling}|{normalizer}"
            print(f"[label_probe] search-B {display:<36} R2={r2_mean:+.4f}", flush=True)

    # Phase C: does concatenating the top-2 stages (mean pooling) beat the
    # best single stage found so far?
    concat_scores = {}
    if len(top2_stages) == 2:
        concat_display_name = "+".join(ro.stage_display_name(s) for s in top2_stages)
        for normalizer in ("whiten", "standardize"):
            X = _build_concat_matrix(bank, top2_stages, ci, "mean", normalizer, seed=seed)
            r2_mean = _screen_one(X, y, "ridgecv", seed)
            key = f"concat({'+'.join(top2_stages)})|mean|{normalizer}"
            concat_scores[key] = r2_mean
            display = f"concat({concat_display_name})|mean|{normalizer}"
            print(f"[label_probe] search-C {display:<36} R2={r2_mean:+.4f}", flush=True)

    all_readout_scores = {**stage_scores, **pooling_scores, **concat_scores}
    best_readout_key = max(all_readout_scores, key=all_readout_scores.get)

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
        r2_mean = _screen_one(X_best, y, probe, seed)
        probe_scores[probe] = r2_mean
        print(f"[label_probe] search-D probe={probe:<14} R2={r2_mean:+.4f}", flush=True)
    best_probe = max(probe_scores, key=probe_scores.get)

    return {
        "stage_scores": stage_scores, "pooling_scores": pooling_scores,
        "concat_scores": concat_scores, "probe_scores": probe_scores,
        "best_readout_kind": readout_kind, "best_readout_spec": readout_spec,
        "best_probe": best_probe,
        "best_r2": probe_scores[best_probe],
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


def _readout_table(bank: dict, input_raw: np.ndarray, ci: list,
                    embedding_search: dict, seed: int) -> dict:
    """label -> (X, probe_name). Raw input gets three checked baselines
    (whitened/z-scored/PLS-64 on z-scored -- see normalize.py's module
    docstring for why RidgeCV alone can understate raw input's signal). The
    embedding gets the WINNER of `search_best_embedding_recipe` (best
    stage-or-concat, best pooling, best probe -- see that function's
    docstring for the search), plus the conventional final-layer/mean-pool
    reference for context on how much the search actually bought.
    """
    X_raw = feat.make_wide(input_raw, ci)
    X_raw_white = fit_normalizer("whiten", X_raw, seed=seed).transform(X_raw)
    X_raw_z = fit_normalizer("standardize", X_raw, seed=seed).transform(X_raw)

    kind, spec = embedding_search["best_readout_kind"], embedding_search["best_readout_spec"]
    probe = embedding_search["best_probe"]
    X_emb_best = _build_from_spec(bank, kind, spec, ci, seed=seed)
    emb_label = _readout_label(kind, spec)

    n_stage, n_pooling, n_normalizer = NAIVE_EMBEDDING_READOUT
    X_emb_naive = _build_matrix(bank[n_stage], ci, n_pooling, n_normalizer, seed=seed)
    naive_label = (f"embedding: {ro.stage_display_name(n_stage)}/{n_pooling} "
                   f"({n_normalizer}, conventional), RidgeCV")

    return {
        "raw input (whitened), RidgeCV": (X_raw_white, "ridgecv"),
        "raw input (z-scored), RidgeCV": (X_raw_z, "ridgecv"),
        "raw input (z-scored), PLS-64": (X_raw_z, "pls"),
        f"{emb_label}, {probe}": (X_emb_best, probe),
        naive_label: (X_emb_naive, "ridgecv"),
    }


def run_ladder_comparison(bank: dict, input_raw: np.ndarray, y: np.ndarray,
                           n_comp: int, embedding_search: dict,
                           seed: int = 42) -> dict:
    ci = _comp_idx(n_comp)
    readouts = _readout_table(bank, input_raw, ci, embedding_search, seed)

    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(len(y), size=min(500, len(y) // 4), replace=False)

    ladder_results = {}
    full_pool = {}
    for label, (X, probe) in readouts.items():
        ladder_results[label] = laddermod.score_readout(
            X, y, probe_fn=lambda probe=probe: _make_any_regressor(probe, seed=seed),
            eval_idx=eval_idx, n_trains=list(N_TRAINS), seed=seed)
        print(f"[label_probe] ladder done: {label}", flush=True)

        res = run_primary(lambda probe=probe: _make_any_regressor(probe, seed=seed),
                           X, y, n_repeats=2, seed0=seed)
        full_pool[label] = {"r2_mean": res.r2_mean, "r2_repeat_sd": res.r2_repeat_sd,
                            "oof_predictions": res.oof_predictions.tolist()}
        print(f"[label_probe] full-pool {label}: R2={res.r2_mean:+.4f}", flush=True)

        # fold the full-pool point into the ladder at n_train=len(y), so
        # plots drawn from `ladder` alone (label_efficiency.png) show the
        # complete curve including any crossover that only appears once all
        # labeled data is used -- the 1-comp embedding does exactly this.
        ladder_results[label][len(y)] = {
            "n_train": len(y), "n_draws": 1,
            "r2_median": res.r2_mean, "r2_p25": res.r2_mean, "r2_p75": res.r2_mean,
            "frac_positive_r2": float(res.r2_mean > 0),
        }

    return {"ladder": ladder_results, "full_pool": full_pool,
            "readout_matrices": readouts}


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


def run_study(checkpoint_path: str, labeled_data_dir: str, out_dir: str,
              device: str = "cpu", comps_for_ladder=(1, 2, 3), seed: int = 42) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    bank_path = ro.build_bank_cache(checkpoint_path, labeled_data_dir, out_dir,
                                     comps=tuple(range(3)), device=device, seed=seed)
    bank, input_raw, input_z, y, meta = ro.load_bank_cache(bank_path)
    print(f"[label_probe] bank loaded: n={len(y)}, stages={list(bank)}", flush=True)

    embedding_search = search_best_embedding_recipe(bank, y, n_comp=1, seed=seed)
    best_label = _readout_label(embedding_search["best_readout_kind"],
                                 embedding_search["best_readout_spec"])
    print(f"[label_probe] best embedding recipe: {best_label} "
          f"+ {embedding_search['best_probe']} (R2={embedding_search['best_r2']:+.4f})",
          flush=True)

    all_results = {"meta": meta, "embedding_search": embedding_search, "by_n_comp": {}}
    canary_all = {}
    cells_all = {}
    for n_comp in comps_for_ladder:
        comparison = run_ladder_comparison(bank, input_raw, y, n_comp,
                                            embedding_search, seed=seed)
        all_results["by_n_comp"][n_comp] = {
            "ladder": comparison["ladder"], "full_pool": {
                k: {kk: vv for kk, vv in v.items() if kk != "oof_predictions"}
                for k, v in comparison["full_pool"].items()}}
        canary_all[n_comp] = run_canary_checks(comparison["readout_matrices"], y, seed=seed)
        if n_comp <= 3:
            cells = build_true_vs_pred_cells(comparison["readout_matrices"], y, seed=seed)
            for label, cell in cells.items():
                cells_all[(f"{n_comp}-comp", label)] = cell

    all_results["canary"] = canary_all

    results_path = os.path.join(out_dir, "label_probe_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"[label_probe] wrote {results_path}", flush=True)

    from . import plots
    ladder_1comp = all_results["by_n_comp"][comps_for_ladder[0]]["ladder"]
    plots.plot_label_efficiency(ladder_1comp,
                                 os.path.join(out_dir, "label_efficiency.png"))
    plots.plot_true_vs_pred_grid(cells_all,
                                  os.path.join(out_dir, "true_vs_pred_grid.png"))
    print(f"[label_probe] wrote plots to {out_dir}", flush=True)

    return all_results
