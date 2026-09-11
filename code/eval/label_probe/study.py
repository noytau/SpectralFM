"""
Orchestration: does raw input or a frozen embedding readout support
few-shot regression of `parameter_0` (labeled_data) better?

Pipeline:
  1. Extract a moment bank (all stages/layers) once.
  2. Screen a small set of embedding readouts at large-n to pick the
     strongest one (fold-internal, no label leak — this uses labels only to
     rank readouts, exactly as any model-selection step would).
  3. Whitened raw input vs. the best embedding readout vs. the embedding's
     naive final-layer/mean-pool convention, compared on a shared
     label-efficiency ladder (n_train small -> full pool), paired draws.
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
from .regressors import make_regressor

N_TRAINS = (10, 20, 50, 100, 200, 500, 1000, 2000)

# Candidate embedding readouts to screen — early transformer layers plus the
# final layer (the "naive" convention used elsewhere in this codebase).
# Pooling fixed to "mean" here: RidgeCV's efficient LOOCV path (used
# throughout) is O(n*d^2), so this keeps every screen candidate and every
# ladder rung tractable at this dataset's size (n=4,716) instead of sweeping
# the full pooling axis, which would multiply the cost several-fold for a
# dimension this study's scope doesn't need to resolve.
SCREEN_CANDIDATES = [
    (f"layer{li}", "mean", normalizer)
    for li in (0, 1, 2, 3, 12)
    for normalizer in ("whiten", "standardize")
]

NAIVE_EMBEDDING_READOUT = ("layer12", "mean", "standardize")


def _comp_idx(n_comp: int) -> list:
    return [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]


def _build_matrix(bank_stage: np.ndarray, comp_idx: list, pooling: str,
                   normalizer: str, seed: int = 42):
    X = ro.build_readout(bank_stage, comp_idx, pooling)
    norm = fit_normalizer(normalizer, X, seed=seed)
    return norm.transform(X)


def screen_embedding_readouts(bank: dict, y: np.ndarray, n_comp: int = 1,
                               seed: int = 42) -> dict:
    """Large-n RidgeCV screen over SCREEN_CANDIDATES; returns every score
    plus the best (stage, pooling, normalizer) tuple."""
    ci = _comp_idx(n_comp)
    scores = {}
    for stage, pooling, normalizer in SCREEN_CANDIDATES:
        X = _build_matrix(bank[stage], ci, pooling, normalizer, seed=seed)
        res = run_primary(lambda: make_regressor("ridgecv", seed=seed), X, y,
                           n_repeats=1, n_folds=5, seed0=seed)
        key = f"{stage}|{pooling}|{normalizer}"
        scores[key] = res.r2_mean
        print(f"[label_probe] screen {key:<28} R2={res.r2_mean:+.4f}", flush=True)
    best_key = max(scores, key=scores.get)
    best = tuple(best_key.split("|"))
    return {"scores": scores, "best": best}


def run_ladder_comparison(bank: dict, input_raw: np.ndarray, y: np.ndarray,
                           n_comp: int, best_embedding_readout: tuple,
                           seed: int = 42) -> dict:
    ci = _comp_idx(n_comp)

    X_raw = feat.make_wide(input_raw, ci)
    X_raw_white = fit_normalizer("whiten", X_raw, seed=seed).transform(X_raw)

    stage, pooling, normalizer = best_embedding_readout
    X_emb_best = _build_matrix(bank[stage], ci, pooling, normalizer, seed=seed)

    n_stage, n_pooling, n_normalizer = NAIVE_EMBEDDING_READOUT
    X_emb_naive = _build_matrix(bank[n_stage], ci, n_pooling, n_normalizer, seed=seed)

    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(len(y), size=min(500, len(y) // 4), replace=False)

    readouts = {
        "raw input (whitened)": X_raw_white,
        f"embedding {stage}/{pooling} (whitened)": X_emb_best,
        "embedding layer12/mean (naive)": X_emb_naive,
    }
    ladder_results = {}
    for label, X in readouts.items():
        ladder_results[label] = laddermod.score_readout(
            X, y, probe_fn=lambda: make_regressor("ridgecv", seed=seed),
            eval_idx=eval_idx, n_trains=list(N_TRAINS), seed=seed)
        print(f"[label_probe] ladder done: {label}", flush=True)

    # full-pool point (5-fold CV, all data)
    full_pool = {}
    for label, X in readouts.items():
        res = run_primary(lambda: make_regressor("ridgecv", seed=seed), X, y,
                           n_repeats=2, seed0=seed)
        full_pool[label] = {"r2_mean": res.r2_mean, "r2_repeat_sd": res.r2_repeat_sd,
                            "oof_predictions": res.oof_predictions.tolist()}
        print(f"[label_probe] full-pool {label}: R2={res.r2_mean:+.4f}", flush=True)

    return {"ladder": ladder_results, "full_pool": full_pool,
            "readout_matrices": readouts}


def run_canary_checks(readouts: dict, y: np.ndarray, seed: int = 42) -> dict:
    """Shuffled-label check on a couple of rungs, for the readouts actually
    used in the report."""
    out = {}
    for label, X in readouts.items():
        def fit_predict_oof(Xc, yc):
            from sklearn.model_selection import KFold
            cv = KFold(5, shuffle=True, random_state=seed)
            pred = np.zeros(len(yc))
            for tr, te in cv.split(Xc):
                m = make_regressor("ridgecv", seed=seed)
                m.fit(Xc[tr], yc[tr])
                pred[te] = np.asarray(m.predict(Xc[te])).ravel()
            return pred

        out[label] = shuffled_label_canary(fit_predict_oof, X, y, seed=seed)
        print(f"[label_probe] canary {label}: {out[label]}", flush=True)
    return out


def build_true_vs_pred_cells(readouts: dict, y: np.ndarray, seed: int = 42) -> dict:
    from sklearn.model_selection import KFold
    cells = {}
    for label, X in readouts.items():
        cv = KFold(5, shuffle=True, random_state=seed)
        pred = np.zeros(len(y))
        for tr, te in cv.split(X):
            m = make_regressor("ridgecv", seed=seed)
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

    screen = screen_embedding_readouts(bank, y, n_comp=1, seed=seed)
    print(f"[label_probe] best embedding readout: {screen['best']}", flush=True)

    all_results = {"meta": meta, "screen": screen, "by_n_comp": {}}
    canary_all = {}
    cells_all = {}
    for n_comp in comps_for_ladder:
        comparison = run_ladder_comparison(bank, input_raw, y, n_comp,
                                            screen["best"], seed=seed)
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
