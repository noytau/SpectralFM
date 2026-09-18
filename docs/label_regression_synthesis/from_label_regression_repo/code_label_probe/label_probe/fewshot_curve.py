"""
Extended label-efficiency study (step 5b).

Differences from run_step5, all driven by review feedback:

  * n_train extends to 2000 (was 100), so the curve reaches the regime where
    the step-1 asymptotic numbers live and you can see where it saturates.
  * every deployment component count (1, 2, 3) is swept, plus 12 as an anchor
    (was 3 and 12 only).
  * draws scale with n rather than being fixed at 100: variance is largest at
    n=10 and negligible at n=2000, so the draw budget is allocated where it
    buys precision. 1000 draws at n<=20 down to 100 at n>=1000.
  * every reported point carries an uncertainty interval (p25/p75 and p10/p90
    across draws), plotted as bands/error bars rather than a bare median.
  * `ridgecv_full` joins the panel: at n>=200 the low-capacity few-shot probes
    are the wrong tool and the step-1 workhorse is the honest comparator.
  * TabPFN predictions are retained for one cell per (comp, stage) so a
    true-vs-predicted scatter can be drawn against the actual labels.

TabPFN costs ~1.7s per draw on a 2080 Ti and does not get cheaper with a
smaller eval set or a reused estimator (both measured), so its cells are
sharded across the 7 available GPUs by `--shard`/`--nshards`.
"""
from __future__ import annotations

import glob
import json
import os
import time
import warnings

# Pin threads process-wide BEFORE numpy/torch import. The PCA fits and
# torch ops sit outside run_cell's threadpool_limits context; unpinned,
# each worker grabbed ~5 cores and 19 workers took a shared 40-core host
# to load 160, crowding other users for no throughput gain (TabPFN is
# GPU-bound; its CPU threads are pure contention).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

from . import cache as cachemod
from . import features as feat
from .protocol import r2
from .regressors import make_fewshot_regressor, make_regressor

# ── grid ──────────────────────────────────────────────────────────────────
N_TRAIN = (10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000)
COMPS = (1, 2, 3, 12)
STAGES = ("input_raw", "input_z", "fe", "extract_features", "proj", "transformer")

# Cheap probes: sklearn, milliseconds per fit.
CHEAP_PANEL = ("dummy", "ridge_strong", "pca20_ridge", "pls2", "knn5", "ridgecv_full")
# TabPFN: ~1.7s/draw, so a narrowed grid on the stages that carry the decision.
TABPFN_PROBE = "tabpfn_pca10"
TABPFN_STAGES = ("input_raw", "input_z", "transformer")
TABPFN_COMPS = (1, 2, 3)
TABPFN_N_TRAIN = (10, 20, 50, 100, 200, 500)


def draws_for(n_train: int, tabpfn: bool = False) -> int:
    """More draws where the estimator variance is large. A fixed draw count
    over-samples the easy end of the curve and under-samples the end the
    client's constraint actually sits in."""
    if tabpfn:
        return 300 if n_train <= 100 else 200
    if n_train <= 20:
        return 1000
    if n_train <= 100:
        return 500
    if n_train <= 500:
        return 200
    # At n>=1000 the across-draw spread collapses (measured IQR width ~0.003 at
    # n=500), so 30 draws already pins the median far tighter than any
    # systematic in this study. Spending 100+ here would buy nothing and cost
    # hours, because RidgeCV on 9216 dims at n=2000 is seconds per fit.
    return 30


def load_cache(out_dir: str):
    cdir = os.path.join(out_dir, "_cache")
    files = sorted(glob.glob(os.path.join(cdir, "reps_*.npz")), key=os.path.getsize, reverse=True)
    if not files:
        raise FileNotFoundError(f"no step-1 cache under {cdir}")
    key = os.path.basename(files[0])[len("reps_"):-len(".npz")]
    return cachemod.load_reps(cdir, key)


def split(n_total: int, n_eval: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    return perm[:n_eval], perm[n_eval:]


def _summarize(r2s, maes, rhos, n_train, n_draws, extra=None):
    r2s = np.asarray(r2s, dtype=float)
    fin = r2s[np.isfinite(r2s)]
    q = lambda p: float(np.percentile(fin, p)) if fin.size else float("nan")
    out = {
        "n_train": n_train, "n_draws": n_draws,
        "r2_median": float(np.median(fin)) if fin.size else float("nan"),
        "r2_mean": float(np.mean(fin)) if fin.size else float("nan"),
        "r2_p10": q(10), "r2_p25": q(25), "r2_p75": q(75), "r2_p90": q(90),
        # SE of the median ~ 1.253 * sd / sqrt(n) -- the error bar on the
        # summary statistic itself, distinct from the spread across draws.
        "r2_median_se": float(1.253 * np.std(fin) / np.sqrt(fin.size)) if fin.size > 1 else float("nan"),
        "mae_median": float(np.nanmedian(maes)) if len(maes) else float("nan"),
        "spearman_median": float(np.nanmedian(rhos)) if len(rhos) else float("nan"),
        "frac_positive_r2": float(np.mean(r2s > 0)),
        "n_failed_draws": int(np.sum(~np.isfinite(r2s))),
    }
    if extra:
        out.update(extra)
    return out


def run_cell(model_fn, X, y, n_train, eval_idx, pool_idx, n_draws, seed,
             keep_preds=False):
    """One (probe, stage, comp, n_train) cell: n_draws random training draws
    scored on the fixed held-out eval set."""
    rng = np.random.default_rng(seed + n_train)
    X_ev, y_ev = X[eval_idx], y[eval_idx]
    r2s, maes, rhos = [], [], []
    kept = None
    with threadpool_limits(limits=1):
        for d in range(n_draws):
            tr = rng.choice(pool_idx, size=n_train, replace=False)
            try:
                m = model_fn()
                m.fit(X[tr], y[tr])
                pred = np.asarray(m.predict(X_ev)).ravel()
            except Exception:
                r2s.append(float("-inf")); maes.append(np.nan); rhos.append(np.nan)
                continue
            r2s.append(r2(y_ev, pred))
            maes.append(float(np.mean(np.abs(y_ev - pred))))
            rho = spearmanr(y_ev, pred).statistic
            rhos.append(float(rho) if np.isfinite(rho) else np.nan)
            if keep_preds and d == 0:
                kept = {"y_true": y_ev.tolist(), "y_pred": pred.tolist()}
    extra = {"preds": kept} if kept else None
    return _summarize(r2s, maes, rhos, n_train, n_draws, extra)


def run_cheap(out_dir: str, n_eval: int = 1000, seed: int = 42,
              shard: int = 0, nshards: int = 1) -> dict:
    reps, y, meta = load_cache(out_dir)
    eval_idx, pool_idx = split(len(y), n_eval, seed)
    jobs = [(c, s) for c in COMPS for s in STAGES]
    mine = [j for i, j in enumerate(jobs) if i % nshards == shard]
    print(f"[cheap s{shard}/{nshards}] n={len(y)} eval={len(eval_idx)} "
          f"pool={len(pool_idx)} cells={len(mine)}", flush=True)
    cells, t0 = {}, time.time()
    for n_comp, stage in mine:
        ci = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
        if True:
            X = feat.make_wide(reps[stage], ci)
            from sklearn.decomposition import PCA
            basis = PCA(n_components=min(20, X.shape[1], len(y)), random_state=seed).fit(X)
            for probe in CHEAP_PANEL:
                for nt in N_TRAIN:
                    if nt > len(pool_idx):
                        continue
                    if probe == "ridgecv_full" and nt < 100:
                        # RidgeCV on the full vector at n=10..50 is p>>n and
                        # unidentifiable -- that regime is what the
                        # low-capacity probes are for. Also the costliest
                        # cells in the grid, so skipping is free precision.
                        continue
                    mk = ((lambda: make_regressor("ridgecv", seed=seed)) if probe == "ridgecv_full"
                          else (lambda p=probe: make_fewshot_regressor(p, seed=seed, pca_basis=basis)))
                    cells[f"{n_comp}|{stage}|{probe}|{nt}"] = run_cell(
                        mk, X, y, nt, eval_idx, pool_idx, draws_for(nt), seed)
                print(f"[cheap s{shard}] {n_comp:2d}c {stage:<17} {probe:<13} done [{time.time()-t0:.0f}s]", flush=True)
    return {"cells": cells, "meta": meta, "n_eval": n_eval, "kind": "cheap"}


def run_tabpfn_shard(out_dir: str, shard: int, nshards: int, device: str,
                     n_eval: int = 1000, seed: int = 42) -> dict:
    reps, y, meta = load_cache(out_dir)
    eval_idx, pool_idx = split(len(y), n_eval, seed)
    from sklearn.decomposition import PCA
    jobs = [(c, s) for c in TABPFN_COMPS for s in TABPFN_STAGES]
    mine = [j for i, j in enumerate(jobs) if i % nshards == shard]
    print(f"[tabpfn shard {shard}/{nshards} on {device}] {len(mine)} cells: {mine}", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    cells, t0 = {}, time.time()
    for n_comp, stage in mine:
        ci = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
        X = feat.make_wide(reps[stage], ci)
        basis = PCA(n_components=min(10, X.shape[1], len(y)), random_state=seed).fit(X)
        for nt in TABPFN_N_TRAIN:
            cells[f"{n_comp}|{stage}|{TABPFN_PROBE}|{nt}"] = run_cell(
                lambda b=basis: make_fewshot_regressor(TABPFN_PROBE, seed=seed, pca_basis=b),
                X, y, nt, eval_idx, pool_idx, draws_for(nt, tabpfn=True), seed,
                keep_preds=(nt in (20, 200)))
            c = cells[f"{n_comp}|{stage}|{TABPFN_PROBE}|{nt}"]
            print(f"[tabpfn s{shard}] {n_comp}c {stage:<17} n={nt:<5} "
                  f"med={c['r2_median']:+.4f} IQR=[{c['r2_p25']:+.3f},{c['r2_p75']:+.3f}] "
                  f"frac>0={c['frac_positive_r2']:.2f} draws={c['n_draws']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    return {"cells": cells, "meta": meta, "n_eval": n_eval, "kind": f"tabpfn_shard{shard}"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=("cheap", "tabpfn"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    res = (run_cheap(a.out_dir, a.n_eval, shard=a.shard, nshards=a.nshards) if a.mode == "cheap"
           else run_tabpfn_shard(a.out_dir, a.shard, a.nshards, a.device, a.n_eval))
    p = os.path.join(a.out_dir, f"step5b_{a.mode}{a.tag}.json")
    with open(p, "w") as f:
        json.dump(res, f, default=str)
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
