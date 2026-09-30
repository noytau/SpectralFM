"""
Nested cross-validation: every arm's recipe is chosen without ever seeing
the rows that score it.

Why this exists. The stage grid (study.run_stage_grid) scores every
(block, normalizer, probe) cell on the same CV folds and then reports each
arm's best cell. That max is optimistic, and unevenly so: raw input picks
from 12 recipes, an embedding from ~15 blocks x 12 = 180, so the embedding
collects more winner's-curse inflation. Here the choice happens INSIDE each
outer training fold (inner K-fold over the training rows only), the winner
is refit on the whole training fold, and only then predicts the held-out
fold. Raw input and the embedding are treated identically, and no score is
ever picked on the data that grades it.

Normalizers are fit on the training rows of whichever fold is being fit --
stricter than the rest of the package (normalize.py fits once on the whole
unlabeled pool, which is label-free and legitimate transductively). Here the
held-out rows are unseen in every sense.

Every arm uses the same outer folds (KFold, shuffle, random_state=seed+r --
the same splits protocol.run_primary draws), so out-of-fold predictions of
two arms -- or of two backbones on the same dataset -- are paired, and
`paired_delta` gives the SD of their difference from resampling the same
spectra for both. That is much tighter than combining two independent SDs.

  python -m eval.label_probe.nested <run_dir> [<run_dir> ...] [--n_jobs 10]

reads each <run_dir>/bank.npz and writes nested_results.json (scores,
choices) + nested_oof.npz (out-of-fold predictions, for pairing).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from . import features as feat
from . import readouts as ro
from .normalize import fit_normalizer
from .protocol import r2
from .regressors import make_regressor

NORMALIZERS = ("none", "standardize", "whiten", "whiten8", "whiten32", "whiten128")
PROBES = ("ridgecv", "ols")
RECIPES = tuple((n, p) for n in NORMALIZERS for p in PROBES)

# The block-average readout: the ENSEMBLE_K best blocks (by inner CV, each
# at its own best recipe), predictions averaged.
ENSEMBLE_K = 3

# Below this many labeled spectra nested CV is not meaningful: the inner
# folds would fit on a handful of rows. Such label sets are skipped.
MIN_N = 20


def fold_transforms(X_tr: np.ndarray, X_te: np.ndarray, seed: int = 42) -> dict:
    """{normalizer: (X_tr', X_te')}, every normalizer fit on X_tr only.
    The whitenK variants are the first K columns of one full-rank whitening
    -- identical to fitting each separately (exact SVD), at a sixth of the
    cost."""
    out = {}
    for name in ("none", "standardize"):
        t = fit_normalizer(name, X_tr, seed=seed)
        out[name] = (t.transform(X_tr), t.transform(X_te))
    w = fit_normalizer("whiten", X_tr, seed=seed)
    Ztr, Zte = w.transform(X_tr), w.transform(X_te)
    for name in NORMALIZERS:
        if name.startswith("whiten"):
            k = w.k_keep if name == "whiten" else min(w.k_keep, int(name[len("whiten"):]))
            out[name] = (Ztr[:, :k], Zte[:, :k])
    return out


def _fit_predict(X_tr, y_tr, X_te, probe, seed):
    m = make_regressor(probe, seed=seed)
    m.fit(X_tr, y_tr)
    return np.asarray(m.predict(X_te)).ravel()


def _inner_scores(X, y, n_inner, seed):
    """Inner-CV R² of every recipe, on these (training) rows only."""
    preds = {rc: np.zeros(len(y)) for rc in RECIPES}
    for itr, ite in KFold(n_inner, shuffle=True, random_state=seed).split(X):
        T = fold_transforms(X[itr], X[ite], seed=seed)
        for norm, probe in RECIPES:
            a, b = T[norm]
            preds[(norm, probe)][ite] = _fit_predict(a, y[itr], b, probe, seed)
    return {rc: r2(y, p) for rc, p in preds.items()}


def _outer_fold(arms, families, y, tr, te, n_inner, seed, fixed_arms, ensembles):
    with threadpool_limits(limits=1):
        needed = sorted({a for names in families.values() for a in names}
                        | {a for names, _ in ensembles.values() for a in names})
        inner = {a: _inner_scores(arms[a][tr], y[tr], n_inner, seed) for a in needed}
        cache = {}

        def transforms(arm):
            if arm not in cache:
                cache[arm] = fold_transforms(arms[arm][tr], arms[arm][te], seed=seed)
            return cache[arm]

        chosen = {}
        for fam, names in families.items():
            arm, (norm, probe) = max(((a, rc) for a in names for rc in RECIPES),
                                     key=lambda k: inner[k[0]][k[1]])
            a, b = transforms(arm)[norm]
            chosen[fam] = {"pred": _fit_predict(a, y[tr], b, probe, seed), "arm": arm,
                           "norm": norm, "probe": probe, "inner_r2": inner[arm][(norm, probe)]}
        for name, (names, k) in ensembles.items():
            # each arm at its own best recipe, then the k best arms
            best = {a: max(RECIPES, key=lambda rc: inner[a][rc]) for a in names}
            top = sorted(names, key=lambda a: -inner[a][best[a]])[:k]
            preds = []
            for a_ in top:
                norm, probe = best[a_]
                a, b = transforms(a_)[norm]
                preds.append(_fit_predict(a, y[tr], b, probe, seed))
            chosen[name] = {"pred": np.mean(preds, axis=0),
                            "members": [{"arm": a_, "norm": best[a_][0], "probe": best[a_][1],
                                         "inner_r2": inner[a_][best[a_]]} for a_ in top]}
        fixed = {}
        for arm in fixed_arms:
            for norm, probe in RECIPES:
                a, b = transforms(arm)[norm]
                fixed[(arm, norm, probe)] = _fit_predict(a, y[tr], b, probe, seed)
    return chosen, fixed


def bootstrap_sd(y, P, n_boot=500, seed=0):
    """SD of the repeat-averaged R² under resampling spectra. P: [R, n]."""
    rng = np.random.default_rng(seed)
    vals = [np.mean([r2(y[i], p[i]) for p in P])
            for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.std(vals))


def paired_delta(y, PA, PB, n_boot=1000, seed=0):
    """R²(A) - R²(B) on the same folds, with its bootstrap SD: both arms are
    rescored on the SAME resampled spectra, so shared difficulty cancels.
    PA, PB: [R, n] out-of-fold predictions."""
    def diff(i):
        return float(np.mean([r2(y[i], a[i]) - r2(y[i], b[i]) for a, b in zip(PA, PB)]))
    rng = np.random.default_rng(seed)
    boots = np.array([diff(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    return {"delta": diff(np.arange(len(y))), "sd": float(boots.std()),
            "p_a_better": float((boots > 0).mean())}


def nested_cv(arms: dict, families: dict, y: np.ndarray, n_repeats: int = 2,
              n_folds: int = 5, n_inner: int = 5, seed: int = 42,
              fixed_arms=(), ensembles=None, n_jobs: int = 1) -> dict:
    """arms: {name: X [n, d]}. families: {family: [arm names]} -- each family
    picks its best (arm, normalizer, probe) inside every outer training fold.
    ensembles: {name: ([arm names], k)} -- each arm at its inner-best recipe,
    the k best arms refit and their predictions averaged.
    fixed_arms: arms additionally scored at every recipe held fixed (no
    selection), same folds, fold-internal normalizer.

    Returns {family: {"oof" [R, n], "r2_per_repeat", "r2_mean",
    "bootstrap_sd", "chosen": [...]}, "fixed": {(arm, norm, probe): oof},
    "test_folds": [R][F] index lists}."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    ensembles = ensembles or {}
    jobs = []
    for r in range(n_repeats):
        for f, (tr, te) in enumerate(KFold(n_folds, shuffle=True, random_state=seed + r).split(y)):
            jobs.append((r, f, tr, te))
    args = [(arms, families, y, tr, te, n_inner, seed + 1000 + 10 * r + f, fixed_arms, ensembles)
            for r, f, tr, te in jobs]
    if n_jobs > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(_outer_fold)(*a) for a in args)
    else:
        outs = [_outer_fold(*a) for a in args]

    res = {fam: {"oof": np.zeros((n_repeats, n)), "chosen": []} for fam in [*families, *ensembles]}
    fixed = {k: np.zeros((n_repeats, n)) for k in outs[0][1]}
    test_folds = [[] for _ in range(n_repeats)]
    for (r, f, tr, te), (chosen, fx) in zip(jobs, outs):
        test_folds[r].append(te.tolist())
        for fam, c in chosen.items():
            res[fam]["oof"][r, te] = c["pred"]
            res[fam]["chosen"].append({"repeat": r, "fold": f, **{k: v for k, v in c.items() if k != "pred"}})
        for k, p in fx.items():
            fixed[k][r, te] = p
    for v in res.values():
        v["r2_per_repeat"] = [r2(y, p) for p in v["oof"]]
        v["r2_mean"] = float(np.mean(v["r2_per_repeat"]))
        v["bootstrap_sd"] = bootstrap_sd(y, v["oof"])
    res["fixed"] = fixed
    res["test_folds"] = test_folds
    return res


def run_nested(bank: dict, input_raw: np.ndarray, y: np.ndarray, n_comp: int = 1,
               n_repeats: int = 2, n_folds: int = 5, n_inner: int = 5, seed: int = 42,
               n_jobs: int = 1):
    """Raw input vs the embedding (every block, mean-pooled), each at its
    nested-selected recipe; the average of the ENSEMBLE_K best blocks; every
    block on its own (the depth profile -- and the fixed-block readout, one
    block chosen in advance); raw input at every fixed recipe (the normalizer
    comparison).
    Returns (json-able results, {family: oof [R, n]})."""
    ci = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
    arms = {"raw": feat.make_wide(input_raw, ci)}
    for stage in bank:
        arms[stage] = ro.build_readout(bank[stage], ci, "mean")
    stages = list(bank)
    families = {"raw": ["raw"], "embedding": stages, **{s: [s] for s in stages}}
    ensembles = {f"embedding_top{ENSEMBLE_K}": (stages, ENSEMBLE_K)}
    res = nested_cv(arms, families, y, n_repeats=n_repeats, n_folds=n_folds,
                    n_inner=n_inner, seed=seed, fixed_arms=("raw",), ensembles=ensembles,
                    n_jobs=n_jobs)
    families = {**families, **ensembles}
    y64 = np.asarray(y, dtype=np.float64)
    out = {
        "protocol": {"n": len(y), "n_repeats": n_repeats, "n_folds": n_folds,
                     "n_inner": n_inner, "seed": seed, "n_comp": n_comp,
                     "recipes": [f"{a}+{b}" for a, b in RECIPES]},
        "families": {fam: {k: v for k, v in res[fam].items() if k != "oof"} for fam in families},
        "raw_fixed_recipes": {f"{norm}+{probe}": {"r2_mean": float(np.mean([r2(y64, p) for p in P])),
                                                  "bootstrap_sd": bootstrap_sd(y64, P, n_boot=200)}
                              for (_, norm, probe), P in res["fixed"].items()},
        "embedding_minus_raw": paired_delta(y64, res["embedding"]["oof"], res["raw"]["oof"]),
    }
    return out, {fam: res[fam]["oof"] for fam in families}


def run_nested_for_dir(run_dir: str, n_jobs: int = 1, seed: int = 42):
    """Writes nested_results.json + nested_oof.npz into run_dir and returns
    the JSON path, or None when the label set is below MIN_N."""
    bank, input_raw, _, y, meta = ro.load_bank_cache(os.path.join(run_dir, "bank.npz"))
    if len(y) < MIN_N:
        return None
    out, oof = run_nested(bank, input_raw, y, seed=seed, n_jobs=n_jobs)
    out["meta"] = meta
    np.savez_compressed(os.path.join(run_dir, "nested_oof.npz"), y=y, **oof)
    path = os.path.join(run_dir, "nested_results.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=2, default=str)
    os.replace(path + ".tmp", path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="+", help="label_probe output dirs holding bank.npz")
    ap.add_argument("--n_jobs", type=int, default=1, help="outer folds run in parallel")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_done", action="store_true", help="skip dirs that already have nested_results.json")
    a = ap.parse_args()
    for d in a.run_dirs:
        if a.skip_done and os.path.isfile(os.path.join(d, "nested_results.json")):
            print(f"[nested] skip {d}", flush=True)
            continue
        path = run_nested_for_dir(d, n_jobs=a.n_jobs, seed=a.seed)
        print(f"[nested] wrote {path}" if path else f"[nested] skip {d} (n < {MIN_N})", flush=True)


if __name__ == "__main__":
    main()
