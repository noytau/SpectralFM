"""
Label efficiency under nested CV: how well does each arm do with only n
labels, when it also has to choose its recipe from those n labels alone?

For each outer fold (plain 5-fold), and each label budget n, several random
n-row subsets are drawn from the training fold. On each subset an inner
K-fold CV over those n rows only picks one of the 12 recipes (nested.RECIPES);
the winner is refit on the subset and scored on the whole held-out fold.
Raw input and the embedding use the SAME subsets, so their gap at each draw is
paired; curves are medians over (fold, draw).

The embedding arm is a single block fixed in advance (--block, or
--block_from a run dir whose nested per-block scores pick it), so the budget
is spent on the recipe choice, not on a block search.

  python -m eval.label_probe.nested_ladder <run_dir> [<run_dir> ...] \
      (--block layer0 | --block_from <run_dir>) [--n_jobs 10]

writes <run_dir>/nested_ladder.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from . import features as feat
from . import readouts as ro
from .nested import RECIPES, _fit_predict, _inner_scores, fold_transforms
from .protocol import r2

LADDER_N = (10, 20, 50, 100, 200, 500, 1000, 2000)


def n_draws(n, n_full, max_draws=None):
    """More draws where one draw is noisy and cheap; one at the full fold."""
    if n >= n_full:
        return 1
    d = 20 if n <= 100 else 10 if n <= 500 else 5
    return min(d, max_draws) if max_draws else d


def _one_draw(arms, y, sub, te, n_inner, seed):
    """Every arm on one training subset: recipe chosen by inner CV on `sub`
    only, refit on `sub`, scored on the held-out fold `te`."""
    out = {}
    with threadpool_limits(limits=1):
        for name, X in arms.items():
            inner = _inner_scores(X[sub], y[sub], min(n_inner, len(sub)), seed)
            norm, probe = max(RECIPES, key=lambda rc: inner[rc])
            a, b = fold_transforms(X[sub], X[te], seed=seed)[norm]
            out[name] = {"r2": r2(y[te], _fit_predict(a, y[sub], b, probe, seed)),
                         "recipe": f"{norm}+{probe}"}
    return out


def nested_ladder(arms: dict, y: np.ndarray, n_trains=LADDER_N, n_folds: int = 5,
                  n_inner: int = 5, seed: int = 42, n_jobs: int = 1, max_draws: int = None) -> dict:
    """arms: {name: X [n, d]}. Returns {"rungs": [...], "arms": {name: {rung:
    {median, p25, p75, recipes}}}, "gaps": {name: {rung: median paired gap to
    the first arm}}, "draws": {rung: count}}."""
    y = np.asarray(y, dtype=np.float64)
    folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(y))
    n_full = min(len(tr) for tr, _ in folds)
    rungs = [n for n in n_trains if n < n_full] + [n_full]
    jobs = []
    for f, (tr, te) in enumerate(folds):
        rng = np.random.default_rng(seed + 100 * f)
        for n in rungs:
            for d in range(n_draws(n, n_full, max_draws)):
                sub = tr if n >= n_full else rng.choice(tr, size=n, replace=False)
                jobs.append((n, (arms, y, np.sort(sub), te, n_inner, seed + 7 * d + 1000 * f)))
    if n_jobs > 1:
        from joblib import Parallel, delayed
        outs = Parallel(n_jobs=n_jobs)(delayed(_one_draw)(*a) for _, a in jobs)
    else:
        outs = [_one_draw(*a) for _, a in jobs]

    names = list(arms)
    per = {a: collections.defaultdict(list) for a in names}
    rec = {a: collections.defaultdict(collections.Counter) for a in names}
    gap = {a: collections.defaultdict(list) for a in names[1:]}
    for (n, _), o in zip(jobs, outs):
        for a in names:
            per[a][n].append(o[a]["r2"])
            rec[a][n][o[a]["recipe"]] += 1
        for a in names[1:]:
            gap[a][n].append(o[a]["r2"] - o[names[0]]["r2"])
    summ = lambda v: {"median": float(np.median(v)), "p25": float(np.percentile(v, 25)),
                      "p75": float(np.percentile(v, 75))}
    return {
        "rungs": rungs,
        "draws": {str(n): n_folds * n_draws(n, n_full, max_draws) for n in rungs},
        "arms": {a: {str(n): {**summ(per[a][n]), "recipes": dict(rec[a][n].most_common())}
                     for n in rungs} for a in names},
        "gaps": {a: {str(n): {**summ(gap[a][n]), "frac_positive": float(np.mean(np.array(gap[a][n]) > 0))}
                     for n in rungs} for a in names[1:]},
    }


def best_block(run_dir: str) -> str:
    """The block with the highest nested-CV score on its own, in run_dir."""
    fam = json.load(open(os.path.join(run_dir, "nested_results.json")))["families"]
    blocks = {k: v for k, v in fam.items() if k != "raw" and not k.startswith("embedding")}
    return max(blocks, key=lambda k: blocks[k]["r2_mean"])


def run_ladder_for_dir(run_dir: str, block: str, n_jobs: int = 1, seed: int = 42,
                       block_source: str = None) -> str:
    bank, input_raw, _, y, meta = ro.load_bank_cache(os.path.join(run_dir, "bank.npz"))
    ci = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[1]]
    arms = {"raw": feat.make_wide(input_raw, ci), "block": ro.build_readout(bank[block], ci, "mean")}
    out = nested_ladder(arms, y, seed=seed, n_jobs=n_jobs)
    out["block"] = block
    out["block_source"] = block_source
    out["meta"] = meta
    path = os.path.join(run_dir, "nested_ladder.json")
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, indent=2, default=str)
    os.replace(path + ".tmp", path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", nargs="+")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--block", help="bank stage key, e.g. layer0")
    g.add_argument("--block_from", help="run dir whose nested per-block scores pick the block")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    block = a.block or best_block(a.block_from)
    for d in a.run_dirs:
        print(f"[nested_ladder] wrote {run_ladder_for_dir(d, block, a.n_jobs, a.seed, a.block_from)} "
              f"(block {block})", flush=True)


if __name__ == "__main__":
    main()
