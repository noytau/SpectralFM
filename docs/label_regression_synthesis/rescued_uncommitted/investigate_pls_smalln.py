"""
Investigate the parallel-branch finding (SpectralFM-label-regression,
step6_pls_correction.md 'Update 2026-09-10'): raw input's true ceiling under
PLS is much higher than RidgeCV/other probes reach (0.825 vs 0.404 asymptotic
at 1-comp raw, n=4716), because PLS selects directions by label-covariance
instead of shrinking by singular value -- and raw input's near-degenerate
collinear directions are exactly where our probe panel's shrinkage-based
methods (ridge_strong, gp_rbf, krr_rbf) lose the signal.

That branch's own panel never included PLS at n<=50 (our own CONFIRM_PANEL
doesn't either -- verified: pls64 only enters via LARGE_N_EXTRAS at n>50).
This script closes that gap: add a fold-internal PLS probe (k capped safely
for small n, via the existing _PLSWrapper) to the small-n panel for the two
series where the ceiling was found (raw_whiten, raw_unwhiten, z_unwhiten),
at n=10,15,20,30,50, with a shuffled-label canary run once per series to
rule out any leak (our _PLSWrapper.fit(X[tr], y[tr]) is called per-draw
inside score_readout, same as every other probe -- expected clean, verified
here rather than assumed).
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from threadpoolctl import threadpool_limits

from eval.label_probe import geometry as geo
from eval.label_probe import readouts as ro
from eval.label_probe import screen as scr
from eval.label_probe import study6
from eval.label_probe.regressors import _PLSWrapper

RUNGS = (10, 15, 20, 30, 50)
N_DRAWS = 300
SEED = 42

SERIES = {
    "raw_whiten": geo.Readout("raw", "raw", "-", "whiten"),
    "raw_unwhiten": geo.Readout("raw", "raw", "-", "none"),
    "z_unwhiten": geo.Readout("raw", "z", "-", "none"),
}


def pls_probe_factory(name, seed, X_all):
    # k must stay << n_train for a stable fold-internal fit; sweep small k
    # values known from the parallel study to matter well before the n=4716
    # optimum (k~96-128) -- at n=50 even k=8 already uses 1/6 of the budget.
    k = int(name.split("_")[1])
    return _PLSWrapper(n_components=k)


PLS_PANEL = tuple(f"pls_{k}" for k in (2, 4, 8))


def best_of(panel_result):
    finite = [(p, c) for p, c in panel_result.items() if np.isfinite(c["r2_median"])]
    return max(finite, key=lambda t: t[1]["r2_median"])


def canary(X, y, split, n, name_probe):
    """Shuffled-label canary: same protocol, same n, labels permuted. Expect ~0."""
    rng = np.random.default_rng(0)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    draws = {n: scr.draw_indices(split["pool"], n, 60, seed=SEED + 2000)}
    with threadpool_limits(limits=1):
        s = scr.score_readout(X, y_shuf, split, (name_probe,), (n,), draws, "eval_b",
                              X_for_graph=X, seed=SEED, probe_factory=pls_probe_factory)
    return s[name_probe][n]["r2_median"]


def main():
    bank, input_raw, input_z, y, meta = ro.load_bank_cache(
        "eval_outputs/label_probe/step6/bank/bank.npz")
    split = scr.make_split(len(y), seed=SEED)
    comp_idx = list(range(len(study6.COMP_LADDER[1])))
    cache = {}

    # already-measured current best-of-panel numbers (full_crossover_measured.json),
    # no need to recompute -- avoids re-running TabPFN for nothing.
    import json
    existing_json = json.load(open(
        "/tmp/claude-1040/-mnt5-home-hadar-nova-SpectralFM-few-shot/"
        "3f1d465e-2c1b-4fbb-aa8b-a1becdef4194/scratchpad/full_crossover_measured.json"))["series"]

    for sname, rd in SERIES.items():
        X = study6.materialize(rd, bank, input_raw, input_z, comp_idx, cache, seed=SEED)
        print(f"\n=== {sname} ({rd.label()}) X.shape={X.shape} ===", flush=True)
        for n in RUNGS:
            draws = {n: scr.draw_indices(split["pool"], n, N_DRAWS, seed=SEED + 1000)}
            with threadpool_limits(limits=1):
                s_pls = scr.score_readout(X, y, split, PLS_PANEL, (n,), draws, "eval_b",
                                          X_for_graph=X, seed=SEED, probe_factory=pls_probe_factory)
            best_pls_name, best_pls = best_of({p: s_pls[p][n] for p in PLS_PANEL})
            existing_r2 = existing_json[sname][str(n)]["r2_median"]

            can = canary(X, y, split, n, best_pls_name)
            print(f"n={n:<4} best_pls={best_pls_name:<8} r2={best_pls['r2_median']:+.4f}  |  "
                  f"existing(measured)  r2={existing_r2:+.4f}  |  "
                  f"canary(shuffled y)={can:+.4f}", flush=True)


if __name__ == "__main__":
    main()
