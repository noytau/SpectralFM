"""
Dedicated single-protocol measurement pass for the report's expanded crossover
chart: 5 fixed recipes x 11 rungs (n=10 to n=3716, the full training pool),
TabPFN excluded from every 'best' selection, draws bumped well above the
published step6/7 budget for tighter error bars. Real measurements only --
nothing here is interpolated or borrowed from a different eval split.

Run with the venv-labelprobe interpreter from code/. Logs progress with
flush=True to eval_outputs/label_probe/_report_crossover_full.log.
"""
import json
import sys
import time

sys.path.insert(0, ".")

import numpy as np
from threadpoolctl import threadpool_limits

from eval.label_probe import crossover as cx
from eval.label_probe import geometry as geo
from eval.label_probe import readouts as ro
from eval.label_probe import screen as scr
from eval.label_probe import study6

OUT_JSON = ("/tmp/claude-1040/-mnt5-home-hadar-nova-SpectralFM-few-shot/"
            "3f1d465e-2c1b-4fbb-aa8b-a1becdef4194/scratchpad/full_crossover_measured.json")

RUNGS = (10, 15, 20, 30, 50, 100, 200, 500, 1000, 2000, 3716)
DRAWS = {10: 300, 15: 300, 20: 300, 30: 300, 50: 300,
         100: 200, 200: 150, 500: 100, 1000: 60, 2000: 40, 3716: 1}

SERIES = {
    "raw_whiten": geo.Readout("raw", "raw", "-", "whiten"),
    "emb_whiten": geo.Readout("emb", "layer1", "segment4", "whiten"),
    "historical": cx.HISTORICAL,  # emb|layer12|mean|none
    "raw_unwhiten": geo.Readout("raw", "raw", "-", "none"),
    "z_unwhiten": geo.Readout("raw", "z", "-", "none"),
}

SEED = 42


def best_excl_tabpfn(panel_result: dict) -> dict:
    """panel_result: {probe_name: cell_dict}. Argmax over r2_median, TabPFN
    excluded, non-finite guarded (never let a NaN-median probe win)."""
    cands = [(p, c) for p, c in panel_result.items() if not p.startswith("tabpfn")]
    finite = [(p, c) for p, c in cands if np.isfinite(c["r2_median"])]
    if not finite:
        return {"r2_median": float("nan"), "r2_p25": float("nan"),
                "r2_p75": float("nan"), "probe": None, "n_draws": cands[0][1]["n_draws"]}
    p, c = max(finite, key=lambda t: t[1]["r2_median"])
    out = dict(c)
    out["probe"] = p
    return out


def main():
    t_start = time.time()
    bank, input_raw, input_z, y, meta = ro.load_bank_cache(
        "eval_outputs/label_probe/step6/bank/bank.npz")
    n_total = len(y)
    split = scr.make_split(n_total, seed=SEED)
    print(f"[measure] n={n_total} pool={len(split['pool'])}", flush=True)

    comp_idx = list(range(len(study6.COMP_LADDER[1])))  # 1-component
    cache = {}
    X_by_series = {}
    for name, rd in SERIES.items():
        X_by_series[name] = study6.materialize(rd, bank, input_raw, input_z,
                                               comp_idx, cache, seed=SEED)
        print(f"[measure] materialized {name} ({rd.label()}) -> "
              f"{X_by_series[name].shape}", flush=True)

    results = {name: {} for name in SERIES}
    for n in RUNGS:
        n_draws = DRAWS[n]
        draws = {n: scr.draw_indices(split["pool"], n, n_draws, seed=SEED + 1000)}
        panel = cx.panel_for_n(n)
        for name, rd in SERIES.items():
            X = X_by_series[name]
            with threadpool_limits(limits=1):
                s = scr.score_readout(X, y, split, panel, (n,), draws, "eval_b",
                                      X_for_graph=X, seed=SEED, probe_factory=cx.make_probe_n)
            panel_result = {p: s[p][n] for p in panel}
            best = best_excl_tabpfn(panel_result)
            results[name][str(n)] = {
                "r2_median": best["r2_median"], "r2_p25": best["r2_p25"],
                "r2_p75": best["r2_p75"], "probe": best["probe"], "n_draws": n_draws,
            }
            print(f"[measure] n={n:<5} {name:<14} r2={best['r2_median']:+.4f} "
                  f"({best['probe']})  [{time.time()-t_start:.0f}s elapsed]", flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump({"rungs": list(RUNGS), "draws": DRAWS, "series": results}, f, indent=2)
    print(f"[measure] wrote {OUT_JSON}  total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
