"""
Step 6 -- can any frozen-backbone readout beat raw input at 20-50 labels?

Screen cheap on eval_a, confirm expensive on eval_b, verdict by the rule
pre-registered in the design doc sec. 2. See
docs/superpowers/specs/2026-09-09-few-shot-readout-geometry-design.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import geometry as geo
from . import readouts as ro
from . import screen as scr

COMP_LADDER = {1: (0,), 2: (0, 1), 3: (0, 1, 2)}
N_TRAINS = (20, 50)
RAW_STAGES = ("raw", "z", "moments")


def iter_readouts(wave: int, stages=None) -> list:
    """
    wave 0 -> the raw-input family (3 stages x 4 normalizers).
    wave 1 -> every embedding stage, pooling fixed to mean (H-B, H-C).
    wave 2 -> the given finalist stages x every pooling (H-A).
    """
    out = []
    if wave == 0:
        for stage in RAW_STAGES:
            for nrm in geo.NORMALIZERS:
                out.append(geo.Readout("raw", stage, "-", nrm))
    elif wave == 1:
        for stage in ro.BANK_STAGES:
            for nrm in geo.NORMALIZERS:
                out.append(geo.Readout("emb", stage, "mean", nrm))
    elif wave == 2:
        for stage in stages:
            for pooling in ro.POOLINGS:
                for nrm in geo.NORMALIZERS:
                    out.append(geo.Readout("emb", stage, pooling, nrm))
    else:
        raise ValueError(f"unknown wave {wave}")
    return out


def materialize(readout, bank: dict, input_raw, input_z, comp_idx: list,
                cache: dict, seed: int = 42) -> np.ndarray:
    """
    Assemble one readout's [N, D] matrix and apply its unlabeled normalizer.

    `cache` memoizes the fitted normalizer per (stage, pooling, comps): a
    whiten fit is a randomized SVD over the whole corpus and is the single
    most expensive operation in the screen, so refitting it per probe would
    dominate the runtime.
    """
    key = (readout.family, readout.stage, readout.pooling,
           readout.normalizer, tuple(comp_idx))
    if readout.family == "raw":
        if readout.stage == "raw":
            base = np.asarray(input_raw)[:, comp_idx, :]
        elif readout.stage == "z":
            base = np.asarray(input_z)[:, comp_idx, :]
        elif readout.stage == "moments":
            base = ro.raw_moments(np.asarray(input_raw))[:, comp_idx, :]
        else:
            raise ValueError(f"unknown raw stage {readout.stage!r}")
        X = base.reshape(base.shape[0], -1).astype(np.float32)
    else:
        arr = np.asarray(bank[readout.stage], dtype=np.float32)
        X = ro.build_readout(arr, comp_idx, readout.pooling)

    if key not in cache:
        cache[key] = geo.fit_normalizer(readout.normalizer, X, seed=seed)
    return cache[key].transform(X)


def _cell_key(n_comp, n_train):
    return f"{n_comp}|{n_train}"


def _best_finite(candidates, score_fn):
    """argmax(candidates, key=score_fn), but a NaN score never wins.

    `max()` compares pairwise via `>`, and `x > nan` and `nan > x` are both
    False for every x -- so a plain `max(..., key=...)` can get permanently
    stuck on a NaN candidate (e.g. a probe whose every draw failed) if that
    candidate is encountered first, silently reporting a failed probe as the
    best. Filter to finite scores before taking the max; if none are finite,
    say so explicitly instead of picking one anyway.

    Returns (winner, all_failed). When all_failed is True, `winner` is an
    arbitrary candidate (the first) purely so callers that need *a* record
    to attach metadata to still have one -- its score must not be reported
    as a real result without the all_failed flag alongside it.
    """
    finite = [c for c in candidates if np.isfinite(score_fn(c))]
    if finite:
        return max(finite, key=score_fn), False
    return candidates[0], True


def run_screen(bank_path: str, output_dir: str, comp_counts=(1, 2, 3),
               n_draws: int = 30, seed: int = 42, n_top_stages: int = 4) -> dict:
    """
    Wave 1 (stage x normalizer, mean pooling) then wave 2 (top stages x every
    pooling), plus the raw family, all scored on eval_a only.

    DEVIATION from a naive wave-1-then-wave-2 sweep: wave 2 iterates every
    pooling for the finalist stages, which regenerates the pooling="mean"
    readouts wave 1 already scored -- byte-identical cells, each costing the
    full probe panel plus diagnostics. `_score` skips any readout whose
    result key is already present in `results["readouts"]` before scoring
    it, so those cells are never recomputed; the skip count is printed so
    the dedup is visible in the run log rather than silent, and is also
    recorded in the returned/saved `results["wave2_skipped"][n_comp]` so it
    survives into `step6_screen.json` for later reconciliation.
    """
    os.makedirs(output_dir, exist_ok=True)
    bank, input_raw, input_z, y, meta = ro.load_bank_cache(bank_path)
    n = len(y)
    split = scr.make_split(n, seed=seed)
    draws = {nt: scr.draw_indices(split["pool"], nt, n_draws, seed=seed)
             for nt in N_TRAINS}
    print(f"[step6:screen] n={n} eval_a={len(split['eval_a'])} "
          f"eval_b={len(split['eval_b'])} pool={len(split['pool'])}", flush=True)

    results = {"meta": meta, "n_draws": n_draws, "seed": seed,
               "split_sizes": {k: len(v) for k, v in split.items()},
               "readouts": {}, "diagnostics": {}, "finalist_stages": {},
               "wave2_skipped": {}}

    for n_comp in comp_counts:
        comp_idx = list(range(len(COMP_LADDER[n_comp])))
        cache = {}
        grid = iter_readouts(wave=0) + iter_readouts(wave=1)

        def _score(rd_list, tag):
            n_skipped = 0
            for rd in rd_list:
                result_key = f"{n_comp}|{rd.label()}"
                if result_key in results["readouts"]:
                    n_skipped += 1
                    continue
                X = materialize(rd, bank, input_raw, input_z, comp_idx, cache,
                                seed=seed)
                s = scr.score_readout(X, y, split, geo.SCREEN_PANEL, N_TRAINS,
                                      draws, "eval_a", seed=seed)
                candidates = [(p, nt) for p in s for nt in N_TRAINS]
                (best_p, best_nt), all_failed = _best_finite(
                    candidates, lambda t: s[t[0]][t[1]]["r2_median"])
                if all_failed:
                    print(f"[step6:screen:{tag}] n_comp={n_comp} "
                          f"{rd.label():<40} ALL probes failed (NaN median) "
                          "-- no finite candidate", flush=True)
                    best = (None, None, float("nan"))
                else:
                    best = (best_p, best_nt, s[best_p][best_nt]["r2_median"])
                d = scr.readout_diagnostics(X, y, seed=seed)
                results["readouts"][result_key] = {
                    "n_comp": n_comp, "family": rd.family, "stage": rd.stage,
                    "pooling": rd.pooling, "normalizer": rd.normalizer,
                    "scores": s, "best_probe": best[0],
                    "best_n_train": best[1], "best_r2": best[2],
                    "best_all_failed": all_failed,
                }
                results["diagnostics"][result_key] = d
                print(f"[step6:screen:{tag}] n_comp={n_comp} {rd.label():<40} "
                      f"best={best[0]}@n{best[1]}:{best[2]:+.4f} "
                      f"pr={d['participation_ratio']:.1f} "
                      f"topk10={d['whitened_topk_r2'][10]:+.3f}", flush=True)
            if n_skipped:
                print(f"[step6:screen:{tag}] n_comp={n_comp} skipped "
                      f"{n_skipped} readouts already scored", flush=True)
            return n_skipped

        _score(grid, "w1")

        emb = [(k, v) for k, v in results["readouts"].items()
               if v["n_comp"] == n_comp and v["family"] == "emb"]
        by_stage = {}
        for _, v in emb:
            by_stage[v["stage"]] = max(by_stage.get(v["stage"], -np.inf),
                                       v["best_r2"])
        top_stages = [s for s, _ in sorted(by_stage.items(), key=lambda t: -t[1])
                      ][:n_top_stages]
        results["finalist_stages"][n_comp] = top_stages
        print(f"[step6:screen] n_comp={n_comp} wave-2 stages: {top_stages}",
              flush=True)

        results["wave2_skipped"][n_comp] = _score(
            iter_readouts(wave=2, stages=top_stages), "w2")

    path = os.path.join(output_dir, "step6_screen.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[step6:screen] wrote {path}", flush=True)
    return results


def run_confirm(bank_path: str, screen_results: dict, output_dir: str,
                n_draws: int = 100, top_emb: int = 3, top_raw: int = 2,
                use_tabpfn: bool = True, seed: int = 42) -> dict:
    """
    Re-score the screen's finalists on eval_b with the full probe panel.
    Draw seed is deliberately offset from the screen's so the confirmation
    uses different training draws as well as a different evaluation set.
    """
    from .regressors import tabpfn_available

    os.makedirs(output_dir, exist_ok=True)
    bank, input_raw, input_z, y, meta = ro.load_bank_cache(bank_path)
    split = scr.make_split(len(y), seed=seed)
    confirm_seed = seed + 1000
    draws = {nt: scr.draw_indices(split["pool"], nt, n_draws, seed=confirm_seed)
             for nt in N_TRAINS}

    panel = list(geo.CONFIRM_PANEL)
    if not use_tabpfn or not tabpfn_available():
        dropped = [p for p in panel if p.startswith("tabpfn")]
        panel = [p for p in panel if not p.startswith("tabpfn")]
        if dropped:
            print(f"[step6:confirm] tabpfn unavailable/disabled -- dropping "
                  f"{dropped}", flush=True)

    comp_counts = sorted({v["n_comp"] for v in screen_results["readouts"].values()})
    results = {"meta": meta, "n_draws": n_draws, "seed": confirm_seed,
               "panel": panel, "finalists": {}, "cells": {}, "detail": {}}

    for n_comp in comp_counts:
        comp_idx = list(range(len(COMP_LADDER[n_comp])))
        cache = {}
        rows = [(k, v) for k, v in screen_results["readouts"].items()
                if v["n_comp"] == n_comp]
        picks = []
        for fam, k_top in (("emb", top_emb), ("raw", top_raw)):
            fam_rows = sorted([r for r in rows if r[1]["family"] == fam],
                              key=lambda t: -t[1]["best_r2"])[:k_top]
            picks += fam_rows
        results["finalists"][n_comp] = [k for k, _ in picks]
        print(f"[step6:confirm] n_comp={n_comp} finalists: "
              f"{[k for k, _ in picks]}", flush=True)

        best_by_family = {"emb": None, "raw": None}
        for key, v in picks:
            rd = geo.Readout(v["family"], v["stage"], v["pooling"], v["normalizer"])
            X = materialize(rd, bank, input_raw, input_z, comp_idx, cache, seed=seed)
            s = scr.score_readout(X, y, split, panel, N_TRAINS, draws, "eval_b",
                                  seed=seed)
            results["detail"][f"{n_comp}|{rd.label()}"] = s
            for nt in N_TRAINS:
                # Compute the winning probe ONCE (not once for the dict and
                # once for the name -- those can disagree under ties or,
                # worse, both independently get stuck on a NaN-median probe
                # that comes first in `panel`) and never let a NaN-median
                # (all-draws-failed) probe be picked over a finite one.
                winner_probe, all_failed = _best_finite(
                    panel, lambda p: s[p][nt]["r2_median"])
                cand = dict(s[winner_probe][nt])
                cand["readout"] = rd.label()
                cand["probe"] = winner_probe
                cand["all_failed"] = all_failed
                if all_failed:
                    print(f"[step6:confirm] n_comp={n_comp} {rd.label():<40} "
                          f"n={nt}: ALL probes failed (NaN median) -- no "
                          "finite candidate", flush=True)
                cur = best_by_family[v["family"]]
                slot = (cur or {}).get(nt)
                if slot is None or slot.get("all_failed") or (
                        not all_failed and cand["r2_median"] > slot["r2_median"]):
                    best_by_family[v["family"]] = {**(cur or {}), nt: cand}
                print(f"[step6:confirm] n_comp={n_comp} {rd.label():<40} "
                      f"n={nt} best={cand['probe']}:{cand['r2_median']:+.4f} "
                      f"frac>0={cand['frac_positive_r2']:.2f}", flush=True)

        for nt in N_TRAINS:
            results["cells"][_cell_key(n_comp, nt)] = {
                "emb": best_by_family["emb"][nt],
                "raw": best_by_family["raw"][nt],
            }

    path = os.path.join(output_dir, "step6_confirm.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[step6:confirm] wrote {path}", flush=True)
    return results


def verdict(confirm_results: dict, n_boot: int = 2000, seed: int = 42) -> dict:
    """
    The pre-registered rule (design doc sec. 2), applied verbatim:

      a cell is WON if the paired median delta (emb - raw) over the shared
      draws is > 0 with >=90% of draw-level bootstrap resamples positive,
      AND the embedding's frac_positive_r2 >= 0.8.

      >=4 of 6 cells won -> WIN; 2-3 -> PARTIAL; else NO.

    The reliability clause matters: an embedding that edges raw on the median
    while working half the time is not a deployable win.
    """
    rng = np.random.default_rng(seed)
    cells_out, won = {}, 0
    for key, cell in confirm_results["cells"].items():
        e = np.array(cell["emb"]["r2_per_draw"], dtype=float)
        r = np.array(cell["raw"]["r2_per_draw"], dtype=float)
        m = min(len(e), len(r))
        d = e[:m] - r[:m]                     # paired: identical draws
        fin = d[np.isfinite(d)]
        n_pairs = int(fin.size)
        n_dropped = int(m - fin.size)
        if fin.size:
            boots = np.array([np.median(rng.choice(fin, fin.size, replace=True))
                              for _ in range(n_boot)])
            frac_pos = float(np.mean(boots > 0))
            med = float(np.median(fin))
        else:
            frac_pos, med = 0.0, float("nan")
        reliable = cell["emb"].get("frac_positive_r2", 0.0) >= 0.8
        cell_won = bool(med > 0 and frac_pos >= 0.90 and reliable)
        won += int(cell_won)
        cells_out[key] = {
            "delta_median": med, "boot_frac_positive": frac_pos,
            "emb_frac_positive_r2": cell["emb"].get("frac_positive_r2"),
            "emb_r2_median": cell["emb"]["r2_median"],
            "raw_r2_median": cell["raw"]["r2_median"],
            "reliable": reliable, "won": cell_won,
            "n_pairs": n_pairs, "n_dropped": n_dropped,
        }
    v = "WIN" if won >= 4 else ("PARTIAL" if won >= 2 else "NO")
    return {"verdict": v, "n_cells_won": won, "n_cells": len(cells_out),
            "cells": cells_out}
