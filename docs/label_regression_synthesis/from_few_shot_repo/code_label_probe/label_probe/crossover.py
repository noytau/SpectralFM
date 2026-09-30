"""
Step 7 -- where does the embedding overtake raw input?

Step 6 settled that no frozen readout wins at 20-50 labels (0/6 cells). Step 1
measured the embedding WINNING at n~3,773 (0.628 vs raw 0.468, 1-comp). Those
bracket a crossing that nobody has located, because step 5 swept n only at
layer12/mean -- which step 6 showed is among the worst taps -- and step 6 swept
readouts only at n in {20, 50}.

See docs/superpowers/specs/2026-09-09-label-efficiency-crossover-design.md.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo

# The rungs. Capped at 2000 rather than 3716: draws are without replacement from
# a 3716-spectrum pool, so two draws at n=2000 already share ~54% of their rows
# and their spread understates true sampling variance. Step 1's n~3,773 result is
# carried as an anchor on the figure instead of as a rung -- different protocol.
N_LADDER = (20, 50, 100, 200, 500, 1000, 2000)

# Draws per rung. Monotonically non-increasing: estimator variance falls with n
# while per-fit cost rises, so the budget follows the information, not the grid.
DRAWS_BY_N = {20: 100, 50: 100, 100: 60, 200: 40, 500: 25, 1000: 15, 2000: 10}

# Probes that only make sense once n is large. `ridge_strong` biases its alpha
# grid to logspace(0, 6) because that is right at n=20; `ridgecv` sweeps
# logspace(-3, 3), which is right once the fit is identifiable. `pls64` is step
# 4's best large-n feature reduction.
LARGE_N_EXTRAS = ("ridgecv", "pls64")

# Cubic-or-worse in n: exact GP factorizes an n x n kernel, and krr_rbf does that
# 30 times inside a 3-fold grid search.
_CUBIC = ("gp_rbf", "krr_rbf")


def panel_for_n(n: int) -> tuple:
    """
    The probe panel at rung `n`. Both arms always receive the identical panel at
    a given rung; the panel changing ACROSS rungs is a disclosed confound (spec
    sec. 3b) and is the lesser of two evils -- a fixed panel would measure
    regularization mismatch rather than label efficiency.
    """
    if n <= 50:
        # verbatim, so the bottom rungs are comparable with published step-6 cells
        return geo.CONFIRM_PANEL
    panel = [p for p in geo.CONFIRM_PANEL if not p.startswith("tabpfn")]
    if n >= 1000:
        panel = [p for p in panel if p not in _CUBIC]
        panel = [p for p in panel if p != "ridge_strong"]
    return tuple(panel) + LARGE_N_EXTRAS


def make_probe_n(name: str, seed: int = 42, X_all=None):
    """Probe factory for the ladder.

    Only "ridgecv" is special-cased to the plain regressor registry. "pls64"
    deliberately falls through to geo.make_probe instead of being routed to
    regressors.make_regressor("pls"): that bare path returns a raw
    PLSRegression(n_components=64), which raises whenever a readout has fewer
    than 64 features or rows -- and one of the real raw-input readouts
    (raw|moments) has only 10 features at 1-component / 30 at 3-components, so
    it would crash on the real run, not just in tests. geo.make_probe's
    "pls{k}" branch wraps PLS in a clamping wrapper
    (k = min(k, n_samples-1, n_features)) that handles this safely.
    """
    if name == "ridgecv":
        from .regressors import make_regressor
        return make_regressor("ridgecv", seed=seed)
    return geo.make_probe(name, seed=seed, X_all=X_all)


# The tap every prior measurement of this backbone used. Carried as a third arm
# so the report can separate "the regime changed" from "the readout was wrong".
HISTORICAL = geo.Readout("emb", "layer12", "mean", "none")

_R = geo.Readout

# Step 6's top 6 embedding and top 4 raw readouts per component count, by eval_a
# screen R2, plus HISTORICAL. Re-selected per rung on eval_a (spec sec. 3c).
CANDIDATES = {
    1: [
        _R("emb", "layer2", "segment4", "whiten"),
        _R("emb", "layer0", "segment4", "whiten"),
        _R("emb", "layer1", "segment4", "whiten"),
        _R("emb", "layer0", "first_last", "whiten"),
        _R("emb", "layer2", "first_last", "whiten"),
        _R("emb", "layer1", "mean", "whiten"),
        HISTORICAL,
        _R("raw", "raw", "-", "whiten"),
        _R("raw", "z", "-", "whiten"),
        _R("raw", "raw", "-", "standardize"),
        _R("raw", "raw", "-", "none"),
    ],
    2: [
        _R("emb", "layer2", "segment4", "none"),
        _R("emb", "layer3", "segment4", "standardize"),
        _R("emb", "layer3", "segment4", "none"),
        _R("emb", "layer1", "segment4", "none"),
        _R("emb", "layer1", "mean", "none"),
        _R("emb", "layer3", "mean", "none"),
        HISTORICAL,
        _R("raw", "raw", "-", "standardize"),
        _R("raw", "moments", "-", "whiten"),
        _R("raw", "raw", "-", "whiten"),
        _R("raw", "z", "-", "whiten"),
    ],
    3: [
        _R("emb", "layer1", "mean_std", "standardize"),
        _R("emb", "layer1", "segment4", "none"),
        _R("emb", "layer2", "mean_std", "standardize"),
        _R("emb", "layer1", "segment4", "standardize"),
        _R("emb", "layer0", "segment4", "standardize"),
        _R("emb", "layer0", "segment4", "none"),
        HISTORICAL,
        _R("raw", "raw", "-", "standardize"),
        _R("raw", "moments", "-", "whiten"),
        _R("raw", "moments", "-", "standardize"),
        _R("raw", "raw", "-", "whiten"),
    ],
}

# Step 1's large-n anchor (pooled 5-fold over all 4,716 => ~3,773 training
# spectra, RidgeCV). Plotted as a reference, never treated as a rung.
STEP1_ANCHOR = {
    1: {"n_eff": 3773, "emb": 0.628, "raw": 0.468},
    2: {"n_eff": 3773, "emb": 0.757, "raw": 0.831},
    3: {"n_eff": 3773, "emb": 0.793, "raw": 0.867},
}


# ── The pre-registered crossing estimator (spec sec. 2) ───────────────────
# Written and unit-tested before any real number is measured, so the
# definition of "the embedding won" cannot drift once the data is in.

WIN_BOOT_FRAC = 0.90


def _paired(emb, raw):
    """Element-wise paired delta over the shared draws, non-finite dropped.
    Paired -- never a difference of medians, which would discard the pairing
    that makes the two arms comparable draw for draw."""
    e = np.asarray(emb, dtype=float)
    r = np.asarray(raw, dtype=float)
    m = min(e.size, r.size)
    d = e[:m] - r[:m]
    fin = d[np.isfinite(d)]
    return fin, int(d.size - fin.size)


def estimate_crossover(per_n: dict, n_boot: int = 2000, seed: int = 42) -> dict:
    """
    per_n: {n: {"emb_per_draw": [...], "raw_per_draw": [...]}}

    A rung is WON when the paired median delta > 0 AND at least WIN_BOOT_FRAC of
    draw-level bootstrap resamples of that delta are > 0.

    The crossover is the smallest won rung whose every larger rung is also won.
    Requiring persistence is what separates a crossing from one noisy rung.
    """
    rng = np.random.default_rng(seed)
    rungs = {}
    for n in N_LADDER:
        cell = per_n.get(n)
        if cell is None:
            continue
        fin, dropped = _paired(cell["emb_per_draw"], cell["raw_per_draw"])
        if fin.size:
            med = float(np.median(fin))
            boots = np.array([np.median(rng.choice(fin, fin.size, replace=True))
                              for _ in range(n_boot)])
            frac = float(np.mean(boots > 0))
        else:
            med, frac = float("nan"), 0.0
        rungs[n] = {
            "delta_median": med,
            "boot_frac_positive": frac,
            "won": bool(med > 0 and frac >= WIN_BOOT_FRAC),
            "n_pairs": int(fin.size),
            "n_dropped": dropped,
        }

    ladder = [n for n in N_LADDER if n in rungs]
    # smallest won rung with every larger rung also won
    n_cross = None
    for i, n in enumerate(ladder):
        if all(rungs[m]["won"] for m in ladder[i:]):
            n_cross = n
            break

    finite = [(n, rungs[n]["delta_median"]) for n in ladder
              if np.isfinite(rungs[n]["delta_median"])]
    best_n, best_d = (max(finite, key=lambda t: t[1]) if finite
                      else (ladder[-1] if ladder else None, float("nan")))

    out = {"rungs": rungs, "crossed": n_cross is not None, "n_cross": n_cross,
           "n_cross_interp": None, "ci": None,
           "best_delta": best_d, "best_delta_n": best_n}
    if n_cross is None:
        return out

    idx = ladder.index(n_cross)
    if idx == 0:
        # already ahead at the first rung: nothing to interpolate from
        out["n_cross_interp"] = float(n_cross)
        out["ci"] = [float(n_cross), float(n_cross)]
        return out

    n_lo, n_hi = ladder[idx - 1], n_cross
    lo_fin, _ = _paired(per_n[n_lo]["emb_per_draw"], per_n[n_lo]["raw_per_draw"])
    hi_fin, _ = _paired(per_n[n_hi]["emb_per_draw"], per_n[n_hi]["raw_per_draw"])

    def _solve(d_lo, d_hi):
        """Log-linear in n: where does the delta line cross zero?"""
        if d_hi == d_lo:
            return float(n_hi)
        t = (0.0 - d_lo) / (d_hi - d_lo)
        t = min(max(t, 0.0), 1.0)
        return float(np.exp(np.log(n_lo) + t * (np.log(n_hi) - np.log(n_lo))))

    out["n_cross_interp"] = _solve(float(np.median(lo_fin)), float(np.median(hi_fin)))
    if lo_fin.size and hi_fin.size:
        vals = [_solve(float(np.median(rng.choice(lo_fin, lo_fin.size, replace=True))),
                       float(np.median(rng.choice(hi_fin, hi_fin.size, replace=True))))
                for _ in range(n_boot)]
        out["ci"] = [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]
    return out


# ── The ladder driver ─────────────────────────────────────────────────────

def _best_finite(cands, key):
    """Max over candidates with a finite score; None when none is finite.
    Guards the NaN-poisoning that `max` would otherwise allow: x > nan is
    False, so a NaN first element would never be displaced."""
    ok = [c for c in cands if np.isfinite(key(c))]
    return max(ok, key=key) if ok else None


def run_crossover(bank_path: str, output_dir: str, comp_counts=(1, 2, 3),
                  seed: int = 42, n_ladder=None,
                  sel_draws: int | None = None,
                  sel_top_emb: int | None = None,
                  sel_top_raw: int | None = None) -> dict:
    """
    Walk the n-ladder. At each rung: score every candidate readout on eval_a,
    pick the best per arm, then score just those on eval_b -- step 6's
    select-then-confirm discipline extended along n. The historical
    layer12/mean arm is carried at every rung so the report can separate "the
    regime changed" from "the readout was wrong".

    Reuses step 6's bank, split and paired-draw scorer; no GPU except TabPFN at
    the two lowest rungs.

    sel_draws: selection (eval_a) is a RANKING problem -- it only has to order
    candidate readouts -- while reporting (eval_b) is an ESTIMATION problem
    that needs the full draw budget. When given, the selection pass uses
    exactly `sel_draws` draws at every rung instead of `DRAWS_BY_N[n]`; the
    reporting pass always keeps `DRAWS_BY_N[n]`, unaffected. None (default)
    reproduces today's behaviour exactly.

    sel_top_emb / sel_top_raw: when given, only the first N embedding / raw
    candidates from `CANDIDATES[n_comp]` (already stored in descending step-6
    screen order, so a prefix is the top-k) are considered during selection.
    HISTORICAL is never subject to this slicing -- it is carried as its own
    single-candidate pool regardless, since it is a reported arm, not a
    candidate to be ranked. None (default) considers every candidate, matching
    today's behaviour exactly.
    """
    import json
    import os

    from . import readouts as ro
    from . import screen as scr
    from . import study6

    os.makedirs(output_dir, exist_ok=True)
    bank, input_raw, input_z, y, meta = ro.load_bank_cache(bank_path)
    n_total = len(y)
    split = scr.make_split(n_total, seed=seed)
    ladder = list(n_ladder if n_ladder is not None else N_LADDER)
    print(f"[step7] n={n_total} eval_a={len(split['eval_a'])} "
          f"eval_b={len(split['eval_b'])} pool={len(split['pool'])}", flush=True)

    # Selection draws (eval_a) and reporting draws (eval_b) use different seeds,
    # so the reported numbers are not scored on the draws that chose the readout.
    # sel_draws overrides only the selection count -- reporting always keeps
    # DRAWS_BY_N[n], since ranking (selection) needs far fewer draws than
    # estimation (reporting).
    draws_sel = {n: scr.draw_indices(
                    split["pool"], n,
                    sel_draws if sel_draws is not None else DRAWS_BY_N[n],
                    seed=seed)
                 for n in ladder}
    draws_rep = {n: scr.draw_indices(split["pool"], n, DRAWS_BY_N[n], seed=seed + 1000)
                 for n in ladder}

    results = {"meta": meta, "ladder": ladder,
               "draws_by_n": {str(n): DRAWS_BY_N[n] for n in ladder},
               "panels": {str(n): list(panel_for_n(n)) for n in ladder},
               "split_sizes": {k: len(v) for k, v in split.items()},
               # Execution-time configuration, so the report can state exactly
               # what ran without hardcoding it: the selection pass (eval_a)
               # can use a reduced draw count and a trimmed candidate pool
               # relative to the reporting pass (eval_b), which always keeps
               # the full DRAWS_BY_N and full CANDIDATES pool.
               "sel_config": {"sel_draws": sel_draws, "sel_top_emb": sel_top_emb,
                              "sel_top_raw": sel_top_raw},
               "per_comp": {}, "crossings": {}}

    for n_comp in comp_counts:
        comp_idx = list(range(len(study6.COMP_LADDER[n_comp])))
        cache = {}
        cands = CANDIDATES[n_comp]
        # Top-k slicing (a no-op when sel_top_emb/sel_top_raw are None, since
        # list[:None] is the whole list) applies only to the emb/raw pools --
        # HISTORICAL is excluded from the emb pool the same way it always was,
        # and its own pool is never sliced.
        emb_pool_full = [r for r in cands if r.family == "emb" and r != HISTORICAL]
        raw_pool_full = [r for r in cands if r.family == "raw"]
        emb_pool_sel = emb_pool_full[:sel_top_emb]
        raw_pool_sel = raw_pool_full[:sel_top_raw]
        # Only readouts that can actually be selected or reported need a
        # materialized matrix -- a narrowed selection pool need not pay for
        # normalizing candidates that can never be picked.
        needed = list(dict.fromkeys(emb_pool_sel + raw_pool_sel + [HISTORICAL]))
        # Normalizers are fitted on the whole unlabeled corpus and are
        # n-independent, so materialize once per readout and reuse across rungs.
        X_by_label = {}
        for rd in needed:
            X_by_label[rd.label()] = study6.materialize(
                rd, bank, input_raw, input_z, comp_idx, cache, seed=seed)
            print(f"[step7] materialized {n_comp}c {rd.label()} "
                  f"-> {X_by_label[rd.label()].shape}", flush=True)

        per_n, selected, sel_scores_by_n = {}, {}, {}
        for n in ladder:
            panel = panel_for_n(n)

            # --- selection pass, eval_a ---
            sel_scores = {}
            for rd in needed:
                s = scr.score_readout(X_by_label[rd.label()], y, split, panel, (n,),
                                      draws_sel, "eval_a", seed=seed,
                                      probe_factory=make_probe_n)
                best = _best_finite([(p, s[p][n]) for p in panel],
                                    key=lambda t: t[1]["r2_median"])
                sel_scores[rd.label()] = (float("nan") if best is None
                                          else best[1]["r2_median"])
            # Persisted verbatim so a later run can audit *why* a given
            # readout won a rung, rather than only which readout won --
            # the natural trail for diagnosing an artifact like step 7's own
            # (see report7's disclosure of the trimmed selection pool).
            sel_scores_by_n[str(n)] = dict(sel_scores)

            picks = {}
            for arm, pool_of in (("emb", emb_pool_sel),
                                 ("raw", raw_pool_sel),
                                 ("historical", [HISTORICAL])):
                if not pool_of:
                    raise ValueError(
                        f"no candidate readouts for arm {arm!r} at {n_comp}-comp; "
                        f"check CANDIDATES")
                win = _best_finite(pool_of, key=lambda r: sel_scores[r.label()])
                picks[arm] = win if win is not None else pool_of[0]
            selected[str(n)] = {a: r.label() for a, r in picks.items()}

            # --- reporting pass, eval_b, fresh draws ---
            cell = {}
            for arm, rd in picks.items():
                s = scr.score_readout(X_by_label[rd.label()], y, split, panel, (n,),
                                      draws_rep, "eval_b", seed=seed,
                                      probe_factory=make_probe_n)
                best = _best_finite([(p, s[p][n]) for p in panel],
                                    key=lambda t: t[1]["r2_median"])
                if best is None:
                    c = dict(s[panel[0]][n]); c["probe"] = None; c["all_failed"] = True
                else:
                    c = dict(best[1]); c["probe"] = best[0]
                c["readout"] = rd.label()
                cell[arm] = c
            per_n[str(n)] = cell
            print(f"[step7] {n_comp}c n={n:<5} "
                  f"emb {cell['emb']['r2_median']:+.4f} ({cell['emb']['readout']}) "
                  f"raw {cell['raw']['r2_median']:+.4f} ({cell['raw']['readout']}) "
                  f"hist {cell['historical']['r2_median']:+.4f}", flush=True)

        results["per_comp"][str(n_comp)] = {
            "per_n": per_n, "selected": selected, "sel_scores": sel_scores_by_n}
        results["crossings"][str(n_comp)] = estimate_crossover(
            {int(k): {"emb_per_draw": v["emb"]["r2_per_draw"],
                      "raw_per_draw": v["raw"]["r2_per_draw"]}
             for k, v in per_n.items()}, seed=seed)
        c = results["crossings"][str(n_comp)]
        print(f"[step7] {n_comp}c CROSSING: crossed={c['crossed']} "
              f"n_cross={c['n_cross']} interp={c['n_cross_interp']} ci={c['ci']}",
              flush=True)

    path = os.path.join(output_dir, "step7_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[step7] wrote {path}", flush=True)
    return results
