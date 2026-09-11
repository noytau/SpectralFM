"""
Orchestration for the four-step label-probe study. See plan.md.

run_step1: four-stage baseline (input/fe/extract_features/proj/transformer)
           x component ladder x probe panel x protocol panel. Produces the
           headline figure and REPORT.md.
run_step2: adds xgb/hgb to the probe panel on cached step-1 reps.
run_step3: DS hypothesis pass (H2 normalization, H3 comp budget -- read off
           step1's grid --, H8 pairwise/cross-component features, H9
           recovery ratios) plus H6's reframing as a documentation note.
run_step4: feature-vector expressivity sweep (comp count / cross-component
           combination / dimensionality) on cached reps.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ..checkpoint_loader import CheckpointLoader
from ..data_loader import load_labeled_data
from . import cache as cachemod
from . import features as feat
from . import taps
from .protocol import paired_delta, r2, run_label_group, run_legacy, run_primary
from .regressors import make_regressor


def _extract_all(model, raw_inputs: np.ndarray, device: str, batch_size: int,
                  pooling: str = "mean") -> dict:
    """
    raw_inputs: [N, K, L] raw (un-normalized) signal, K components.
    Returns {stage: [N, K, D_stage]} for stage in
      {'input_raw', 'input_z', 'fe', 'extract_features', 'proj', 'transformer'}.

    input_raw/input_z never touch the model (free); fe/extract_features/proj/
    transformer are extracted from ONE forward pass per batch on the
    z-scored signal (the distribution the backbone was trained on) via
    taps.extract_stage_reps.
    """
    n, k, L = raw_inputs.shape
    flat_raw = raw_inputs.reshape(n * k, L).astype(np.float32)
    flat_z = feat.normalize_like_fairseq(flat_raw)

    model_reps = taps.extract_stage_reps(model, flat_z, device=device,
                                          batch_size=batch_size, pooling=pooling)

    out = {
        "input_raw": flat_raw.reshape(n, k, L),
        "input_z": flat_z.reshape(n, k, L),
    }
    for stage in ("fe", "extract_features", "proj", "transformer"):
        d = model_reps[stage].shape[-1]
        out[stage] = model_reps[stage].reshape(n, k, d)
    return out


def _load_or_extract(model, checkpoint_path: str, labeled_data_dir: str,
                      cache_dir: str, max_samples: int, comps: tuple,
                      device: str, batch_size: int, pooling: str, seed: int):
    key = cachemod.cache_key(checkpoint_path, max_samples, comps, pooling, seed)
    hit = cachemod.load_reps(cache_dir, key)
    if hit is not None:
        print(f"[label_probe] cache hit: {key}")
        return hit
    print(f"[label_probe] cache miss: {key} -- extracting", flush=True)
    raw_inputs, y = load_labeled_data(labeled_data_dir, max_samples=max_samples,
                                       seed=seed, comps=comps)
    t0 = time.time()
    reps = _extract_all(model, raw_inputs, device=device, batch_size=batch_size,
                         pooling=pooling)
    print(f"[label_probe] extracted {raw_inputs.shape} in {time.time()-t0:.1f}s", flush=True)
    meta = {"checkpoint": checkpoint_path, "max_samples": max_samples,
            "comps": comps, "pooling": pooling, "seed": seed, "n": len(y)}
    cachemod.save_reps(cache_dir, key, reps, y, meta)
    return reps, y, meta


STAGE_ORDER = ("input_raw", "input_z", "fe", "extract_features", "proj", "transformer")
STAGE_LABELS = {
    "input_raw": "Raw input", "input_z": "Input (z-scored)",
    "fe": "Post-FE (pre-LN)", "extract_features": "Post-FE (post-LN)",
    "proj": "Post-Projection", "transformer": "Post-Transformer",
}


def run_step1(checkpoint_path: str, labeled_data_dir: str, output_dir: str,
              device: str = "cuda", batch_size: int = 64, max_samples: int = 5000,
              probe_panel=("ols", "ridge", "ridgecv"),
              comp_counts=(1, 2, 3, 7, 12), n_repeats: int = 5, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = os.path.join(output_dir, "_cache")

    print(f"[label_probe] loading checkpoint {checkpoint_path}", flush=True)
    model = CheckpointLoader.from_file(checkpoint_path)

    reps, y, meta = _load_or_extract(
        model, checkpoint_path, labeled_data_dir, cache_dir,
        max_samples=max_samples, comps=feat.UNIQUE_COMPS, device=device,
        batch_size=batch_size, pooling="mean", seed=seed)
    print(f"[label_probe] full corpus: n={len(y)}, comps={feat.UNIQUE_COMPS}")

    # legacy subsample: EXACT historical draw (max_samples=1000, seed=42, comps=(0,1,2))
    legacy_reps, legacy_y, _ = _load_or_extract(
        model, checkpoint_path, labeled_data_dir, cache_dir,
        max_samples=1000, comps=(0, 1, 2), device=device,
        batch_size=batch_size, pooling="mean", seed=42)

    results = {"meta": meta, "stages": {}}

    for n_comp in comp_counts:
        comp_tuple = feat.COMP_LADDER[n_comp]
        comp_idx = [feat.UNIQUE_COMPS.index(c) for c in comp_tuple]
        stage_block = {}
        for stage in STAGE_ORDER:
            X = feat.make_wide(reps[stage], comp_idx)
            probe_scores = {}
            best_probe, best_r2 = None, -np.inf
            for probe in probe_panel:
                pr = run_primary(lambda p=probe: make_regressor(p, seed=seed),
                                  X, y, n_repeats=n_repeats, seed0=seed)
                probe_scores[probe] = {
                    "r2_mean": pr.r2_mean, "r2_repeat_sd": pr.r2_repeat_sd,
                    "r2_bootstrap_sd": pr.r2_bootstrap_sd,
                }
                if pr.r2_mean > best_r2:
                    best_r2, best_probe = pr.r2_mean, probe
            lg = run_label_group(lambda: make_regressor(best_probe, seed=seed), X, y)
            stage_block[stage] = {
                "probes": probe_scores,
                "best_probe": best_probe,
                "best_r2": best_r2,
                "label_group_r2": lg.r2_mean,
            }
            print(f"[label_probe] n_comp={n_comp:2d} stage={stage:<17} "
                  f"best={best_probe}:{best_r2:+.4f}  label_group={lg.r2_mean:+.4f}",
                  flush=True)
        results["stages"][n_comp] = stage_block

    # legacy row: comps (0,1,2) at 1/2/3-comp, ridgecv only, input_raw + transformer
    legacy_block = {}
    for n_comp in (1, 2, 3):
        comp_idx = list(range(n_comp))  # legacy_reps already only has comps 0,1,2
        row = {}
        for stage in ("input_raw", "input_z", "transformer"):
            X = feat.make_wide(legacy_reps[stage], comp_idx)
            lr = run_legacy(lambda: make_regressor("ridgecv", seed=seed), X, legacy_y)
            row[stage] = lr.r2_mean
        legacy_block[n_comp] = row
        print(f"[label_probe] LEGACY n_comp={n_comp} "
              f"input_raw={row['input_raw']:+.4f} input_z={row['input_z']:+.4f} "
              f"transformer={row['transformer']:+.4f}", flush=True)
    results["legacy"] = legacy_block

    # ceiling: best input R2 at the largest available comp count, either
    # normalization -- everything else is reported as a recovery ratio
    # against this. (Uses max(comp_counts), not a hardcoded 12, so smaller
    # smoke runs -- e.g. comp_counts=(1,2) -- don't KeyError.)
    ceiling_n = max(results["stages"])
    ceiling = max(results["stages"][ceiling_n]["input_raw"]["best_r2"],
                  results["stages"][ceiling_n]["input_z"]["best_r2"])
    results["input_ceiling"] = ceiling
    for n_comp, block in results["stages"].items():
        for stage, s in block.items():
            s["recovery_ratio"] = s["best_r2"] / ceiling if ceiling > 0 else float("nan")

    with open(os.path.join(output_dir, "step1_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _write_step1_report(results, output_dir)

    from . import plots
    cells = collect_true_vs_pred_grid(reps, y, results)
    plots.plot_true_vs_pred_grid(
        cells, os.path.join(output_dir, "label_reg_true_vs_pred_grid.png"))

    return results, reps, y


def collect_true_vs_pred_grid(reps: dict, y, results: dict,
                                comp_counts=None, stages=None, n_repeats: int = 1,
                                seed: int = 42) -> dict:
    """
    Gather one out-of-fold true-vs-pred array per (n_comp, stage) cell, using
    the SAME best-of-panel probe run_step1 already picked for that cell (so
    the grid is consistent with the numbers in step1_REPORT.md, not a
    re-fit with a different probe). n_repeats=1 by default -- this is for
    visualization, not a new R2 estimate; the reported R2/probe choice comes
    from `results`, this function only supplies the scatter data.

    Returns {(n_comp, stage): {"y_true", "y_pred", "r2", "pearson_r", "mae"}}.
    """
    from scipy.stats import pearsonr

    if comp_counts is None:
        comp_counts = sorted(results["stages"])
    if stages is None:
        stages = STAGE_ORDER

    cells = {}
    for n_comp in comp_counts:
        comp_tuple = feat.COMP_LADDER[n_comp]
        comp_idx = [feat.UNIQUE_COMPS.index(c) for c in comp_tuple]
        for stage in stages:
            probe = results["stages"][n_comp][stage]["best_probe"]
            X = feat.make_wide(reps[stage], comp_idx)
            pr = run_primary(lambda p=probe: make_regressor(p, seed=seed), X, y,
                              n_repeats=n_repeats, seed0=seed)
            y_pred = pr.oof_predictions
            r2_val = r2(y, y_pred)
            pearson_r, _ = pearsonr(y, y_pred)
            mae = float(np.mean(np.abs(y - y_pred)))
            cells[(n_comp, stage)] = {
                "y_true": y, "y_pred": y_pred, "r2": r2_val,
                "pearson_r": float(pearson_r), "mae": mae, "probe": probe,
            }
            print(f"[label_probe] scatter-grid cell n_comp={n_comp} stage={stage} "
                  f"probe={probe} R2={r2_val:+.4f}", flush=True)
    return cells


def _write_step1_report(results: dict, output_dir: str):
    lines = ["# Step 1 — Four-Stage Baseline\n"]
    lines.append(f"Checkpoint: `{results['meta']['checkpoint']}`  \n"
                 f"n={results['meta']['n']} spectra, "
                 f"comps={results['meta']['comps']}\n")
    lines.append(f"**Input ceiling (best linear, 12-comp): "
                 f"R²={results['input_ceiling']:.4f}**\n")

    lines.append("\n## Stage table (best-of-panel R², recovery ratio vs ceiling)\n")
    lines.append("| n-comp | " + " | ".join(STAGE_LABELS[s] for s in STAGE_ORDER) + " |")
    lines.append("|---|" + "---|" * len(STAGE_ORDER))
    for n_comp, block in results["stages"].items():
        row = [f"{n_comp}"]
        for stage in STAGE_ORDER:
            s = block[stage]
            row.append(f"{s['best_r2']:.3f} ({s['recovery_ratio']*100:.0f}%)")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Legacy row (n=1000, seed=42, KFold(5,shuffle=False), RidgeCV)\n")
    lines.append("| n-comp | raw input | z-scored input | transformer |")
    lines.append("|---|---|---|---|")
    for n_comp, row in results["legacy"].items():
        lines.append(f"| {n_comp} | {row['input_raw']:.4f} | {row['input_z']:.4f} "
                     f"| {row['transformer']:.4f} |")
    lines.append("\nHistorical recorded values: 0.3772 / 0.7060 / 0.7568 "
                 "(z-scored input, 1/2/3-comp).\n")

    lines.append("\n## Verdict on the T6 headline claim\n")
    t_1c = results["stages"][1]["transformer"]["best_r2"]
    i_1c_raw = results["stages"][1]["input_raw"]["best_r2"]
    i_1c_z = results["stages"][1]["input_z"]["best_r2"]
    beats_z = t_1c > i_1c_z
    beats_raw = t_1c > i_1c_raw
    lines.append(f"- 1-comp transformer R²={t_1c:.4f} vs input (z-scored, best "
                 f"probe)={i_1c_z:.4f} vs input (raw, best probe)={i_1c_raw:.4f}\n"
                 f"- Embedding beats z-scored input: **{beats_z}**\n"
                 f"- Embedding beats raw input: **{beats_raw}**\n")

    with open(os.path.join(output_dir, "step1_REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[label_probe] wrote {output_dir}/step1_REPORT.md")


# ─────────────────────────────────────────────────────────────────────────
# Step 2 — XGBoost regressor
# ─────────────────────────────────────────────────────────────────────────

def run_step2(step1_output_dir: str, n_repeats: int = 5, seed: int = 42,
              comp_counts=(1, 2, 3, 7, 12)) -> dict:
    """
    Reuses step 1's cached reps (no GPU work). Adds xgb (falls back to hgb)
    to the probe panel and reports paired Delta(xgb-ridge) per stage per
    comp config.

    Caveat baked in from plan.md: at 3 comps the embedding vector is
    2,304-dim, at 12 comps 9,216-dim, against n~4,716 -- p>>n is XGBoost's
    weakest regime and linear's strongest (raw input already reaches ~0.98
    there). XGBoost's real target is the embedding stages, far from ceiling.
    """
    import glob
    cache_dir = os.path.join(step1_output_dir, "_cache")
    cache_files = glob.glob(os.path.join(cache_dir, "reps_*.npz"))
    if not cache_files:
        raise FileNotFoundError(f"no step-1 cache found under {cache_dir}; run step 1 first")
    # pick the largest cache file (the full-corpus one, not the legacy subsample)
    cache_files.sort(key=os.path.getsize, reverse=True)
    key = os.path.basename(cache_files[0])[len("reps_"):-len(".npz")]
    reps, y, meta = cachemod.load_reps(cache_dir, key)
    print(f"[label_probe:step2] loaded cache n={len(y)} comps={meta['comps']}")

    results = {"meta": meta, "stages": {}}
    for n_comp in comp_counts:
        comp_tuple = feat.COMP_LADDER[n_comp]
        comp_idx = [feat.UNIQUE_COMPS.index(c) for c in comp_tuple]
        stage_block = {}
        for stage in STAGE_ORDER:
            X = feat.make_wide(reps[stage], comp_idx)
            pd = paired_delta(
                lambda: make_regressor("xgb", seed=seed),
                lambda: make_regressor("ridge", seed=seed),
                X, y, n_repeats=n_repeats, seed0=seed)
            stage_block[stage] = pd
            print(f"[label_probe:step2] n_comp={n_comp:2d} stage={stage:<17} "
                  f"xgb={pd['r2_a_mean']:+.4f} ridge={pd['r2_b_mean']:+.4f} "
                  f"delta={pd['delta_mean']:+.4f}+-{pd['delta_sd']:.4f} "
                  f"tie={pd['is_tie']}", flush=True)
        results["stages"][n_comp] = stage_block

    with open(os.path.join(step1_output_dir, "step2_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    _write_step2_report(results, step1_output_dir)
    return results


def _write_step2_report(results: dict, output_dir: str):
    lines = ["# Step 2 — XGBoost vs Ridge\n"]
    lines.append("Paired delta (xgb − ridge), identical folds, 5 repeats. "
                 "`tie` = |delta| < 2×delta_sd.\n")
    lines.append("| n-comp | " + " | ".join(STAGE_LABELS[s] for s in STAGE_ORDER) + " |")
    lines.append("|---|" + "---|" * len(STAGE_ORDER))
    for n_comp, block in results["stages"].items():
        row = [f"{n_comp}"]
        for stage in STAGE_ORDER:
            pd = block[stage]
            tie = " (tie)" if pd["is_tie"] else ""
            row.append(f"{pd['delta_mean']:+.3f}{tie}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("\nPositive = XGBoost beats Ridge at that stage/config.\n")
    with open(os.path.join(output_dir, "step2_REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[label_probe] wrote {output_dir}/step2_REPORT.md")


# ─────────────────────────────────────────────────────────────────────────
# Step 3 — DS investigation (H1-H9, see plan.md)
# ─────────────────────────────────────────────────────────────────────────

def run_step3(step1_output_dir: str, seed: int = 42) -> dict:
    """
    Hypothesis pass on cached step-1 reps. Implements the hypotheses that
    are cheap given the cache (H2, H3, H8, H9 directly; H1/H5/H6/H7 are
    read off step-1's grid + the split-design evidence already in plan.md,
    since they were answered by the pre-implementation diagnostics rather
    than needing new code). H4 (pooling) and the deep pooling-head method
    require re-extraction at other poolings -- left to step 4 / a follow-up,
    flagged explicitly in the report rather than silently skipped.
    """
    import glob
    cache_dir = os.path.join(step1_output_dir, "_cache")
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "reps_*.npz")),
                         key=os.path.getsize, reverse=True)
    key = os.path.basename(cache_files[0])[len("reps_"):-len(".npz")]
    reps, y, meta = cachemod.load_reps(cache_dir, key)

    with open(os.path.join(step1_output_dir, "step1_results.json")) as f:
        step1 = json.load(f)

    hyp = {}

    # H2 -- normalization cost, read directly off step1's input_raw vs input_z
    h2 = {}
    for n_comp_str, block in step1["stages"].items():
        raw_r2 = block["input_raw"]["best_r2"]
        z_r2 = block["input_z"]["best_r2"]
        h2[n_comp_str] = {"raw": raw_r2, "z_scored": z_r2, "cost": raw_r2 - z_r2}
    hyp["H2_normalization_cost"] = h2

    # H3 -- component budget, read off step1's stage table directly
    hyp["H3_component_budget"] = {
        n_comp_str: {s: step1["stages"][n_comp_str][s]["best_r2"] for s in STAGE_ORDER}
        for n_comp_str in step1["stages"]
    }

    # H8 -- explicit cross-component / pairwise features vs concat, at 3-comp
    comp_idx3 = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[3]]
    h8 = {}
    for stage in ("input_raw", "transformer"):
        X_concat = feat.make_wide(reps[stage], comp_idx3)
        X_pair = feat.make_pairwise_features(reps[stage], comp_idx3)
        r_concat = run_primary(lambda: make_regressor("ridgecv", seed=seed), X_concat, y, n_repeats=3)
        r_pair = run_primary(lambda: make_regressor("ridgecv", seed=seed), X_pair, y, n_repeats=3)
        h8[stage] = {"concat_r2": r_concat.r2_mean, "pairwise_r2": r_pair.r2_mean,
                     "pairwise_dim": X_pair.shape[1], "concat_dim": X_concat.shape[1]}
        print(f"[label_probe:step3] H8 stage={stage} concat={r_concat.r2_mean:+.4f} "
              f"pairwise={r_pair.r2_mean:+.4f}", flush=True)
    hyp["H8_pairwise_features"] = h8

    # H9 -- recovery ratio vs ceiling, per stage, at the best comp count
    ceiling = step1["input_ceiling"]
    best_n = max(step1["stages"], key=lambda n: step1["stages"][n]["transformer"]["best_r2"])
    hyp["H9_recovery_ratios"] = {
        s: step1["stages"][best_n][s]["best_r2"] / ceiling for s in STAGE_ORDER
    }
    hyp["H9_ceiling"] = ceiling
    hyp["H9_best_n_comp"] = best_n

    with open(os.path.join(step1_output_dir, "step3_results.json"), "w") as f:
        json.dump(hyp, f, indent=2, default=str)
    _write_step3_report(hyp, step1_output_dir)
    return hyp


def _write_step3_report(hyp: dict, output_dir: str):
    lines = ["# Step 3 — DS Investigation\n"]
    lines.append("## H2 — normalization cost (raw − z-scored best-linear R²)\n")
    lines.append("| n-comp | raw | z-scored | cost |\n|---|---|---|---|")
    for n_comp, v in hyp["H2_normalization_cost"].items():
        lines.append(f"| {n_comp} | {v['raw']:.4f} | {v['z_scored']:.4f} | {v['cost']:+.4f} |")

    lines.append("\n## H3 — component budget (best R² per stage per n-comp)\n")
    lines.append("| n-comp | " + " | ".join(STAGE_LABELS[s] for s in STAGE_ORDER) + " |")
    lines.append("|---|" + "---|" * len(STAGE_ORDER))
    for n_comp, row in hyp["H3_component_budget"].items():
        lines.append("| " + n_comp + " | " + " | ".join(f"{row[s]:.3f}" for s in STAGE_ORDER) + " |")

    lines.append("\n## H8 — explicit cross-component features vs plain concat\n")
    lines.append("| stage | concat R² (dim) | pairwise-feature R² (dim) |\n|---|---|---|")
    for stage, v in hyp["H8_pairwise_features"].items():
        lines.append(f"| {stage} | {v['concat_r2']:.4f} ({v['concat_dim']}) "
                     f"| {v['pairwise_r2']:.4f} ({v['pairwise_dim']}) |")

    lines.append(f"\n## H9 — recovery ratio vs input ceiling (R²={hyp['H9_ceiling']:.4f}, "
                 f"at {hyp['H9_best_n_comp']}-comp)\n")
    lines.append("| stage | R² | recovery ratio |\n|---|---|---|")
    for s in STAGE_ORDER:
        r = hyp["H9_recovery_ratios"][s]
        lines.append(f"| {STAGE_LABELS[s]} | {r*hyp['H9_ceiling']:.4f} | {r*100:.1f}% |")

    lines.append("\n## H1, H5, H6, H7 — answered by the pre-implementation "
                 "diagnostics (see plan.md), not re-run here\n")
    lines.append("- **H1** (sample-size x probe interaction): OLS degrades "
                 "monotonically with n while RidgeCV improves (measured "
                 "0.735->0.665 vs 0.706->0.756, n=500->4716, 2-comp input). "
                 "See plan.md's learning-curve table.\n"
                 "- **H5** (168 discrete levels): ruled out as a leakage "
                 "source (`label_group` protocol in step 1 matches `primary` "
                 "to 3 decimals).\n"
                 "- **H6** (comparability): split noise is 0.0005, three "
                 "orders below the probe/n/normalization systematics -- the "
                 "job is reporting the surface, which step 1's grid does.\n"
                 "- **H7** (non-linearity): see step 2's xgb-vs-ridge report.\n")
    lines.append("\n## Not run this pass\n")
    lines.append("- **H4** (pooling scheme) and the trained attention-pooling "
                 "head require re-extracting at other poolings -- deferred to "
                 "step 4 / a follow-up session; not silently skipped.\n")

    with open(os.path.join(output_dir, "step3_REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[label_probe] wrote {output_dir}/step3_REPORT.md")


# ─────────────────────────────────────────────────────────────────────────
# Step 4 — feature-vector expressivity
# ─────────────────────────────────────────────────────────────────────────

def run_step4(step1_output_dir: str, seed: int = 42) -> dict:
    """
    Staged sweep (fix two axes, vary one) over component count, cross-
    component combination, and dimensionality, on cached step-1 reps.

    Per plan.md: mean-over-components and the naive long layout are
    EXCLUDED as defaults (both collapse to R2~0.006 -- components are not
    exchangeable, this reproduces the historical bf94422 bug). One of them
    (mean-over-components) is included here as a deliberate negative
    control so the report shows the failure mode rather than asserting it.
    """
    import glob
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression

    cache_dir = os.path.join(step1_output_dir, "_cache")
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "reps_*.npz")),
                         key=os.path.getsize, reverse=True)
    key = os.path.basename(cache_files[0])[len("reps_"):-len(".npz")]
    reps, y, meta = cachemod.load_reps(cache_dir, key)

    default_stage = "transformer"
    default_n_comp = 3

    def score(X, n_repeats=3):
        pr = run_primary(lambda: make_regressor("ridgecv", seed=seed), X, y,
                          n_repeats=n_repeats)
        return pr.r2_mean

    results = {"meta": meta}

    # Axis 1: component count (wide concat, transformer stage)
    axis1 = {}
    for n_comp in (1, 2, 3, 7, 12):
        comp_idx = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
        X = feat.make_wide(reps[default_stage], comp_idx)
        axis1[n_comp] = score(X)
        print(f"[label_probe:step4] axis1 n_comp={n_comp} R2={axis1[n_comp]:+.4f}", flush=True)
    results["axis1_component_count"] = axis1
    best_n_comp = max(axis1, key=axis1.get)
    comp_idx_best = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[best_n_comp]]

    # Axis 2: cross-component combination (at best_n_comp)
    axis2 = {}
    X_concat = feat.make_wide(reps[default_stage], comp_idx_best)
    axis2["concat"] = score(X_concat)

    mean_over_comps = reps[default_stage][:, comp_idx_best, :].mean(axis=1)
    axis2["mean_over_comps (negative control)"] = score(mean_over_comps)

    per_comp_mean = reps[default_stage][:, comp_idx_best, :].mean(axis=-1)
    per_comp_std = reps[default_stage][:, comp_idx_best, :].std(axis=-1)
    X_meanstd = np.concatenate([X_concat, per_comp_mean, per_comp_std], axis=-1)
    axis2["concat(mean,std)"] = score(X_meanstd)

    X_pairwise = feat.make_pairwise_features(reps[default_stage], comp_idx_best)
    axis2["pairwise_features_only"] = score(X_pairwise)

    for name, v in axis2.items():
        print(f"[label_probe:step4] axis2 combo={name} R2={v:+.4f}", flush=True)
    results["axis2_combination"] = axis2
    best_combo_X = {"concat": X_concat, "concat(mean,std)": X_meanstd,
                     "pairwise_features_only": X_pairwise}[
        max({k: v for k, v in axis2.items() if "negative control" not in k}, key=axis2.get)]

    # Axis 3: dimensionality reduction on the winning combo
    axis3 = {}
    axis3["full"] = score(best_combo_X)
    if best_combo_X.shape[1] > 256:
        Xp = PCA(n_components=256, random_state=seed).fit_transform(best_combo_X)
        axis3["pca_256"] = score(Xp)
    if best_combo_X.shape[1] > 64:
        pls = PLSRegression(n_components=64)
        Xpls = pls.fit_transform(best_combo_X, y)[0]
        axis3["pls_64"] = score(Xpls)
    for name, v in axis3.items():
        print(f"[label_probe:step4] axis3 dim={name} R2={v:+.4f}", flush=True)
    results["axis3_dimensionality"] = axis3

    # Comparison vs the historical (0,1)-concat default
    comp_idx01 = [feat.UNIQUE_COMPS.index(c) for c in (0, 1)]
    historical_X = feat.make_wide(reps[default_stage], comp_idx01)
    results["historical_default_r2"] = score(historical_X)
    results["winner_r2"] = max(axis3.values())
    results["winner_delta_vs_historical"] = results["winner_r2"] - results["historical_default_r2"]

    with open(os.path.join(step1_output_dir, "step4_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    _write_step4_report(results, step1_output_dir)
    return results


def _write_step4_report(results: dict, output_dir: str):
    lines = ["# Step 4 — Feature-Vector Expressivity\n"]
    lines.append("Staged sweep on cached `transformer`-stage reps: fix two "
                 "axes at defaults, vary the third, keep the winner.\n")

    lines.append("\n## Axis 1 — component count\n")
    lines.append("| n-comp | R² |\n|---|---|")
    for n, v in results["axis1_component_count"].items():
        lines.append(f"| {n} | {v:.4f} |")

    lines.append("\n## Axis 2 — cross-component combination\n")
    lines.append("| combination | R² |\n|---|---|")
    for name, v in results["axis2_combination"].items():
        lines.append(f"| {name} | {v:.4f} |")
    lines.append("\n`mean_over_comps` is included as a deliberate negative "
                 "control (plan.md: exchangeability is false, this "
                 "formulation is expected to collapse).\n")

    lines.append("\n## Axis 3 — dimensionality\n")
    lines.append("| reduction | R² |\n|---|---|")
    for name, v in results["axis3_dimensionality"].items():
        lines.append(f"| {name} | {v:.4f} |")

    lines.append(f"\n## Verdict\n")
    best_n = max(results["axis1_component_count"], key=results["axis1_component_count"].get)
    concat_at_best_n = results["axis1_component_count"][best_n]
    matched_delta = results["winner_r2"] - concat_at_best_n
    lines.append(f"- Historical `(0,1)`-concat default (2-comp): "
                 f"R²={results['historical_default_r2']:.4f}\n"
                 f"- Best found this sweep ({best_n}-comp): R²={results['winner_r2']:.4f}\n"
                 f"- Raw delta: {results['winner_delta_vs_historical']:+.4f} — but this "
                 f"**conflates two changes** and should not be quoted alone: it compares "
                 f"{best_n}-comp against the 2-comp default, so most of it is the "
                 f"component count, not the feature-vector choice.\n"
                 f"- **Matched-component delta (the feature-vector effect alone): "
                 f"{matched_delta:+.4f}** ({best_n}-comp best vs {best_n}-comp plain "
                 f"concat) — this is the number attributable to the combination/"
                 f"dimensionality choice.\n")
    lines.append("\n**Dimensionality note:** supervised reduction (PLS) *beats* the full "
                 "vector while unsupervised reduction (PCA) loses heavily — see axis 3. "
                 "The label direction is not in the top principal components, so any "
                 "method built on unsupervised PCA features inherits that loss.\n")
    lines.append("\n**Component-count caveat:** these axes were swept at the *best* "
                 "component count from axis 1, which is 12 — but a linear probe already "
                 "reaches ~0.99 at 12-comp, so there is little headroom there. For the "
                 "1-3 component deployment regime see `step4_deployment_axes.md`.\n")

    with open(os.path.join(output_dir, "step4_REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[label_probe] wrote {output_dir}/step4_REPORT.md")


# ─────────────────────────────────────────────────────────────────────────
# Step 5 — Few-shot / label-efficiency regime
# ─────────────────────────────────────────────────────────────────────────

FEWSHOT_N_TRAIN = (10, 15, 20, 30, 50, 100)

# TabPFN costs ~2.4s per draw on GPU (measured) vs microseconds for the
# linear/kNN probes. The full grid at 100 draws would be ~43k calls ~= 29
# GPU-hours, so TabPFN gets its own reduced budget. The trade is fewer draws
# -> wider uncertainty on its medians than the rest of the panel; the report
# states the per-probe draw count so this is never hidden.
TABPFN_BUDGET = {
    "probes": ("tabpfn_pca10", "tabpfn_pca20"),   # pca50 dropped: 50 features on
                                                   # 20 rows is past TabPFN's regime
    "comp_counts": (1, 2, 3),   # deployment ladder only -- 12-comp is not
                                # decision-relevant and has no headroom (linear
                                # already reaches 0.989 there)
    "n_train": (10, 20, 50),    # brackets the client's 20-label budget
    "n_draws": 20,
}


def run_step5(step1_output_dir: str, n_draws: int = 100, n_eval: int = 1000,
              comp_counts=(3, 12), seed: int = 42, transductive: bool = True,
              also_inductive: bool = True) -> dict:
    """
    Few-shot label-efficiency study. Motivated by the deployment constraint
    of ~20 labeled samples; see plan.md "Step 5" for why steps 1-4 cannot
    answer this and why the protocol differs (repeated subsampling with a
    distribution, low-capacity probes, unlabeled-PCA reported separately
    from the strictly inductive variant).

    Reuses step 1's cached representations -- no GPU, no re-extraction.
    """
    import glob
    from sklearn.decomposition import PCA

    from .protocol import run_fewshot
    from .regressors import FEWSHOT_PANEL, make_fewshot_regressor, tabpfn_available

    panel = list(FEWSHOT_PANEL)
    if not tabpfn_available():
        dropped = [p for p in panel if p.startswith("tabpfn")]
        panel = [p for p in panel if not p.startswith("tabpfn")]
        if dropped:
            print(f"[label_probe:step5] tabpfn not installed -- skipping {dropped}",
                  flush=True)

    cache_dir = os.path.join(step1_output_dir, "_cache")
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "reps_*.npz")),
                         key=os.path.getsize, reverse=True)
    if not cache_files:
        raise FileNotFoundError(f"no step-1 cache under {cache_dir}; run step 1 first")
    key = os.path.basename(cache_files[0])[len("reps_"):-len(".npz")]
    reps, y, meta = cachemod.load_reps(cache_dir, key)
    n_total = len(y)
    print(f"[label_probe:step5] loaded cache n={n_total} comps={meta['comps']}", flush=True)

    # Fixed held-out evaluation set, disjoint from every training draw.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    eval_idx, pool_idx = perm[:n_eval], perm[n_eval:]
    print(f"[label_probe:step5] eval set={len(eval_idx)}, train pool={len(pool_idx)}",
          flush=True)

    step1 = None
    p1 = os.path.join(step1_output_dir, "step1_results.json")
    if os.path.exists(p1):
        with open(p1) as f:
            step1 = json.load(f)

    modes = []
    if transductive:
        modes.append("transductive")     # PCA fitted on ALL spectra (unlabeled)
    if also_inductive:
        modes.append("inductive")        # PCA fitted inside the tiny draw only

    results = {"meta": meta, "n_eval": n_eval, "n_draws": n_draws,
               "n_train_values": list(FEWSHOT_N_TRAIN), "modes": modes, "cells": {}}

    for n_comp in comp_counts:
        comp_idx = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
        for stage in STAGE_ORDER:
            X = feat.make_wide(reps[stage], comp_idx)
            # unlabeled PCA basis: fitted on the FULL set of representations,
            # no labels involved. Legitimate (deployment has unlabeled
            # spectra) but a different assumption from inductive -- reported
            # separately, never merged.
            pca_basis = PCA(n_components=min(10, X.shape[1], n_total),
                             random_state=seed).fit(X)
            for mode in modes:
                basis = pca_basis if mode == "transductive" else None
                for probe in panel:
                    is_tabpfn = probe.startswith("tabpfn")
                    if mode == "inductive" and not (probe.startswith("pca") or is_tabpfn):
                        continue   # only the pca-based probes differ between modes
                    if is_tabpfn:
                        # reduced budget -- see TABPFN_BUDGET
                        if probe not in TABPFN_BUDGET["probes"]:
                            continue
                        if n_comp not in TABPFN_BUDGET["comp_counts"]:
                            continue
                        if mode == "inductive":
                            continue   # transductive only, to stay in budget
                        probe_n_train = TABPFN_BUDGET["n_train"]
                        probe_draws = TABPFN_BUDGET["n_draws"]
                    else:
                        probe_n_train = FEWSHOT_N_TRAIN
                        probe_draws = n_draws
                    for n_train in probe_n_train:
                        r = run_fewshot(
                            lambda p=probe, b=basis: make_fewshot_regressor(
                                p, seed=seed, pca_basis=b),
                            X, y, n_train=n_train, eval_idx=eval_idx,
                            pool_idx=pool_idx, n_draws=probe_draws, seed0=seed,
                        )
                        results["cells"][f"{n_comp}|{stage}|{mode}|{probe}|{n_train}"] = r
                    print(f"[label_probe:step5] n_comp={n_comp:2d} stage={stage:<17} "
                          f"{mode[:5]} {probe:<12} "
                          f"n=20 median R2="
                          f"{results['cells'][f'{n_comp}|{stage}|{mode}|{probe}|20']['r2_median']:+.4f} "
                          f"frac>0="
                          f"{results['cells'][f'{n_comp}|{stage}|{mode}|{probe}|20']['frac_positive_r2']:.2f}",
                          flush=True)

    # PCA-ceiling diagnostic. A weak TabPFN/PCA result is ambiguous between
    # "the probe is bad" and "PCA (which maximizes variance, not
    # label-relevance) discarded the label direction before the probe saw
    # it". Fitting a LARGE-n linear probe on the same PCA-k features bounds
    # what any probe could possibly achieve on them, which disambiguates.
    print("[label_probe:step5] PCA-ceiling diagnostic (large-n linear probe "
          "on the same PCA-k features)", flush=True)
    pca_ceiling = {}
    for n_comp in comp_counts:
        comp_idx = [feat.UNIQUE_COMPS.index(c) for c in feat.COMP_LADDER[n_comp]]
        for stage in STAGE_ORDER:
            X = feat.make_wide(reps[stage], comp_idx)
            for k in (10, 20, 50):
                if k >= X.shape[1]:
                    continue
                Xk = PCA(n_components=k, random_state=seed).fit_transform(X)
                pr = run_primary(lambda: make_regressor("ridgecv", seed=seed),
                                  Xk, y, n_repeats=1, seed0=seed)
                pca_ceiling[f"{n_comp}|{stage}|{k}"] = pr.r2_mean
            print(f"[label_probe:step5] pca-ceiling n_comp={n_comp} stage={stage:<17} "
                  + "  ".join(f"k={k}:{pca_ceiling.get(f'{n_comp}|{stage}|{k}', float('nan')):+.4f}"
                              for k in (10, 20, 50)), flush=True)
    results["pca_ceiling_r2"] = pca_ceiling

    if step1 is not None:
        results["asymptotic_r2"] = {
            f"{n_comp}|{stage}": step1["stages"][str(n_comp)][stage]["best_r2"]
            for n_comp in comp_counts for stage in STAGE_ORDER
            if str(n_comp) in step1["stages"]
        }

    with open(os.path.join(step1_output_dir, "step5_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    _write_step5_report(results, step1_output_dir, comp_counts)
    return results


def _best_fewshot(results: dict, n_comp: int, stage: str, n_train: int,
                   mode: str = "transductive"):
    """Best probe (by median R2) for one cell, and its stats."""
    best, best_r = None, -np.inf
    for k, v in results["cells"].items():
        c, s, m, probe, n = k.split("|")
        if (int(c), s, m, int(n)) != (n_comp, stage, mode, n_train):
            continue
        if v["r2_median"] > best_r:
            best_r, best = v["r2_median"], (probe, v)
    return best


def _write_step5_report(results: dict, output_dir: str, comp_counts):
    lines = ["# Step 5 — Few-shot / label-efficiency regime\n"]
    lines.append(f"Held-out eval set: {results['n_eval']} spectra (fixed, disjoint "
                 f"from every training draw). {results['n_draws']} random draws "
                 f"per (n_train, probe, stage).\n")
    lines.append("**Note on `draws`:** TabPFN costs ~2.4s/draw on GPU vs "
                 "microseconds for the linear/kNN probes, so it runs on a "
                 "reduced budget (20 draws, transductive only, 1/2/3-comp, "
                 "n_train ∈ {10,20,50}); the full grid would be ~29 GPU-hours. "
                 "Its medians therefore carry wider uncertainty than the rest "
                 "of the panel — compare with that in mind.\n")
    lines.append("**Why this is a separate study:** steps 1-4 measure how much "
                 "label information a probe recovers given ~4,716 labels. This "
                 "measures what a deployment with ~20 labels can actually do. "
                 "They are different questions and need not agree — see plan.md.\n")

    for n_comp in comp_counts:
        lines.append(f"\n## {n_comp}-component — median held-out R² at n_train=20 "
                     f"(best probe per stage, transductive PCA)\n")
        lines.append("| stage | best probe | median R² | IQR | frac draws R²>0 | draws | asymptotic R² (step 1) |")
        lines.append("|---|---|---|---|---|---|---|")
        for stage in STAGE_ORDER:
            got = _best_fewshot(results, n_comp, stage, 20, "transductive")
            if got is None:
                continue
            probe, v = got
            asym = results.get("asymptotic_r2", {}).get(f"{n_comp}|{stage}")
            asym_s = f"{asym:.3f}" if asym is not None else "—"
            lines.append(f"| {STAGE_LABELS[stage]} | `{probe}` | {v['r2_median']:+.3f} "
                         f"| [{v['r2_p25']:+.3f}, {v['r2_p75']:+.3f}] "
                         f"| {v['frac_positive_r2']:.2f} | {v['n_draws']} | {asym_s} |")

        lines.append(f"\n### {n_comp}-component — label-efficiency curve "
                     f"(median R², transductive, best probe per cell)\n")
        header = " | ".join(str(n) for n in results["n_train_values"])
        lines.append(f"| stage | {header} |")
        lines.append("|---|" + "---|" * len(results["n_train_values"]))
        for stage in STAGE_ORDER:
            cells = []
            for n_train in results["n_train_values"]:
                got = _best_fewshot(results, n_comp, stage, n_train, "transductive")
                cells.append(f"{got[1]['r2_median']:+.3f}" if got else "—")
            lines.append(f"| {STAGE_LABELS[stage]} | " + " | ".join(cells) + " |")

    if "inductive" in results["modes"]:
        lines.append("\n## Transductive vs inductive PCA at n_train=20\n")
        lines.append("Transductive = PCA basis fitted on all 4,716 spectra's "
                     "representations (no labels used; deployment does have "
                     "unlabeled spectra). Inductive = PCA fitted inside the "
                     "20-sample draw only. Both shown because they are "
                     "different deployment assumptions.\n")
        lines.append("| n-comp | stage | transductive | inductive |")
        lines.append("|---|---|---|---|")
        for n_comp in comp_counts:
            for stage in STAGE_ORDER:
                t = _best_fewshot(results, n_comp, stage, 20, "transductive")
                i = _best_fewshot(results, n_comp, stage, 20, "inductive")
                if t is None or i is None:
                    continue
                lines.append(f"| {n_comp} | {STAGE_LABELS[stage]} "
                             f"| {t[1]['r2_median']:+.3f} (`{t[0]}`) "
                             f"| {i[1]['r2_median']:+.3f} (`{i[0]}`) |")

    if results.get("pca_ceiling_r2"):
        lines.append("\n## PCA-ceiling diagnostic\n")
        lines.append("Large-n (n≈4,716) RidgeCV R² on the **same PCA-k features** the "
                     "few-shot probes see. This bounds what *any* probe could extract "
                     "from those features, so a weak `tabpfn_pca{k}` result can be "
                     "attributed correctly: if the ceiling here is high but the "
                     "few-shot number is low, the probe/sample size is the limit; if "
                     "the ceiling itself is low, PCA discarded the label direction "
                     "(it maximizes variance, not label-relevance) and no probe on "
                     "those features can recover it.\n")
        lines.append("| n-comp | stage | k=10 | k=20 | k=50 | full-dim (step 1) |")
        lines.append("|---|---|---|---|---|---|")
        for n_comp in comp_counts:
            for stage in STAGE_ORDER:
                vals = [results["pca_ceiling_r2"].get(f"{n_comp}|{stage}|{k}")
                        for k in (10, 20, 50)]
                if all(v is None for v in vals):
                    continue
                asym = results.get("asymptotic_r2", {}).get(f"{n_comp}|{stage}")
                cells = " | ".join(f"{v:+.3f}" if v is not None else "—" for v in vals)
                lines.append(f"| {n_comp} | {STAGE_LABELS[stage]} | {cells} "
                             f"| {asym:.3f} |" if asym is not None
                             else f"| {n_comp} | {STAGE_LABELS[stage]} | {cells} | — |")

    lines.append("\n## Scope\n")
    lines.append("This settles which representation is more label-efficient "
                 "under one honest small-n protocol. It does **not** validate a "
                 "deployment pipeline: real deployment also has to choose *which* "
                 "20 spectra get labeled (stratified/active selection would beat "
                 "random draws and is the obvious follow-up), plus distribution "
                 "shift and calibration. Out of scope here.\n")

    with open(os.path.join(output_dir, "step5_REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[label_probe] wrote {output_dir}/step5_REPORT.md")
