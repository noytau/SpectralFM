"""
Step-7 reporting. Verdict-driven like report6: every claim comes from the
computed crossing, and the template never asserts an overtake the data does not
show (see test_report_does_not_claim_a_crossing_it_did_not_measure).
"""
from __future__ import annotations

import os

import numpy as np

from . import crossover as cx


def _cell(results, n_comp, n, arm):
    return results["per_comp"][str(n_comp)]["per_n"][str(n)][arm]


def _sel_config(results: dict) -> tuple[dict, str]:
    """The execution-time selection configuration, read from the results dict
    when a run has persisted it. Older results files (including this study's
    own step7_results.json, generated before this field existed) carry none
    of this -- fall back to the values the run was actually launched with
    (`TASKS.md` T15: `--sel_draws 10 --sel_top_emb 3 --sel_top_raw 2`) and say
    so, rather than silently presenting a fallback as measured fact."""
    cfg = results.get("sel_config") or {}
    note = ("" if cfg else
            " (not recorded in this results file — taken from the run's "
            "launch configuration, see `TASKS.md` T15)")
    return {
        "sel_draws": cfg.get("sel_draws", 10),
        "sel_top_emb": cfg.get("sel_top_emb", 3),
        "sel_top_raw": cfg.get("sel_top_raw", 2),
    }, note


def _stage(label: str) -> str:
    parts = label.split("|")
    return parts[1] if len(parts) > 1 else label


def write_step7_report(results, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "step7_REPORT.md")
    L = []
    L.append("# Step 7 — Where does the embedding overtake raw input?\n")
    L.append("Pre-registered rule and estimator: "
             "`docs/superpowers/specs/2026-09-09-label-efficiency-crossover-design.md` §2. "
             "Readouts are re-selected per rung on `eval_a`; every number below is "
             "measured on the disjoint **`eval_b`** with independent draws.\n")

    sel_cfg, sel_note = _sel_config(results)
    laddered_ncs = sorted(int(k) for k in results["per_comp"])
    L.append("\n## Run configuration\n")
    L.append(f"- **Component count(s) laddered: {', '.join(str(x) for x in laddered_ncs)}.**")
    for nc_missing in (1, 2, 3):
        if nc_missing in laddered_ncs:
            continue
        anchor = cx.STEP1_ANCHOR.get(nc_missing)
        if anchor and anchor["raw"] > anchor["emb"]:
            L.append(f"  {nc_missing}-component was **not** laddered: step 1's "
                     f"large-n anchor (n ≈ {anchor['n_eff']}) already shows raw "
                     f"ahead there (embedding {anchor['emb']:+.3f} vs raw "
                     f"{anchor['raw']:+.3f}), so no crossing is possible to find "
                     f"in that regime.")
    missing_rungs = [n for n in cx.N_LADDER if n not in results["ladder"]]
    if missing_rungs:
        L.append(f"- **Rung(s) {', '.join(str(n) for n in missing_rungs)} were "
                 f"not (re-)measured here** — they are already published cells "
                 f"from the prior step-6 study, cited below as reference rows "
                 f"rather than re-measured.")
    L.append(f"- **The selection pass (which picks the readout per rung on "
             f"`eval_a`) ran at a reduced budget**{sel_note}: "
             f"{sel_cfg['sel_draws']} draws per rung, ranking only the top "
             f"{sel_cfg['sel_top_emb']} embedding and top {sel_cfg['sel_top_raw']} "
             f"raw candidates (plus the historical arm, never trimmed). The "
             f"reporting pass (the table below) always kept the full per-rung "
             f"draw budget (`draws_by_n`) and reported whichever candidate that "
             f"reduced selection pass picked. Selection only has to rank "
             f"candidates; reporting has to estimate a value, which needs more "
             f"draws.")

    for nc in sorted(results["per_comp"], key=int):
        cr = results["crossings"][nc]
        L.append(f"\n## {nc}-component\n")
        if cr["crossed"]:
            ci = cr["ci"]
            L.append(f"**Crossing at n = {cr['n_cross']}** "
                     f"(interpolated {cr['n_cross_interp']:.0f}, "
                     f"90% CI [{ci[0]:.0f}, {ci[1]:.0f}]). The embedding is ahead "
                     f"from that rung on, and stays ahead.\n")
        else:
            L.append(f"**No crossing on this ladder** — raw input leads at every "
                     f"rung up to n = {max(results['ladder'])}. Largest median "
                     f"ΔR² was {cr['best_delta']:+.3f} at n = {cr['best_delta_n']}; "
                     f"the crossing, if any, lies above the ladder.\n")

        L.append("| n_train | draws | emb R² | raw R² | historical R² | ΔR² | "
                 "boot P(Δ>0) | won | emb readout | emb probe | raw probe | "
                 "reliability (emb frac+/fail, raw frac+/fail, dropped pairs) |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|:--:|---|---|---|---|")
        for n in results["ladder"]:
            e, r, h = (_cell(results, nc, n, a) for a in ("emb", "raw", "historical"))
            ru = cr["rungs"].get(n, cr["rungs"].get(str(n), {}))
            # A missing rung must render as visibly missing in every derived
            # column. `0.00`/`no` are indistinguishable from real computed
            # values (unlike the delta column's `nan`), so a silently-absent
            # rung would read as "measured, near-zero win probability" instead
            # of "not measured" -- the same class of bug as the int/str key
            # lookup that once swallowed four rows via a caught KeyError.
            if ru:
                delta_s = f"{ru.get('delta_median', float('nan')):+.3f}"
                boot_s = f"{ru.get('boot_frac_positive', 0.0):.2f}"
                won_s = "yes" if ru.get("won") else "no"
                dropped_s = str(ru.get("n_dropped", "—"))
            else:
                delta_s = boot_s = won_s = dropped_s = "—"
            # Reliability: zero failures today looks identical to a healthy
            # run, but a future rung where half the draws failed would render
            # an identical-looking median with no signal anything went wrong
            # -- exactly the silent class this project has been bitten by
            # twice. Report frac_positive_r2 / n_failed_draws per arm plus the
            # paired-delta's n_dropped, all already carried in the results.
            rel_s = (f"{e.get('frac_positive_r2', float('nan')):.2f}/"
                    f"{e.get('n_failed_draws', '?')}, "
                    f"{r.get('frac_positive_r2', float('nan')):.2f}/"
                    f"{r.get('n_failed_draws', '?')}, {dropped_s}")
            L.append(
                f"| {n} | {results['draws_by_n'][str(n)]} | "
                f"{e['r2_median']:+.3f} | {r['r2_median']:+.3f} | "
                f"{h['r2_median']:+.3f} | {delta_s} | "
                f"{boot_s} | "
                f"{won_s} | "
                f"`{e['readout']}` | `{e['probe']}` | `{r['probe']}` | {rel_s} |")

        anchor = cx.STEP1_ANCHOR.get(int(nc))
        if anchor:
            L.append(f"\nStep-1 anchor at n ≈ {anchor['n_eff']} (pooled 5-fold, "
                     f"different protocol, plotted not scored): embedding "
                     f"{anchor['emb']:+.3f} vs raw {anchor['raw']:+.3f}.")

        # --- Finding 2/3: the selection pool was trimmed, and the winner moved. ---
        emb_pool_full = [rd for rd in cx.CANDIDATES[int(nc)]
                         if rd.family == "emb" and rd != cx.HISTORICAL]
        ranked_emb = min(sel_cfg["sel_top_emb"], len(emb_pool_full))
        selected = results["per_comp"][nc]["selected"]
        emb_stage_seq = [_stage(selected[str(n)]["emb"]) for n in results["ladder"]]
        distinct_in_order = list(dict.fromkeys(emb_stage_seq))
        n_changes = sum(1 for i in range(1, len(emb_stage_seq))
                        if emb_stage_seq[i] != emb_stage_seq[i - 1])
        L.append(f"\n**Only {ranked_emb} of {len(emb_pool_full)} embedding "
                 f"candidates were ever ranked during selection** "
                 f"(`sel_top_emb={sel_cfg['sel_top_emb']}`), and the winning "
                 f"embedding readout changed {n_changes} time(s) across the "
                 f"{len(emb_stage_seq)} rungs on this ladder "
                 f"({' → '.join(distinct_in_order)}) — i.e. the pick sat on "
                 f"the boundary of the trimmed pool rather than settling on "
                 f"one clear winner. \"The best embedding readout stays behind "
                 f"raw\" is therefore bounded by this "
                 f"{ranked_emb}-candidate pool, not verified against the full "
                 f"panel at every rung.\n")

        # --- Finding 4b: was any unwhitened raw readout ever measured here? ---
        raw_pool_full = [rd for rd in cx.CANDIDATES[int(nc)] if rd.family == "raw"]
        raw_pool_sel = raw_pool_full[:sel_cfg["sel_top_raw"]]
        unwhitened_sel = [rd for rd in raw_pool_sel if rd.normalizer != "whiten"]
        if raw_pool_sel and not unwhitened_sel:
            sel_labels = ", ".join(f"`{rd.label()}`" for rd in raw_pool_sel)
            L.append(f"**No unwhitened raw readout was ever measured at "
                     f"n ≥ {min(results['ladder'])} on this ladder.** The raw "
                     f"selection pool was trimmed to the top "
                     f"{sel_cfg['sel_top_raw']} (of {len(raw_pool_full)}) "
                     f"step-6-ranked raw candidates "
                     f"(`sel_top_raw={sel_cfg['sel_top_raw']}`), and all of "
                     f"them use `whiten` normalization ({sel_labels}). The "
                     f"attribution of the study's original premise to an "
                     f"under-whitened raw baseline therefore rests on the "
                     f"*previous* step-6 study's n=20/50 evidence for an "
                     f"unwhitened raw readout, not on a measurement from this "
                     f"ladder.\n")

    L.append("\n## How to read this\n")
    L.append("- **The reported number is also argmax-selected, on `eval_b` "
             "itself.** The *readout* (which representation) is chosen per "
             "rung on `eval_a`, as stated above — but for whichever readout "
             "wins, the reported R² is the maximum over every probe in that "
             "rung's panel scored on `eval_b` (9-12 probes over 10-60 draws). "
             "That is a second, unadvertised selection step on the same set "
             "the numbers are reported from. It inflates *both* arms — "
             "embedding and raw always see the identical panel and the "
             "identical argmax procedure at a given rung — so it cannot flip "
             "the sign of the embedding-vs-raw comparison, but it does mean "
             "the columns below are not unbiased held-out estimates of any "
             "single probe's performance.")
    L.append("- **Draws above n = 1000 overlap heavily.** They are drawn without "
             "replacement from a 3,716-spectrum pool, so two draws at n = 2000 share "
             "~54% of their rows. The spread at the top two rungs therefore "
             "understates true sampling variance and their bootstrap intervals are "
             "optimistically narrow.")
    L.append(f"- **The probe panel varies with n** — `ridge_strong`'s alpha grid is "
             f"biased for a ~20-label fit, a rung not run on this ladder "
             f"(n = 20, 50 are cited from step 6, not re-measured here); it is "
             f"likely already over-regularized by the smallest rung actually "
             f"measured (n = {min(results['ladder'])}) and drops from the panel "
             f"entirely at n ≥ 1000. Exact GP and the kernel-ridge grid search "
             f"are cubic and are dropped there too. Both arms always get the "
             f"identical panel at a given rung, so the confound is between "
             f"rungs, not between arms.")
    L.append("- **The historical column is `layer12`/mean**, the tap every prior "
             "measurement of this backbone used. Its gap to the embedding column is "
             "how much of the story was the readout rather than the regime.")
    L.append("- `parameter_0` is a 168-level designed grid, pre-standardized "
             "(mean 0.000, std 0.9999). R² is measured against the most favourable "
             "target variance it can have; the crossing *n* is more transportable "
             "than the R² values either side of it.")
    L.append("- Exact-label overlap: 100% of `eval_b` labels also occur in the pool, "
             "and a 20-label draw matches the exact label of ~15% of `eval_b` rows "
             "(~33% at n = 50). This inflates both arms and is not corrected here.")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[step7] wrote {path}", flush=True)
    return path


def plot_crossover(results, path: str, n_comp: int) -> str:
    """Both arms and the historical arm against n (log x), with the crossing
    band and the step-1 anchor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = results["ladder"]
    series = {a: [_cell(results, n_comp, n, a)["r2_median"] for n in ns]
              for a in ("emb", "raw", "historical")}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(ns, series["raw"], marker="o", lw=2, label="raw input (best readout)")
    ax.plot(ns, series["emb"], marker="o", lw=2, label="embedding (best readout)")
    ax.plot(ns, series["historical"], marker="^", lw=1.4, ls=":",
            label="embedding, layer12/mean (historical tap)")

    cr = results["crossings"][str(n_comp)]
    if cr["crossed"] and cr["ci"]:
        ax.axvspan(cr["ci"][0], cr["ci"][1], alpha=0.14, color="grey")
        ax.axvline(cr["n_cross_interp"], ls="--", lw=1.4, color="grey")
        ax.annotate(f"crossing ≈ {cr['n_cross_interp']:.0f}",
                    xy=(cr["n_cross_interp"], ax.get_ylim()[0]),
                    xytext=(4, 6), textcoords="offset points", fontsize=8)

    anchor = cx.STEP1_ANCHOR.get(n_comp)
    if anchor:
        ax.plot([anchor["n_eff"]], [anchor["emb"]], marker="*", ms=13, ls="none",
                label=f"step-1 anchor n≈{anchor['n_eff']} (emb)")
        ax.plot([anchor["n_eff"]], [anchor["raw"]], marker="*", ms=13, ls="none",
                label=f"step-1 anchor n≈{anchor['n_eff']} (raw)")

    ax.set_xscale("log")
    ax.set_xlabel("labeled training spectra (log scale)")
    ax.set_ylabel("median held-out R² (eval_b)")
    ax.set_title(f"Label efficiency — {n_comp}-component")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
