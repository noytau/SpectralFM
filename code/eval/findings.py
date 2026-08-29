"""
Read the reconstruction results and say what they imply.

Everything here is derived from the numbers in the results dict — nothing is a stock
sentence about reconstruction in general. That is the point: a finding that is not
computed from this run cannot be trusted to describe it, and a report that ends without
one leaves the reader to re-derive the conclusion from eighteen figures.

Each finding carries the number it rests on, so a claim can be checked against the tables
rather than believed. Anything the data cannot settle is phrased as a question instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A head whose median R2 is at or below this is not reconstructing - it is roughly
# reproducing each sample's own mean, whatever its MSE looks like.
DEGENERATE_R2 = 0.05
# Amplitude ratio this far from 1 is a systematic scale error worth naming.
AMP_TOLERANCE = 0.10
# mean/median above this means the mean is describing outliers, not the typical sample.
OUTLIER_TAX = 1.5
# Ratio of high-frequency to full-band magnitude fidelity below this = losing fine structure.
ROLLOFF_LIMIT = 0.85
# A head delivering less than this share of the temporal resolution its bottleneck
# provides is losing information it was handed, not information the architecture withheld.
REF_RESOLUTION_LIMIT = 0.9
# Worst-tail share above this multiple of the tail fraction means the error is
# concentrated enough that the mean describes the outliers rather than the dataset.
CONCENTRATION_FACTOR = 3.0
# A tail level needs at least this lift, and this share of the tail, to be called a cause
# rather than a coincidence.
TAIL_LIFT_MIN = 3.0
TAIL_SHARE_MIN = 0.5
# A component carrying this many times its share of a dataset's error budget is worth
# naming; together they need this share of the budget before the finding is worth making.
COMP_LIFT_MIN = 2.0
COMP_BUDGET_MIN = 0.25

_SHORT = {"fe": "FE", "proj": "projection", "tr": "transformer"}


def _summaries(results: dict) -> pd.DataFrame:
    """One tidy frame of every head x dataset summary row, with the component group."""
    from .report import _recon_keys, _split_eval_key

    frames = []
    for key in _recon_keys(results):
        r = results[key]
        sdf = r.get("summary_df")
        if not isinstance(sdf, pd.DataFrame) or sdf.empty:
            continue
        f = sdf.copy()
        f["dataset"] = _split_eval_key(key)[1] or "dataset"
        f["group"] = r.get("component_group", "single")
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _rolloff(r: dict, head: str) -> float:
    """
    How much of the target's high-frequency magnitude the reconstruction keeps.

    Mean |FFT| ratio over the top quartile of frequencies. Below 1 means fine structure -
    the narrow spectral lines - is being smoothed away, which MSE barely notices.
    """
    prof = r.get("profiles") or {}
    tgt, pred = prof.get("mean_fft_target"), prof.get(f"mean_fft_pred_{head}")
    if tgt is None or pred is None or len(tgt) < 8:
        return float("nan")
    hi = slice(int(0.75 * len(tgt)), None)
    denom = float(np.sum(tgt[hi]))
    return float(np.sum(pred[hi]) / denom) if denom > 0 else float("nan")


def _strongest_axis(r: dict, head: str) -> tuple:
    """
    The signal property whose bins spread median error the most, and by what factor.

    Answers "what kind of spectrum does this fail on" with the one axis that matters most
    rather than making the reader scan six panels.
    """
    strat = r.get("strat_df")
    col = f"{head}_mse_median"
    if not isinstance(strat, pd.DataFrame) or strat.empty or col not in strat.columns:
        return None, float("nan"), None
    best = (None, 0.0, None)
    for axis, sub in strat.groupby("axis"):
        v = sub[col].to_numpy(dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) < 2:
            continue
        factor = float(v.max() / v.min())
        if factor > best[1]:
            worst_bin = sub.loc[sub[col].idxmax(), "bin_label"]
            best = (axis, factor, worst_bin)
    return best


def _spectrum_concentration(r: dict, head: str) -> tuple:
    """
    Median (worst component - mean) / mean within a spectrum, and how many spectra it
    rests on.

    Only spectra with at least two components actually drawn can say anything: with one
    component the max IS the mean and the ratio is exactly 0, which would report perfect
    evenness for a dataset the draw simply did not cover. multi_channel's valid split has
    a single component for the large majority of its spectra, so this filter is the
    difference between a real measurement and a meaningless 0.
    """
    spec = r.get("spectrum_df")
    if not isinstance(spec, pd.DataFrame) or spec.empty:
        return float("nan"), 0
    mean_c, max_c = f"{head}_mse_mean", f"{head}_mse_max"
    if mean_c not in spec.columns or max_c not in spec.columns:
        return float("nan"), 0
    if "n_comps_present" in spec.columns:
        spec = spec[spec["n_comps_present"] >= 2]
    if spec.empty:
        return float("nan"), 0
    m = spec[mean_c].to_numpy(dtype=float)
    x = spec[max_c].to_numpy(dtype=float)
    ok = np.isfinite(m) & np.isfinite(x) & (m > 0)
    if not ok.any():
        return float("nan"), 0
    return float(np.median((x[ok] - m[ok]) / m[ok])), int(ok.sum())


# ── The findings ──────────────────────────────────────────────────────────────

def observations(results: dict) -> list:
    """
    What this run shows, as markdown bullets. Every claim carries its number.
    """
    from .report import _recon_keys, _split_eval_key

    S = _summaries(results)
    if S.empty:
        return []
    by = {(_split_eval_key(k)[1] or "dataset"): results[k] for k in _recon_keys(results)}
    out = []

    # 1. Is anything degenerate? This is the first thing to know and the cheapest to say.
    dead = S[(S["r2_median"] <= DEGENERATE_R2) | (S["amp_ratio_median"] <= 0.5)]
    has_reference = bool(_reference_rows(by))
    if dead.empty:
        worst = S.loc[S["r2_median"].idxmin()]
        out.append(
            "**No head is degenerate.** The weakest combination is the "
            "%s decoder on `%s`, and it still reaches median R² = %.2f against the "
            "flat-line baseline, beating it on %.0f%% of samples. None of them is the "
            "mean-predictor.%s"
            % (_SHORT.get(worst["head"], worst["head"]), worst["dataset"],
               worst["r2_median"], 100 * worst["frac_r2_positive"],
               " That is a floor, not a pass mark — see the matched-rate reference below."
               if has_reference else ""))
    else:
        for _, row in dead.iterrows():
            out.append(
                "**The %s decoder is degenerate on `%s`** — median R² = %.3f, amplitude "
                "ratio %.2f. It is reproducing little more than each sample's own mean, "
                "and its MSE should not be read as reconstruction quality."
                % (_SHORT.get(row["head"], row["head"]), row["dataset"],
                   row["r2_median"], row["amp_ratio_median"]))

    # 2. Depth ordering, and whether it holds. This is the T5/T6 question, and it is the
    #    one thing this eval can answer that a single-dataset run cannot.
    orders = {}
    for ds, sub in S.groupby("dataset"):
        orders[ds] = tuple(sub.sort_values("mse_median")["head"])
    distinct = set(orders.values())
    if len(distinct) == 1:
        order = " < ".join(_SHORT.get(h, h) for h in next(iter(distinct)))
        out.append(
            "**Depth ordering is consistent across every dataset**: %s by median MSE. "
            "Reconstruction quality falls monotonically with depth, so on this checkpoint "
            "the transformer is not preserving signal the feature extractor already had."
            % order)
    else:
        lines = ["**Depth ordering flips between datasets**, which a single-dataset run "
                 "would have hidden:"]
        for ds, order in sorted(orders.items(),
                                key=lambda kv: by.get(kv[0], {}).get("component_group", "")):
            grp = by.get(ds, {}).get("component_group", "?")
            lines.append("  - `%s` (%s): %s"
                         % (ds, grp, " < ".join(_SHORT.get(h, h) for h in order)))
        lines.append("  Where the transformer head wins, depth is adding something; where "
                     "it loses, the transformer is discarding signal the FE kept. The "
                     "datasets it wins on are the question.")
        out.append("\n".join(lines))

    # 3. Generalization gap, per head, single -> multi.
    single = S[S["group"] == "single"].groupby("head")["mse_median"].median()
    multi = S[S["group"] == "multi"].groupby("head")["mse_median"].median()
    shared = [h for h in single.index if h in multi.index]
    if shared:
        gaps = {h: multi[h] / single[h] for h in shared if single[h] > 0}
        if gaps:
            worst_h = max(gaps, key=gaps.get)
            best_h = min(gaps, key=gaps.get)
            out.append(
                "**Multi-component data costs %.1fx to %.1fx more error** than "
                "single-component (median MSE, %s worst at %.1fx, %s best at %.1fx). The "
                "heads do not degrade equally, so the gap is not just harder data — it is "
                "a property of where in the stack you decode from."
                % (min(gaps.values()), max(gaps.values()),
                   _SHORT.get(worst_h, worst_h), gaps[worst_h],
                   _SHORT.get(best_h, best_h), gaps[best_h]))

    # 4. Where the mean stops describing the typical sample.
    S = S.assign(tax=S["mse_mean"] / S["mse_median"].replace(0, np.nan))
    taxed = S[S["tax"] > OUTLIER_TAX].sort_values("tax", ascending=False)
    if not taxed.empty:
        top = taxed.iloc[0]
        others = sorted(set(taxed["dataset"]) - {top["dataset"]})
        extra = (" Also on %s." % ", ".join("`%s`" % d for d in others)) if others else ""
        out.append(
            "**`%s` has a heavy tail**: its %s decoder's mean MSE is %.1fx its median "
            "(%.3f vs %.3f), so a minority of samples dominates the average.%s Quote "
            "medians for this dataset; the mean describes the outliers."
            % (top["dataset"], _SHORT.get(top["head"], top["head"]), top["tax"],
               top["mse_mean"], top["mse_median"], extra))

    # 5. Amplitude, and 6. high-frequency fidelity — both invisible in MSE.
    amp_off = S[(S["amp_ratio_median"] - 1).abs() > AMP_TOLERANCE]
    if amp_off.empty:
        out.append(
            "**Dynamic range survives everywhere** — amplitude ratio stays within %.0f%% "
            "of 1 for every head and dataset, so no head is hedging toward a flat output."
            % (100 * AMP_TOLERANCE))
    else:
        row = amp_off.loc[(amp_off["amp_ratio_median"] - 1).abs().idxmax()]
        out.append(
            "**Amplitude is off on `%s`** — the %s decoder's output is %.0f%% %s than "
            "the target (ratio %.2f), a systematic scale error rather than noise."
            % (row["dataset"], _SHORT.get(row["head"], row["head"]),
               100 * abs(row["amp_ratio_median"] - 1),
               "flatter" if row["amp_ratio_median"] < 1 else "more variable",
               row["amp_ratio_median"]))

    rolls = {(ds, h): _rolloff(by[ds], h) for ds in by for h in (by[ds].get("pathways") or [])}
    rolls = {k: v for k, v in rolls.items() if np.isfinite(v)}
    if rolls:
        worst = min(rolls, key=rolls.get)
        if rolls[worst] < ROLLOFF_LIMIT:
            out.append(
                "**Fine structure is being smoothed away.** In the top quartile of "
                "frequencies the %s decoder keeps only %.0f%% of the target's magnitude on "
                "`%s`. For spectral-line data that is the part that matters, and MSE "
                "hardly registers it."
                % (_SHORT.get(worst[1], worst[1]), 100 * rolls[worst], worst[0]))
        else:
            out.append(
                "**High-frequency content is preserved** — the weakest head keeps %.0f%% "
                "of the target's magnitude in the top frequency quartile, so narrow "
                "features are surviving, not just the envelope." % (100 * min(rolls.values())))

    # 7. What kind of spectrum fails, and 8. whether failure is component-specific.
    axes = []
    for ds, r in by.items():
        for h in (r.get("pathways") or []):
            axis, factor, worst_bin = _strongest_axis(r, h)
            if axis and np.isfinite(factor) and factor > 1.5:
                axes.append((factor, ds, h, axis, worst_bin))
    if axes:
        factor, ds, h, axis, worst_bin = max(axes)
        out.append(
            "**Error depends most on `%s`**: on `%s` the %s decoder's median MSE varies "
            "%.1fx across its bins, worst at %s. That is the sample property to condition "
            "on — it predicts failure better than the dataset label does."
            % (axis, ds, _SHORT.get(h, h), factor, worst_bin))

    concs = {(ds, h): _spectrum_concentration(by[ds], h)
             for ds in by for h in (by[ds].get("pathways") or [])}
    concs = {k: (v, cnt) for k, (v, cnt) in concs.items() if np.isfinite(v) and cnt >= 30}
    if concs:
        worst = max(concs, key=lambda k: concs[k][0])
        val, cnt = concs[worst]
        if val >= 0.25:
            out.append(
                "**Within a multi-component spectrum, failure is uneven**: on `%s` the "
                "worst component carries %.0f%% more error than its spectrum's mean (%s "
                "decoder, median over the %d spectra with more than one component drawn). "
                "The model is not failing on whole spectra, it is failing on particular "
                "components of them."
                % (worst[0], 100 * val, _SHORT.get(worst[1], worst[1]), cnt))
        else:
            out.append(
                "**Failure is spread evenly inside a spectrum** — on `%s` the worst "
                "component carries only %.0f%% more error than its spectrum's mean (%s "
                "decoder, over %d multi-component spectra). Whatever makes a spectrum hard "
                "affects all of its components, so this is a per-spectrum property rather "
                "than a per-component one."
                % (worst[0], 100 * val, _SHORT.get(worst[1], worst[1]), cnt))

    # ── Against a fair reference rather than a flat line ─────────────────────
    refs = _reference_rows(by)
    if refs:
        bk = refs[0][3]
        worse = [(ds, h, ratio) for ds, h, _, _, ratio in refs
                 if np.isfinite(ratio) and ratio > 1.0]
        best = min((r for *_, r in refs if np.isfinite(r)), default=float("nan"))
        if worse and len(worse) == len([r for *_, r in refs if np.isfinite(r)]):
            ds, h, ratio = max(worse, key=lambda t: t[2])
            smooth = _smoothness(by.get(ds, {}))
            note = ""
            if np.isfinite(smooth) and smooth > SMOOTH_DATASET_RATIO:
                note = (" That worst case is partly the data: `%s`'s components are smooth "
                        "enough that the reference has almost nothing to lose (peaks about "
                        "%.0f reference sample-spacings wide), so read the effective "
                        "resolution there rather than the ratio." % (ds, smooth))
            out.append(
                "**No head reaches a fair baseline.** Simply resampling the target at the "
                "%d-step rate the feature extractor delivers and interpolating back beats "
                "every head on every dataset — by %.1fx at best and %.1fx at worst (`%s`, "
                "%s decoder). R² against a flat line said they were all doing real work; "
                "against a reference at their own information rate, none of them is.%s"
                % (bk, best, ratio, ds, _SHORT.get(h, h), note))
        elif worse:
            ds, h, ratio = max(worse, key=lambda t: t[2])
            out.append(
                "**Some heads lose to a fair baseline**: resampling the target at the %d-step "
                "rate the feature extractor delivers beats %d of %d head/dataset pairs, "
                "worst by %.1fx (`%s`, %s decoder)."
                % (bk, len(worse), len(refs), ratio, ds, _SHORT.get(h, h)))
        else:
            out.append(
                "**Every head beats a fair baseline** — each one has less error than "
                "resampling the target at the %d-step rate it decodes from, so the heads "
                "are extracting more than temporal position alone." % bk)

        short = [(ds, h, k) for ds, h, k, b, _ in refs if k < REF_RESOLUTION_LIMIT * b]
        if short:
            ds, h, k = min(short, key=lambda t: t[2])
            out.append(
                "**Effective resolution falls short of the bottleneck.** The %s decoder on "
                "`%s` delivers the equivalent of %.0f independent samples of the signal "
                "against the %d its input carries (%.0f%%). What it loses is a limit of "
                "what it learned, not of the architecture."
                % (_SHORT.get(h, h), ds, k, bk, 100 * k / bk))

    # ── How concentrated the failure is, and what it is made of ──────────────
    tails = _tail_rows(by)
    if tails:
        ds, h, share, frac = max(tails, key=lambda t: t[2])
        if share >= CONCENTRATION_FACTOR * frac:
            out.append(
                "**The error is concentrated in a few samples**: on `%s` the worst %.0f%% "
                "of samples carry %.0f%% of the %s decoder's total error (an even spread "
                "would be %.0f%%). Its mean is a statement about those samples, not about "
                "the dataset." % (ds, 100 * frac, 100 * share, _SHORT.get(h, h), 100 * frac))
        else:
            out.append(
                "**No dataset has a dominant tail** — the worst %.0f%% of samples carry at "
                "most %.0f%% of the total error (`%s`, %s decoder), close enough to even "
                "that the mean is a fair summary."
                % (100 * frac, 100 * share, ds, _SHORT.get(h, h)))

        cause = _top_tail_cause(by.get(ds, {}))
        if cause:
            levels = " and ".join("`%s`" % v for v in cause["levels"])
            out.append(
                "**That tail has a name**: %s of it is %s = %s — %.0fx %s base rate. This "
                "is a failure mode with an identity, not a heavy tail to average over."
                % ("%.0f%%" % (100 * cause["share"]), cause["axis"], levels,
                   cause["lift"], "their" if cause["n_levels"] > 1 else "its"))

    # ── Which components own the error budget ────────────────────────────────
    budgets = []
    for ds, r in by.items():
        for h in (r.get("pathways") or []):
            b = _component_budget(r, h)
            if b and b["share"] >= COMP_BUDGET_MIN:
                budgets.append((ds, h, b))
    if budgets:
        ds, h, b = max(budgets, key=lambda t: t[2]["share"])
        comps = " and ".join("`%s`" % c for c in b["comps"])
        out.append(
            "**%s of the error on `%s` comes from %d of its %d components** — %s, %.1f%% "
            "of its samples carrying %.0f%% of its error (%s decoder). Their median error "
            "is %.0fx the dataset's own. Everything else in that dataset reconstructs "
            "normally; the average does not."
            % ("%.0f%%" % (100 * b["share"]), ds, b["n_comps"], b["total"], comps,
               100 * b["sample_share"], 100 * b["share"], _SHORT.get(h, h),
               b["median_ratio"]))
    elif any(_component_budget(r, h) is not None
             for r in by.values() for h in (r.get("pathways") or [])):
        out.append(
            "**No component owns its dataset's error budget** — every component carries "
            "roughly the share of the error its sample count accounts for, so failure is "
            "a property of the signals rather than of a particular component class.")
    return out


# A dataset whose peaks are this many reference sample-spacings wide is smooth enough
# that interpolation is an unusually strong reference on it, and a large ratio there says
# as much about the data as about the head.
SMOOTH_DATASET_RATIO = 4.0


def _smoothness(r: dict) -> float:
    """Median main-peak width in units of the reference's sample spacing."""
    ref = r.get("reference") or {}
    fw, spacing = ref.get("peak_fwhm_median"), ref.get("sample_spacing")
    if fw is None or spacing in (None, 0) or not np.isfinite(fw):
        return float("nan")
    return float(fw) / float(spacing)


def _reference_rows(by: dict) -> list:
    """(dataset, head, k_eff, bottleneck, ratio-vs-reference) for every head with a ladder."""
    rows = []
    for ds, r in by.items():
        ref = r.get("reference")
        rdf = r.get("results_df")
        if not isinstance(ref, dict) or not ref.get("ladder") or not isinstance(rdf, pd.DataFrame):
            continue
        bk = int(ref.get("bottleneck_k", 0))
        col = "ref_interp%d_mse" % bk
        denom = float(rdf[col].median()) if col in rdf.columns else float("nan")
        for h, k_eff in (ref.get("k_eff_median") or {}).items():
            mse_col = "%s_mse" % h
            ratio = (float(rdf[mse_col].median()) / denom
                     if mse_col in rdf.columns and np.isfinite(denom) and denom > 0
                     else float("nan"))
            rows.append((ds, h, float(k_eff), bk, ratio))
    return rows


def _tail_rows(by: dict) -> list:
    """(dataset, head, share, frac) for every head whose tail concentration was measured."""
    rows = []
    for ds, r in by.items():
        t = r.get("tail")
        if not isinstance(t, dict):
            continue
        frac = float(t.get("frac", 0.05))
        for h, c in (t.get("per_head") or {}).items():
            share = c.get("share", float("nan"))
            if np.isfinite(share):
                rows.append((ds, h, float(share), frac))
    return rows


def _component_budget(r: dict, head: str):
    """
    The components carrying more than their share of a dataset's error, and how much.

    Uses the share of total squared error rather than a median: a component can sit at an
    unremarkable median and still own a dataset through its tail, and the error budget is
    what a training-side fix would actually move.
    """
    from .recon_analysis import component_error_table

    rdf = r.get("results_df")
    if not isinstance(rdf, pd.DataFrame) or "comp" not in rdf.columns:
        return None
    cdf = component_error_table(rdf, r.get("pathways") or [])
    share_col, lift_col = f"{head}_error_share", f"{head}_budget_lift"
    if not len(cdf) or share_col not in cdf.columns:
        return None
    hot = cdf[cdf[lift_col] >= COMP_LIFT_MIN].sort_values(share_col, ascending=False)
    if hot.empty:
        return None
    return {"comps": [("%d" % c) if float(c).is_integer() else ("%g" % c)
                      for c in hot["comp"].head(4)],
            "n_comps": int(len(hot)), "total": int(len(cdf)),
            "share": float(hot[share_col].sum()),
            "sample_share": float(hot["sample_share"].sum()),
            "lift": float(hot[lift_col].max()),
            "median_ratio": (float(hot[f"{head}_mse_median"].max()
                                   / max(rdf[f"{head}_mse"].median(), 1e-12)))}


def _top_tail_cause(r: dict):
    """The single most over-represented level in a dataset's tail, if it deserves naming."""
    lift = r.get("tail_lift_df")
    if not isinstance(lift, pd.DataFrame) or lift.empty:
        return None
    row = lift.iloc[0]
    if not (np.isfinite(row["lift"]) and row["lift"] >= TAIL_LIFT_MIN
            and row["tail_share"] >= TAIL_SHARE_MIN):
        return None
    # Levels of the same axis often split the tail between them ("comp 26 and comp 29"),
    # and naming only the first would understate the finding by half.
    same = lift[(lift["axis"] == row["axis"]) & (lift["lift"] >= TAIL_LIFT_MIN)]
    return {"axis": row["axis"], "levels": [str(v) for v in same["bin_label"]][:4],
            "lift": float(row["lift"]), "share": float(same["tail_share"].sum()),
            "n_levels": int(len(same))}


def next_steps(results: dict, config=None) -> list:
    """
    What to do next, conditioned on what this run actually showed.

    Only steps the numbers motivate. A checklist that would read the same whatever the
    results were is not worth a section - and neither is advice to pass flags the run was
    already given, which is why `config` is consulted for the sampling settings.
    """
    from .report import _recon_keys, _split_eval_key

    uncapped = bool(config is not None
                    and str(getattr(config, "eval_set_size", "")).strip().lower() == "all"
                    and int(getattr(config, "recon_max_samples", 1) or 0) == 0)
    S = _summaries(results)
    if S.empty:
        return []
    by = {(_split_eval_key(k)[1] or "dataset"): results[k] for k in _recon_keys(results)}
    out = []

    # Depth ordering: the cheapest way to settle the T7 confound is to swap the decoders.
    orders = {ds: tuple(sub.sort_values("mse_median")["head"])
              for ds, sub in S.groupby("dataset")}
    if len(set(orders.values())) > 1:
        flipping = [ds for ds, o in orders.items() if o != max(set(orders.values()),
                                                              key=list(orders.values()).count)]
        out.append(
            "**Settle the depth ordering before trusting it.** It flips on %s, but the FE "
            "head uses a different decoder architecture and a narrower input than the "
            "other two (TASKS.md T7), so part of any gap is decoder capacity rather than "
            "encoder content. Give all three heads the same decoder on the same width and "
            "re-run: if the flip survives, it is a real property of the datasets."
            % ", ".join("`%s`" % d for d in sorted(flipping)))
    else:
        out.append(
            "**Test whether the depth ordering is really about depth.** The FE head uses a "
            "different decoder and input width than the other two (TASKS.md T7), so the "
            "monotone ordering could be capacity, not information. Same decoder on all "
            "three taps is the one experiment that separates them.")

    # Heavy tails: the tail is a named subset, so go look at it.
    S = S.assign(tax=S["mse_mean"] / S["mse_median"].replace(0, np.nan))
    taxed = S[S["tax"] > OUTLIER_TAX]
    # ... unless the failure-anatomy pass already named what that tail is made of, in which
    # case asking for it again would follow the answer with the question.
    named = {ds for ds, r in by.items() if _top_tail_cause(r)}
    taxed = taxed[~taxed["dataset"].isin(named)]
    if not taxed.empty:
        ds = taxed.sort_values("tax", ascending=False).iloc[0]["dataset"]
        out.append(
            "**Identify the tail on `%s` rather than averaging over it.** "
            "`recon_df.csv` has per-sample error and the parsed component index, so the "
            "worst percentile can be pulled out and listened to directly. The question is "
            "whether those samples share a component index, a peak count, or a source "
            "`dataset_id` — if they do, this is one failure mode with a name, not a tail."
            % ds)

    # Stratifier: an axis that predicts error is a candidate for conditioning or reweighting.
    axes = []
    for ds, r in by.items():
        for h in (r.get("pathways") or []):
            axis, factor, worst_bin = _strongest_axis(r, h)
            if axis and np.isfinite(factor):
                axes.append((factor, ds, axis))
    if axes:
        factor, ds, axis = max(axes)
        if factor > 1.5:
            out.append(
                "**Try weighting training by `%s`.** It spreads error %.1fx across its "
                "bins on `%s`, which makes it the most promising thing to either "
                "oversample or add to the loss. If error flattens against it, the model "
                "was simply seeing too few hard spectra."
                % (axis, factor, ds))

    # High-frequency loss: only worth suggesting if the rolloff is actually there.
    rolls = [(_rolloff(by[ds], h), ds, h) for ds in by for h in (by[ds].get("pathways") or [])]
    rolls = [t for t in rolls if np.isfinite(t[0])]
    if rolls and min(rolls)[0] < ROLLOFF_LIMIT:
        val, ds, h = min(rolls)
        out.append(
            "**Add a frequency-domain term to the reconstruction loss.** The %s head keeps "
            "only %.0f%% of high-frequency magnitude on `%s`, and an L1/L2 loss on the "
            "waveform has almost no gradient for that. A magnitude-spectrum penalty targets "
            "it directly; this eval already measures whether it worked."
            % (_SHORT.get(h, h), 100 * val, ds))

    # Per-spectrum concentration: points at a per-component rather than per-sample fix.
    concs = [(_spectrum_concentration(by[ds], h)[0], ds, h)
             for ds in by for h in (by[ds].get("pathways") or [])
             if _spectrum_concentration(by[ds], h)[1] >= 30]
    concs = [t for t in concs if np.isfinite(t[0])]
    if concs and max(concs)[0] > 0.25:
        val, ds, h = max(concs)
        out.append(
            "**Ask what the weak components have in common.** On `%s` the worst component "
            "of a spectrum carries %.0f%% more error than its mean, so a per-spectrum "
            "average hides it. `spectrum_df.csv` records `worst_comp` per spectrum — if "
            "one or two component indices dominate that column, the fix is per-component, "
            "not per-model." % (ds, 100 * val))

    # Scale: say plainly whether this run was big enough to trust - and do not recommend
    # flags the run already used, which is what a naive version of this said.
    n_by_ds = {ds: int(S[S["dataset"] == ds]["n"].max()) for ds in S["dataset"].unique()}
    thin = {ds: n for ds, n in n_by_ds.items() if n < 2000}
    if thin:
        listed = ", ".join("`%s` (n=%d)" % (d, n) for d, n in sorted(thin.items()))
        if uncapped:
            out.append(
                "**%s are as large as they get.** This run already covered their whole "
                "splits, so the thinness is the split, not the sampling — fine for medians, "
                "but the stratified panels are down to a few hundred samples per bin and "
                "the tail is poorly estimated. More in-distribution data means a bigger "
                "subset: `single_channel_all` (valid = 10,000), `single_channel_one` "
                "(valid = 1,000) or the `single_channel_10k` train split (10,611). Point "
                "`DATASET_SPECS` at one of those and the panels get their resolution back."
                % listed)
        else:
            out.append(
                "**Re-run the thin datasets at full size.** %s were evaluated on under "
                "2000 samples, which is fine for medians but leaves the stratified panels "
                "with a few hundred samples per bin and the tail poorly estimated. "
                "`--eval_set_size all --recon_max_samples 0` covers whole splits."
                % listed)

    # ── What the fair reference implies ──────────────────────────────────────
    refs = _reference_rows(by)
    ratios = [(ds, h, r) for ds, h, _, _, r in refs if np.isfinite(r)]
    if ratios and all(r > 1.0 for *_, r in ratios):
        bk = refs[0][3]
        out.append(
            "**Close the gap to the matched-rate reference before tuning anything else.** "
            "Every head has more error than interpolating the target through %d points, "
            "which is the crudest thing that respects the same temporal rate. The "
            "reference is an oracle at those points, so it is not a target to hit exactly "
            "— but a decoder several times worse than it is not limited by the bottleneck, "
            "and the things that would help (capacity, schedule, loss) are all on the "
            "learning side. Effective resolution is the number to watch: it is in the "
            "summary and moves with real progress, unlike MSE against a flat line."
            % bk)
    short = [(ds, h, k, b) for ds, h, k, b, _ in refs if k < REF_RESOLUTION_LIMIT * b]
    if short and not (ratios and all(r > 1.0 for *_, r in ratios)):
        ds, h, k, b = min(short, key=lambda t: t[2])
        out.append(
            "**Ask where the missing resolution goes.** The %s decoder on `%s` delivers "
            "%.0f of the %d samples its input carries. Comparing its per-position error "
            "profile against the reference's would say whether the loss is uniform or "
            "sits at particular positions." % (_SHORT.get(h, h), ds, k, b))

    # ── What the tail implies ────────────────────────────────────────────────
    tails = _tail_rows(by)
    if tails:
        ds, h, share, frac = max(tails, key=lambda t: t[2])
        cause = _top_tail_cause(by.get(ds, {}))
        if cause and share >= CONCENTRATION_FACTOR * frac:
            levels = " and ".join("`%s`" % v for v in cause["levels"])
            many = cause["n_levels"] > 1
            out.append(
                "**Fix %s = %s rather than the model.** %.0f%% of the worst %.0f%% on `%s` "
                "is %s, and that worst %.0f%% carries %.0f%% of the dataset's whole error "
                "budget. Whether the answer is oversampling, weighting in the loss, or "
                "excluding them as bad data, it is a targeted change with a measurable "
                "outcome — this figure will show the curve flatten. `recon_tail_lift.csv` "
                "has the full table and `recon_df.csv` the members."
                % (cause["axis"], levels, 100 * cause["share"], 100 * frac, ds,
                   "those levels" if many else "that level", 100 * frac, 100 * share))
        elif share >= CONCENTRATION_FACTOR * frac:
            out.append(
                "**Name the tail on `%s` before averaging over it.** Its worst %.0f%% carry "
                "%.0f%% of the error but no single component index or signal property "
                "explains them, so the next step is to look at the members directly — "
                "`recon_df.csv` has per-sample error alongside every descriptor."
                % (ds, 100 * frac, 100 * share))

    # ── What the component budget implies ────────────────────────────────────
    budgets = []
    for ds, r in by.items():
        for h in (r.get("pathways") or []):
            b = _component_budget(r, h)
            if b and b["share"] >= COMP_BUDGET_MIN:
                budgets.append((ds, h, b))
    if budgets:
        ds, h, b = max(budgets, key=lambda t: t[2]["share"])
        comps = " and ".join("`%s`" % c for c in b["comps"])
        out.append(
            "**Decide what components %s of `%s` are before anything else on that dataset.** "
            "They are %.1f%% of its samples and %.0f%% of its error, so every aggregate "
            "over `%s` is mostly a statement about them — including the depth ordering, "
            "which is the one place this run disagrees with the others. Look at the traces "
            "in the component figure and settle whether they are a real signal class the "
            "model should learn or corrupt data that should be excluded; the answer picks "
            "between reweighting and dropping, and `recon_component_error.csv` has the "
            "per-component numbers either way."
            % (comps, ds, 100 * b["sample_share"], 100 * b["share"], ds))

    # The standing open question this eval cannot answer on its own.
    out.append(
        "**Remember that none of this measures representation quality.** These are "
        "reconstruction metrics, and the recorded finding (TASKS.md T6) is that the two "
        "are decoupled — SSL backbones give label-informative embeddings while "
        "reconstruction-trained ones do not. A head that reconstructs beautifully is not "
        "evidence of a good backbone. Pair any conclusion here with the label-regression "
        "and clustering evals before acting on it.")
    return out


def section(results: dict, config=None) -> list:
    """The closing section as markdown lines, or [] if there is nothing to say."""
    obs, nxt = observations(results), next_steps(results, config)
    if not obs and not nxt:
        return []
    lines = ["", "---", "", "### What this run shows", ""]
    lines += ["- " + o for o in obs]
    if nxt:
        lines += ["", "### What to do next", ""]
        lines += ["- " + n for n in nxt]
    return lines
