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
    if dead.empty:
        worst = S.loc[S["r2_median"].idxmin()]
        out.append(
            "**Every head is doing real work.** The weakest combination is the "
            "%s decoder on `%s`, and it still reaches median R² = %.2f against the "
            "flat-line baseline, beating it on %.0f%% of samples. Nothing here is the "
            "degenerate mean-predictor."
            % (_SHORT.get(worst["head"], worst["head"]), worst["dataset"],
               worst["r2_median"], 100 * worst["frac_r2_positive"]))
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
    return out


def next_steps(results: dict) -> list:
    """
    What to do next, conditioned on what this run actually showed.

    Only steps the numbers motivate. A checklist that would read the same whatever the
    results were is not worth a section.
    """
    from .report import _recon_keys, _split_eval_key

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

    # Scale: say plainly whether this run was big enough to trust.
    n_by_ds = {ds: int(S[S["dataset"] == ds]["n"].max()) for ds in S["dataset"].unique()}
    thin = {ds: n for ds, n in n_by_ds.items() if n < 2000}
    if thin:
        out.append(
            "**Re-run the thin datasets at full size.** %s were evaluated on under 2000 "
            "samples, which is fine for medians but leaves the stratified panels with a "
            "few hundred samples per bin and the tail poorly estimated. "
            "`--eval_set_size all --recon_max_samples 0` covers whole splits."
            % ", ".join("`%s` (n=%d)" % (d, n) for d, n in sorted(thin.items())))

    # The standing open question this eval cannot answer on its own.
    out.append(
        "**Remember that none of this measures representation quality.** These are "
        "reconstruction metrics, and the recorded finding (TASKS.md T6) is that the two "
        "are decoupled — SSL backbones give label-informative embeddings while "
        "reconstruction-trained ones do not. A head that reconstructs beautifully is not "
        "evidence of a good backbone. Pair any conclusion here with the label-regression "
        "and clustering evals before acting on it.")
    return out


def section(results: dict) -> list:
    """The closing section as markdown lines, or [] if there is nothing to say."""
    obs, nxt = observations(results), next_steps(results)
    if not obs and not nxt:
        return []
    lines = ["", "---", "", "### What this run shows", ""]
    lines += ["- " + o for o in obs]
    if nxt:
        lines += ["", "### What to do next", ""]
        lines += ["- " + n for n in nxt]
    return lines
