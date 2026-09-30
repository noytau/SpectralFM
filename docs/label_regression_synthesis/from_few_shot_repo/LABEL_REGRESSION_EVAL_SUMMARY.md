# Label-Regression Evaluation, Rebuilt — Summary

Design and pre-implementation evidence: `plan.md` (repo root). Ledger: `TASKS.md` T13.
Per-step detail: `step{1,2,3,4,5}_REPORT.md` + the two supplements
(`step2_addendum_vs_best_linear.md`, `step4_deployment_axes.md`,
`step5_tabpfn_pca50.md`).

Checkpoint: Feb-25 SSL backbone (`runai_long_train_2026-02-25_13-46-46.pt`).
n=4,716 spectra, 12 unique components (comp20/comp21 dropped — exact
duplicates of comp14/comp15).

**Figures:** `label_reg_stages.png` (headline: R² across stages),
`label_reg_true_vs_pred_grid.png` (30-panel true-vs-pred, every stage ×
component count), `label_efficiency_{1,2,3,12}comp.png` (few-shot curves).

---

## 1. The historical headline claim was a measurement artifact

`TASKS.md` T6 / `ARCHITECTURE.md` say: Feb-25 SSL backbone emb R²=0.44, "beats
input baseline 0.38." Both numbers were produced with 1,000 of the 4,716
available spectra, unshuffled folds, and a single RidgeCV probe.

**The old pipeline was not buggy** — this run's `legacy` protocol reproduces
0.3772 / 0.7060 / 0.7568 and 0.4412 / 0.4754 / 0.5062 to four decimals. It was
*under-measured*. At full n with a probe panel:

| n-comp | best input | post-transformer | recovery |
|---:|---:|---:|---:|
| 1 | 0.468 | **0.628** | embedding *wins* (superseded — see below) |
| 2 | 0.831 | 0.757 | 91% |
| 3 | 0.867 | 0.793 | 91% |
| 7 | 0.935 | 0.866 | 93% |
| 12 | 0.989 | 0.939 | 95% |

The backbone preserves 91–95% of the linearly-recoverable label signal at
every count ≥2, and **beats raw input outright at 1-comp**. "Barely beats
input" was wrong in both directions: it understated the embedding *and* the
input.

**Deployment framing matters.** If the real setting is 1–3 components (as the
client indicates), the "95% of a 0.989 ceiling" line is the wrong number to
lead with — it is a 12-comp result. The deployment-relevant statement is the
1-comp one: **+0.16 R² over the best input probe (0.628 vs 0.468)**, which is
a far better argument for the backbone.

**Superseded by step 7 — the 0.468 raw figure above used a weak readout.**
This section's `0.468` is the best-input-probe number under `raw|z|-|none`
(unwhitened). Step 7 (§5 below) measured a whitened raw readout
(`raw|raw|-|whiten`) that was already at +0.582 at n=50 (step 6) and +0.853
at n=2000 — well past 0.468 — while the embedding side of this section
(`layer12`/mean) tracks toward step 1's own 0.628 as step 7's `historical`
column (+0.603 at n=2000). Both arms in this section's "embedding wins"
comparison were measured, correctly, at the readouts available when this
step was run; the *comparison* does not survive a fair readout on both
sides. The 0.468/0.628 numbers themselves are accurately reported and kept
here as history — the conclusion drawn from them ("embedding wins at
1-comp") is superseded. See §5's step 7 paragraph for the corrected
picture.

## 2. Systematics dominate; the metric was never noisy

| source of variation | magnitude |
|---|---:|
| probe choice (1-comp input) | **0.27 R²** (0.238 → 0.508) |
| sample size n (OLS, 2-comp) | 0.07 R² |
| normalization (2-comp) | 0.06–0.17 R² |
| split noise (repeated 5-fold) | **0.0005 R²** |

Split noise is three orders below the systematics. **This corrects an
assumption made early in the plan:** T8's A-vs-B at ΔR²≈0.03 was *not*
"decided inside the noise" — with resolvable ΔR²≈0.001 it sits ~30× above
split noise, and systematics cancel within one A/B at fixed probe/n/
normalization. The real hazard is comparing numbers measured under *different*
unstated settings (e.g. T8's 3c flip between two probes). The value of this
work is a **pinned protocol**, not error bars.

## 3. No exploitable non-linearity (large-n)

XGBoost loses to the best linear probe in **all 30 stage × component cells**,
by 0.165–0.410 R². Its apparent wins in `step2_REPORT.md` are an artifact of
comparing against an untuned Ridge(1.0) — see
`step2_addendum_vs_best_linear.md`, without which that report is misleading.

Caveat on strength of evidence: this shows *axis-aligned tree splits* don't
help, a weak inductive bias for smooth spectral data. It is weaker evidence
about models with different biases — and indeed TabPFN behaves differently in
the few-shot regime (§5).

## 4. Preprocessing and feature-vector recommendations

**Z-score at 1-comp, not at 2–3.** H2 flips sign:

| n-comp | 1 | 2 | 3 | 7 | 12 |
|---|---:|---:|---:|---:|---:|
| cost of z-scoring | **−0.060** | +0.074 | +0.053 | +0.011 | +0.009 |

Mechanism: z-scoring's damage is to *cross-component* scale ratios, which
don't exist at 1-comp — there it only improves conditioning.

**Use PLS-64, not concat, and never PCA at low k** (`step4_deployment_axes.md`):

| | PLS-64 | concat | gain |
|---|---:|---:|---:|
| 1-comp | **0.710** | 0.630 | +0.080 |
| 3-comp | **0.847** | 0.793 | +0.053 |

Supervised reduction beats unsupervised by 2–6× at matched k (3-comp, k≈5:
PLS 0.389 vs PCA 0.066). The label direction is **not** in the top principal
components.

**Two hypotheses of mine that failed:**
- *H8* — explicit cross-component features (per-component moments + pairwise
  diffs/log-energy ratios) recover only 0.398 of raw input's 0.863, and 0.057
  of the transformer's 0.793. The signal is distributed across the full
  spectral shape, not summarizable. My "cross-component ratios are the
  dominant carrier" framing was too strong.
- *Exchangeability over-generalized* — the R²≈0.006 collapse is for pooling
  **raw signal** into one shared model. Averaging **embeddings** across
  components gives 0.661 at 3-comp: lossy, not collapse.

## 5. The 20-label constraint (client's question)

**Conventional probes cannot do it.** At n_train=20, ridge / PCA+ridge / PLS /
kNN all fail to beat predicting the mean; `dummy` is frequently the *best*
median. Best non-TabPFN reliability anywhere: 34% of draws beating the mean.

**TabPFN changes the answer.** `tabpfn_pca10` is the best probe in every cell
it ran, and the only one that reliably beats the mean:

| config (n_train=20) | median R² | frac draws R²>0 |
|---|---:|---:|
| 2-comp z-scored input | **+0.133** | **0.90** |
| 3-comp raw input | +0.098 | 0.90 |
| 1-comp raw input | +0.080 | 0.75 |
| all non-TabPFN probes | −0.01 … −0.85 | 0.00–0.34 |

**Verdict for the client:** 20 labels is *marginally* viable — TabPFN on raw
2–3 component input, R²≈0.10–0.13, working ~90% of the time. It is **not**
viable with any conventional probe. But R²≈0.1 is a weak model; the
label-efficiency curves show ~50–100 labels is where this becomes real
(3-comp z-scored: 0.072 at n=20 → 0.284 at n=50 → 0.404 at n=100). The
actionable recommendation is to negotiate toward 50–100 labels, not 20.

**The embedding is genuinely worse few-shot — hypothesis tested and rejected.**
TabPFN looked good on input and poor on embeddings. The PCA-ceiling diagnostic
suggested a confound: PCA-10 retains only ~11% of the transformer's label
signal (ceiling 0.072 vs full-dim 0.628) versus 77% for raw input, so the
reduction rather than the representation might be at fault. That hypothesis
was tested directly at k=50 (`step5_tabpfn_pca50.md`), where the transformer
ceiling rises to 0.333 — a fair reduction. **It did not close the gap:**

| n_train=50, PCA-50 | raw input | post-transformer |
|---|---:|---:|
| 1-comp | 0.241 | 0.063 |
| 2-comp | 0.400 | 0.078 |
| 3-comp | **0.458** | **0.076** |

Raw input beats the embedding ~6× few-shot, at every component count, across
two independent reductions. At the time this was written the large-n and
few-shot regimes looked like they **inverted**: at n≈4,716 the embedding
wins at 1-comp (0.628 vs 0.468); at n≤50 raw input wins everywhere,
decisively. **Step 7 (below) shows this is not an inversion — it is one
under-measured raw baseline.** The 0.468 anchor used `raw|z|-|none`; a
whitened raw readout (`raw|raw|-|whiten`) was already ahead of it at n=50
(step 6: 0.582) and reaches 0.853 by n=2000, while the embedding side
(`layer12`/mean) tracks toward the same 0.628 as it always did (step 7's
`historical` column: 0.603 at n=2000, still climbing). There is no
crossover to find between the regimes because both regimes were always won
by raw input once raw gets a comparably good readout; the few-shot finding
below is real, but the large-n "embedding wins at 1-comp" comparison that
motivated calling it a genuine inversion was itself an artifact of the same
class this whole document opened by diagnosing in the *original* T6 claim
— an under-powered baseline, one level further in. What *is* real and
label-*efficiency*-specific (not just readout-specific) is developed in the
step 7 paragraph below.

**Step 6 swept the readout, not just the probe — the conclusion holds, but
the readout hypotheses were partly vindicated.** One possible objection to
§5 was that layer12/mean/no-normalization (the one readout tested) was an
unlucky choice, and a different layer, pooling, or unlabeled-data
normalization might close the gap. Step 6 (`code/eval/label_probe/`, report
at `code/eval_outputs/label_probe/step6/step6_REPORT.md`) tested that
directly: it screened 456 embedding readouts — 4 transformer layers × 6
pooling schemes (mean, mean_std, mean_std_max_min, mean_max_min,
first_last, segment4) × 4 unlabeled-fit normalizers (none, standardize, l2,
whiten) — against the matched raw-input family at n_train ∈ {20, 50} and
n_comp ∈ {1, 2, 3}, pre-registering a verdict rule before looking at the
held-out `eval_b` numbers. **Verdict: NO, 0/6 cells.** The best embedding
readout at every (n_comp, n_train) cell still lost to raw input, by ΔR² of
-0.24 to -0.49, with bootstrap P(Δ>0)=0.00 in all six cells. But the layer
and pooling hypotheses were not idle: `layer12` — the only depth §5 ever
tapped — never placed among the best stages at any component count (best
stages were `layer0`/`layer1`/`layer3`/`fe`), and the best embedding readout
at 1-comp/n=50 roughly doubled §5's own number on the same comparison
(post-transformer PCA-50, 0.063 → this readout, 0.119, both measured
out-of-sample). Raw input improved by more over the same span (§5's
PCA-50 raw 0.241 → step 6's best raw readout at 1-comp/n=50, 0.582, using a
normalizer/probe combination §5 didn't have access to) — both sides got
better, raw got better faster, and the gap widened rather than closed.
Two geometry/full-data diagnostics were also tested as cheap proxies for
few-shot performance: full-data `ridgecv_full_r2` correlated only weakly
with the few-shot screen score (Spearman ρ=+0.338), and the geometry probes
even more weakly (`effective_rank` +0.142, `participation_ratio` +0.109).
The diagnostic the design spec specifically predicted would matter,
`whitened_topk_r2`, was initially miscounted as similarly weak due to a
reporting bug (`report6.py`'s diagnostic lookup silently dropped all four
`whitened_topk_r2_*` rows on a str/int key mismatch after the JSON
round-trip — a real defect, flagged rather than patched here since fixing
eval code is out of scope for this task). Recomputed correctly outside the
module against the same screen data, `whitened_topk_r2` is in fact the
*strongest* diagnostic of the seven, ρ≈0.58–0.63 across its four top-k
settings — so that specific hypothesis is **not** refuted. Caveat:
`whitened_topk_r2` and `ridgecv_full_r2` are both large-n RidgeCV fits
against all 4,716 labels (both evaluation sets included) — cheap to
compute, but not label-light, so this is not a transferable cheap-screening
result; the diagnostics that are genuinely label-free (`participation_ratio`
+0.109, `effective_rank` +0.142) are the weak ones. It does not,
however, change the headline: predicting which embedding readout will be
the *best among embeddings* is a different claim from that readout beating
raw input, and on the confirm grid it never did. Varying layer, pooling,
and unlabeled normalization, and checking three geometry/asymptotic-quality
proxies, does not change the §5 conclusion: at n_train ≤ 50 this backbone's
embeddings do not beat raw input, under any readout tested so far.

**Step 7 laddered the label count itself, at the best readout found so
far, restricted to 1-comp (the only component count where a crossing was
thought possible per step 1's large-n anchors) — no crossing found up to
n = 2000, and the study's own premise turns out to be an artifact.**
`code/eval/label_probe/` (`step7`), report at
`code/eval_outputs/label_probe/step7/step7_REPORT.md`. At each rung
{100, 200, 500, 1000, 2000} the embedding readout and probe are reselected
on a disjoint `eval_a` split and scored on `eval_b`, alongside a matched raw
readout and the historical `layer12`/mean readout. Raw input leads at every
rung: ΔR² = -0.487 (n=100), -0.341 (n=200), -0.226 (n=500), -0.151
(n=1000), -0.113 (n=2000) — monotonically narrowing but never crossing
zero, and bootstrap P(Δ>0)=0.00 at every rung. The estimator reports
`crossed=False`, `n_cross_interp=None`, no CI: there is no sign change in
the measured range to interpolate through.

**There is no crossing to find because the "bracket" motivating this study
was never real.** §1/§3 argued step 6 (raw wins at n≤50) and step 1
(embedding 0.628 > raw 0.468 at n≈3,773) bracket a crossing somewhere
between. They don't: step 1's raw figure is `raw|z|-|none`, an unwhitened
readout, while step 7's raw arm (`raw|raw|-|whiten`) reaches 0.853 by
n=2000 — already well past 0.628 with room to spare. Give both arms a good
readout and raw leads at every n measured, from 20 to 2000. Two pieces of
corroboration were already in our own data and went unacted on: (a) step
7's `historical` arm *is* step 1's embedding readout (`layer12`/mean) and
reaches +0.603 at n=2000, heading toward step 1's 0.628 — the pipeline
reproduces step 1's embedding side faithfully, so it is specifically the
raw side of that comparison that was weak; (b) step 6 had already measured
`raw|raw|-|whiten` at +0.582 at n=50 — raw with good levers, at *fifty*
labels, already beating step 1's raw figure at n≈3,773. That should have
been a red flag against the "genuine inversion" framing at the time; it
wasn't caught until this run.

**The narrowing is real; it does not rescue the conclusion.** The gap
shrinks monotonically from n=200 onward: -0.487, -0.341, -0.226, -0.151,
-0.113. Extrapolating that trend, closure — if it happens at all — lies
well beyond the 3,716-spectrum labeled pool this study draws from, so
"collect more labels" is not a fix reachable with the data on hand. The
estimator deliberately does not extrapolate to a specific crossing n, and
this write-up doesn't either.

**The historical-readout comparison is a result in its own right.** It
quantifies what readout choice alone is worth at each rung: `historical`
(`layer12`/mean) vs the reselected embedding readout goes 0.092 vs 0.268 at
n=100 (2.9×) narrowing to 0.603 vs 0.739 at n=2000 (1.2×). Readout choice
matters enormously few-shot and progressively less as n grows toward the
large-n regime — which is also why prior large-n work at `layer12`/mean
was less wrong than prior few-shot work at the same tap.

**Bottom line: this data supports the representation-is-the-binding-
constraint framing, not the 20-label-constraint framing**, at
1-component — not because a crossing was found and shown to lie beyond
reach, but because the "crossing exists somewhere" premise itself
depended on an under-measured raw baseline at the large-n anchor. Once
raw gets a fair readout it leads from 20 labels all the way to 2000; the
20-label constraint was never the sole bottleneck this framing assumed.
Whether a crossing exists at all above 2000 labels remains genuinely open
— outside the range measured here — but there is no longer a documented
reason to expect one.

## Recommendations

1. **Update T6 / ARCHITECTURE.md.** Replace "emb R²=0.44 beats input 0.38"
   with the full-n, probe-panel numbers, and lead with the 1-comp result if
   deployment is 1–3 components.
2. **Re-measure the recon-trained backbones under this protocol.** Their
   "≈0" scores are from the same under-powered protocol that understated this
   backbone by 0.2–0.3 R². They are on the RunAI PVC, not Geoffrey, so this
   was out of scope here — but until re-run, the T6 *ranking* is unverified.
3. **Pin one protocol before comparing checkpoints.** Probe choice alone moves
   R² by 0.27; cross-measurement comparisons are the real hazard.
4. **Adopt PLS-64 + component-count-dependent normalization** as the default
   feature vector *for the large-n regime*. In the few-shot regime the
   recommendation inverts: use **raw input**, not the embedding — it wins ~6x
   at n<=50 (§5).
5. **Fold the orphaned 2026-08-21 numbers into TASKS.md** (currently only in a
   YAML comment).

## Scope / not done

- Only the Feb-25 SSL backbone. No recon-trained checkpoints (PVC-only).
- H4 (pooling schemes) is now partly addressed: step 6 swept 6 fixed pooling
  schemes (mean, mean_std, mean_std_max_min, mean_max_min, first_last,
  segment4) across 4 layers and 4 normalizers and found no combination beats
  raw input at n_train ≤ 50 (verdict NO, 0/6 cells). The trained
  attention-pooling head is **still not run**.
- TabPFN excluded at 12-comp by compute budget (~29 GPU-hours for the full
  grid); it therefore cannot be compared there.
- Few-shot draws are *random*; stratified/active selection of which 20 spectra
  to label would likely beat this and is the obvious follow-up. **Still not
  done** after step 7 — the label-efficiency curve at a good, per-rung-
  reselected readout is now measured (1-comp, n=100..2000, step 7), but every
  draw at every rung remains an i.i.d. random draw from the pool.
- 2-comp and 3-comp were laddered only at n=20/50 (step 6); step 7 restricted
  the n=100..2000 ladder to 1-comp on the strength of step 1's large-n
  anchors (embedding leads only at 1-comp there). Whether 2-/3-comp behave
  differently across 100-2000 labels is untested.
- Distribution shift and calibration untested.
