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

## 1. The historical headline claim was a measurement artifact — corrected again, 2026-09-10

> **UPDATE 2026-09-10:** the table and "embedding wins at 1-comp" conclusion
> below are themselves superseded. The original probe panel (OLS, Ridge,
> RidgeCV) never included PLS, and raw input carries a real signal that only
> a covariance-seeking reducer can reach — RidgeCV's isotropic shrinkage
> structurally kills it. See `step6_pls_correction.md` (2026-09-10 addendum)
> for the full account. Corrected numbers:
>
> | n-comp | raw input (PLS, corrected) | post-transformer | recovery |
> |---:|---:|---:|---:|
> | 1 | **0.825** | 0.645 | 78% |
> | 2 | **0.939** | 0.763 | 81% |
> | 3 | **0.969** | 0.793 | 82% |
> | 7 | **0.993** | 0.866 | 87% |
> | 12 | **0.999** | 0.945 | 95% |
>
> **There is no component count at which the backbone beats correctly-probed
> raw input.** Recovery rises monotonically from 78% to 95% with component
> count — worst exactly at 1-comp, the deployment-relevant regime. The "+0.16
> win at 1-comp" below was raw input's ceiling being underestimated by
> RidgeCV, not a real backbone property. `input(z)`/FE/proj columns are
> spot-checked at 1 and 3 comp only and still need a full re-sweep.

`TASKS.md` T6 / `ARCHITECTURE.md` say: Feb-25 SSL backbone emb R²=0.44, "beats
input baseline 0.38." Both numbers were produced with 1,000 of the 4,716
available spectra, unshuffled folds, and a single RidgeCV probe.

**The old pipeline was not buggy** — this run's `legacy` protocol reproduces
0.3772 / 0.7060 / 0.7568 and 0.4412 / 0.4754 / 0.5062 to four decimals. It was
*under-measured*. At full n with the original (pre-PLS) probe panel:

| n-comp | best input | post-transformer | recovery |
|---:|---:|---:|---:|
| 1 | 0.468 | **0.628** | ~~embedding *wins*~~ superseded above |
| 2 | 0.831 | 0.757 | ~~91%~~ 81% |
| 3 | 0.867 | 0.793 | ~~91%~~ 82% |
| 7 | 0.935 | 0.866 | ~~93%~~ 87% |
| 12 | 0.989 | 0.939 | ~~95%~~ 95% |

This table is kept for the record; use the corrected one at the top of this
section.

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

**~~Use PLS-64, not concat~~ — RETRACTED. Plain concatenation is fine.**

The published version of this section claimed PLS-64 beat concat by +0.080
(1-comp) / +0.053 (3-comp). Those numbers came from a **label leak**: step 4
fitted the supervised PLS projection with `fit_transform(X, y)` on the whole
dataset and only then cross-validated, so the projection had seen the test
labels. Fitting inside the training fold (`step6_pls_correction.md`):

| | honest PLS peak | plain concat | verdict |
|---|---:|---:|---|
| 1-comp | 0.652 (k=96) | 0.630 | +0.022 — marginal |
| 3-comp | 0.789 (k=96–128) | 0.793 | −0.004 — **no gain** |

Leak size was +0.083 to +0.090 R² at k=64, growing with k. The extended sweep
also answers the original question: the optimum is k≈96–128, not 64 (64 was
simply the largest value swept) — but even at its optimum PLS only ties
concatenation at 3 components.

PCA is unsupervised and did **not** leak, so its numbers stand. The
supervised-beats-unsupervised claim survives but narrower: PLS dominates at
small k (1-comp k=16: 0.378 vs 0.204) and the two converge by k=256
(0.562 vs 0.576).

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
two independent reductions. So the large-n and few-shot regimes genuinely
**invert**: at n≈4,716 the embedding wins at 1-comp (0.628 vs 0.468); at
n≤50 raw input wins everywhere, decisively. A representation can be more
label-informative in aggregate and simultaneously less label-*efficient* —
which is exactly the distinction the client's constraint was probing, and it
would have been invisible from steps 1-4.

## Recommendations

1. **Update T6 / ARCHITECTURE.md.** Replace "emb R²=0.44 beats input 0.38"
   with the corrected §1 numbers — raw input (PLS) beats the embedding at
   every component count, most sharply as a recovery fraction at 1-comp
   (78%). Do not lead with the old "1-comp win" framing; it does not survive.
2. **Re-measure the recon-trained backbones under this protocol.** Their
   "≈0" scores are from the same under-powered protocol that understated this
   backbone by 0.2–0.3 R². They are on the RunAI PVC, not Geoffrey, so this
   was out of scope here — but until re-run, the T6 *ranking* is unverified.
3. **Pin one protocol before comparing checkpoints.** Probe choice alone moves
   R² by 0.27; cross-measurement comparisons are the real hazard.
4. **Adopt component-count-dependent normalization** (z-score at 1-comp, not
   at 2-3) as the default. **On the embedding**, do not adopt PLS: the gain
   that motivated it was a label leak, and honestly measured it is +0.022 at
   1-comp and nil at 3-comp versus plain concatenation. **On raw input**, PLS
   is not optional — it is the only probe in this study that reaches raw
   input's true ceiling (0.825–0.999 vs RidgeCV's 0.409–0.989), and the
   corrected §1 table shows raw input beats the embedding at every component
   count once probed this way. In the few-shot regime the recommendation also
   favors **raw input**, not the embedding — it wins ~6x at n<=50 (§5).
5. **Fold the orphaned 2026-08-21 numbers into TASKS.md** (currently only in a
   YAML comment).

## Scope / not done

- Only the Feb-25 SSL backbone. No recon-trained checkpoints (PVC-only).
- H4 (pooling schemes) and the trained attention-pooling head not run — both
  need re-extraction at other poolings.
- TabPFN excluded at 12-comp by compute budget (~29 GPU-hours for the full
  grid); it therefore cannot be compared there.
- Few-shot draws are *random*; stratified/active selection of which 20 spectra
  to label would likely beat this and is the obvious follow-up.
- Distribution shift and calibration untested.
