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
| 1 | 0.468 | **0.628** | embedding *wins* |
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
two independent reductions. So the large-n and few-shot regimes genuinely
**invert**: at n≈4,716 the embedding wins at 1-comp (0.628 vs 0.468); at
n≤50 raw input wins everywhere, decisively. A representation can be more
label-informative in aggregate and simultaneously less label-*efficient* —
which is exactly the distinction the client's constraint was probing, and it
would have been invisible from steps 1-4.

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
- H4 (pooling schemes) and the trained attention-pooling head not run — both
  need re-extraction at other poolings.
- TabPFN excluded at 12-comp by compute budget (~29 GPU-hours for the full
  grid); it therefore cannot be compared there.
- Few-shot draws are *random*; stratified/active selection of which 20 spectra
  to label would likely beat this and is the obvious follow-up.
- Distribution shift and calibration untested.
