# Making the embedding beat raw input few-shot — design

**Date:** 2026-09-09
**Branch:** `eval/few-shot-readout-geometry` (from `label-regression-dev`)
**Ledger:** TASKS.md T13 follow-up ("step 6")
**Predecessor:** `plan.md` steps 1–5, `LABEL_REGRESSION_EVAL_SUMMARY.md`

---

## 1. The problem

The project's end goal is a foundation model whose embeddings, given a very
small labeled set, support a regressor that beats the same regressor trained
on the raw 245-point signal. Step 5 measured the opposite, decisively:

| n_train=50, PCA-50, median R² | raw input | post-transformer |
|---|---:|---:|
| 1-comp | 0.241 | 0.063 |
| 2-comp | 0.400 | 0.078 |
| 3-comp | **0.458** | **0.076** |

Raw input wins ~6× at every component count, across two independent
reductions (PCA-10 and PCA-50), so the gap is not an artifact of the
reduction — that hypothesis was tested and rejected in
`step5_tabpfn_pca50.md`. Meanwhile at n≈4,716 the embedding *wins* at 1-comp
(0.628 vs 0.468). The regimes genuinely invert.

This work attacks the few-shot regime only. The backbone is frozen — this is
an evaluation and readout question, not a training question.

## 2. Success criterion (pre-registered)

**The embedding-only pipeline must beat the raw-input-only pipeline** at
`n_train ∈ {20, 50}`, `n_comp ∈ {1, 2, 3}` — six cells — under an identical
probe family, identical training draws, and an evaluation set never used to
select either side.

Verdict rule, fixed before any step-6 number is measured:

- **WIN** — in ≥4 of 6 cells the paired median ΔR² (embedding − raw) is
  positive with ≥90% of draw-level bootstrap resamples positive, *and* the
  embedding's `frac_positive_r2` ≥ 0.8 in those cells.
- **PARTIAL** — the above holds in 2–3 cells.
- **NO** — otherwise. A NO is reported as a NO, with the diagnostic
  explanation of *why*, and is a legitimate outcome of this work.

Not in scope as a success route: a trained readout head, active/stratified
selection of which spectra to label, embedding+raw concatenation, any
non-frozen backbone. Each was considered and deliberately excluded.

## 3. Why the embedding might be losing — four hypotheses

Nothing below has been tested; step 5 varied only the probe, on one readout.

| # | Hypothesis | Readout axis that tests it |
|---|---|---|
| **H-A** | **Pooling.** Mean over 47 tokens discards *where* in the spectrum structure sits; raw input keeps position for free. | pooling ∈ {mean, mean_std, mean_max_min, segment4, first_last, mean_std_max_min} |
| **H-B** | **Layer.** Only the final layer was tapped. data2vec's last layer is fitted to a teacher target; intermediate layers are usually the stronger probes. | stage ∈ {fe, extract_features, layer0…layer12} |
| **H-C** | **Anisotropy.** Transformer embeddings carry a few enormous-variance rogue directions. PCA-on-unlabeled and ridge at n=20 both lock onto them. Consistent with the measured PCA-ceiling: k=10 retains 77% of raw's label signal but only 11% of the transformer's. | normalizer ∈ {none, standardize, l2, whiten} — all fitted on unlabeled data |
| **H-D** | **Probe.** Coefficient-fitting probes are hopeless at n=20 in 768 dims; the embedding's advantage, if any, is *geometry* — distances and a neighbourhood graph over the unlabeled corpus. Step 5 ran bare kNN only, and only on the disadvantaged readout. | probe ∈ kNN-cosine, GP-RBF, kernel ridge, graph label propagation, TabPFN |

H-C is the strongest prior: it explains the specific shape of the step-5
result (embedding fine at large n, collapses at small n) better than the
others, and it is the cheapest to fix.

## 4. Approach — screen cheap, confirm expensive

Three stages.

**Stage 1 — extract once.** One GPU pass over the 4,716 labeled spectra ×
components (0,1,2), capturing every transformer layer and both FE stages as
a compact **moment bank**: ten per-channel statistics over the 47 tokens
(mean, std, max, min, first, last, and four segment means). Every pooling in
H-A is a subset-concatenation of that bank, so the whole pooling axis is
explorable on CPU without re-extraction. ~3.1 GB fp16.

**Stage 2 — cheap screen, two waves.** Wave 1 fixes pooling=mean and sweeps
all 15 stages × 4 normalizers × 3 comp counts (180 readouts) to answer H-B
and H-C. Wave 2 takes the top 4 stages and sweeps all 6 poolings × 4
normalizers (288 readouts) to answer H-A. Each readout is scored with a fast
probe panel (ridge_strong, pca5_ridge, pca10_ridge, knn_cos5, gp_rbf) at
n_train ∈ {20, 50}, 30 draws, plus label-free and large-n **diagnostics**
(participation ratio, effective rank, large-n RidgeCV R², and large-n R² on
the top-k whitened PCs).

The diagnostics carry their own claim: **if the top-k whitened-PC ceiling
predicts few-shot R² across the 468 readouts, we have a cheap, label-light
criterion for readout quality that generalizes past this checkpoint.** The
Spearman correlation between the two is a reported result either way.

**Stage 3 — confirm the finalists.** Per comp count, the top 3 embedding
readouts and top 2 raw readouts (each selected by its own family's best
screen score) are re-scored with 100 paired draws and the expensive probe
panel, which adds **TabPFN** (on PCA-50 features, GPU, ~2.4 s/draw, ~2
GPU-hours for the confirm grid), kernel ridge, and graph label propagation.

## 5. Selection bias, and the eval split that guards it

468 readouts screened, then the best reported, is a selection-on-the-test-set
trap: the winner's screen score is optimistic by construction.

**Two disjoint evaluation sets.** From the 4,716 spectra, fixed by seed 42:
`eval_A` = 500 spectra (screening only), `eval_B` = 500 spectra (confirmation
only, never scored during screening), `pool` = 3,716 spectra from which all
training draws are taken. Screen ranks on eval_A; every reportable number
comes from eval_B. **Raw input goes through the identical select-then-confirm
procedure**, so whatever optimism survives is symmetric and cannot manufacture
an embedding win.

## 6. Symmetry, and where it honestly breaks

Every lever the embedding gets, raw input gets:

- **Shared:** normalizer (none/standardize/l2/whiten, all fitted on unlabeled
  data), reduction (PCA-k / whitened PCA-k), the entire probe panel including
  TabPFN and the geometry probes, the training draws (identical seeds → paired
  comparison), both evaluation sets.
- **Embedding-only:** layer and pooling. Raw input is a single-channel
  245-point sequence with no channel dimension to pool over, so the axis has
  no raw analogue. As partial compensation the raw family includes a
  `moments` readout (the same ten statistics computed on the raw signal), and
  the raw family carries both `raw` and `z-scored` variants, which step 5
  showed differ by up to 0.06 R².

That asymmetry favours the embedding and is stated in the report rather than
buried.

## 7. Protocol (inherited, unchanged)

From `plan.md` "Split design" and step 5 — these are settled and are not
re-litigated here:

- Repeated random subsampling, never k-fold, at these n. Report the
  distribution (median, IQR, p10/p90), never a point estimate.
- `frac_positive_r2` — fraction of draws beating the mean predictor — is
  reported beside every median. It is the number a deployment actually needs.
- Wide layout only (spectrum = one row, components concatenated). Components
  are not exchangeable; pooling them into one model collapses to R²≈0.006.
- comp20/comp21 dropped (exact duplicates of comp14/comp15).
- Transductive (unlabeled-corpus-fitted) and inductive (fitted inside the
  draw) variants are labeled and reported separately, never merged.
- Every fit runs under `threadpool_limits(1)` — this machine is
  oversubscribed (load ~20 on 40 cores) and unpinned small fits stall for
  seconds. The exception is the per-readout randomized SVD, a single large
  factorization that does benefit from threads.

## 8. Deliverables

1. `code/eval/label_probe/readouts.py` — moment-bank extraction + readout assembly.
2. `code/eval/label_probe/geometry.py` — unlabeled-corpus normalizers/reducers and the geometry probe family.
3. `code/eval/label_probe/screen.py` — diagnostics and the two screening waves.
4. `code/eval/label_probe/study6.py` — orchestration, confirmation, report.
5. `step6_REPORT.md` — the verdict against §2's rule, the per-cell paired table, which hypothesis (if any) explained the gap, and the diagnostic-vs-few-shot correlation.
6. `step6_readout_screen.png`, `step6_confirm.png`.
7. `TASKS.md` T13 updated; `LABEL_REGRESSION_EVAL_SUMMARY.md` §5 amended with the outcome.

## 9. Scope limits, named up front

- One checkpoint (Feb-25 SSL, `runai_long_train_2026-02-25_13-46-46.pt`).
  A win here is a claim about this backbone's readout, not about SSL in general.
- Components (0,1,2) only. The 7- and 12-comp ladders are not extracted; the
  deployment setting is 1–3 components and step 5 showed 12-comp has no
  headroom (linear already reaches 0.989).
- Random draws. Active/stratified label selection would likely beat this and
  remains the obvious follow-up.
- Distribution shift and calibration untested.
- A NO verdict does not prove no readout exists — it bounds what this
  readout family, on this backbone, achieves.
