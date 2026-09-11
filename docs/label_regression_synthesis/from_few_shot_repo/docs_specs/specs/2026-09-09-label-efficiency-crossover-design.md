# Where does the embedding overtake raw input? — design

**Date:** 2026-09-09
**Branch:** `eval/label-efficiency-crossover` (from `label-regression-dev`)
**Ledger:** TASKS.md T15 ("step 7")
**Predecessors:** step 6 (`docs/superpowers/specs/2026-09-09-few-shot-readout-geometry-design.md`), steps 1–5 (`plan.md`)

---

## 1. The question, and why it is now the right one

Step 6 settled that no frozen readout beats raw input at 20–50 labels: 0/6 cells,
bootstrap P(Δ>0) = 0.00 everywhere. But step 1 measured the opposite at the other
end of the ladder — at n ≈ 4,716 (pooled 5-fold, so ~3,773 training spectra) the
embedding **wins** at 1-comp, 0.628 versus raw's 0.468.

Both results are sound and they do not conflict; they bracket a crossing. Nobody
has measured where it is, because:

- Step 5 swept n ∈ {10…100} but only at `layer12`/mean — which step 6 showed is
  among the *worst* taps, costing a factor of ~2. Its curves therefore understate
  the embedding everywhere.
- Step 6 swept 456 readouts but only at n ∈ {20, 50}.

So the label-efficiency curve has never been measured at a *good* readout. The
crossing point is the number that converts a scientific finding into a
negotiable engineering requirement: "collect N labels" is a conversation, where
"20 labels, take it or leave it" was a wall.

**This study locates the crossing and puts a confidence interval on it.**

## 2. Pre-registered claim and estimator

Fixed before any number is measured, exactly as step 6's rule was.

**The ladder.** `n_train ∈ {20, 50, 100, 200, 500, 1000, 2000}`, on step 6's
identical split (`make_split(4716, seed=42)`: `eval_a` 500, `eval_b` 500, `pool`
3,716). Numbers are reported on `eval_b`; readout selection happens on `eval_a`.

**A rung is WON** when the paired median Δ = median(emb − raw) over the shared
draws is > 0 *and* ≥ 90% of draw-level bootstrap resamples of that paired delta
are > 0. Identical to step 6's cell rule minus the `frac_positive_r2 ≥ 0.8`
reliability clause, which is a deployment criterion rather than a crossing
criterion and is reported alongside instead.

**The crossover `n_cross`** is the smallest won rung such that every larger rung
on the ladder is also won. Requiring the win to *persist* is what distinguishes a
crossing from a single noisy rung.

**Point estimate.** Log-linear interpolation of the median-Δ curve between the
last lost rung and the first won rung, solving for Δ = 0. Reported as
`n_cross_interp`, always alongside the bracketing rungs, never on its own.

**Interval.** Bootstrap the per-draw deltas at the two bracketing rungs, re-solve
the interpolation on each resample, and report the 5th/95th percentiles.

**If no rung is won**, the study reports `crossed: false` with the largest median
Δ observed and the rung it occurred at. That is a legitimate outcome and would
mean the crossing lies above n = 2,000.

## 3. Three design decisions, and their costs

**(a) The ladder stops at 2,000, not 3,716.** Draws are without replacement from
a 3,716-spectrum pool, so two draws at n = 2,000 share ~54% of their rows. Above
n ≈ 1,000 the draws are no longer close to independent and their spread
**understates** true sampling variance — so bootstrap intervals at the top two
rungs are optimistically narrow, and the report must say so at the point of use.
Step 1's n ≈ 3,773 result (0.628 vs 0.468, pooled 5-fold) is carried as an
**anchor** on the figure rather than a rung, since it uses a different protocol.
Together the ladder and the anchor bracket the crossing.

**(b) The probe panel must depend on n, and this is not a convenience.**
`ridge_strong` deliberately biases its alpha grid high (`logspace(0, 6)`) because
that is correct at n = 20; at n = 2,000 it is over-regularized and would
understate *both* arms. Exact GP and the `krr_rbf` grid search are O(n³) and
O(30 · n³ / 3) and become infeasible past a few hundred. So:

| n | panel |
|---|---|
| ≤ 50 | step 6's `CONFIRM_PANEL` verbatim, so the bottom rungs reproduce step 6 |
| 100–500 | drop `tabpfn_pca50` (past its regime, GPU-bound); add `ridgecv`, `pls64` |
| ≥ 1000 | drop `gp_rbf`, `krr_rbf` (cubic); keep `ridgecv`, `pls64`, `pca{k}_ridge`, `knn_cos*`, `graph_prop*` |

Both arms always get the identical panel at a given rung. A panel that changes
with n is a confound *between* rungs — that is accepted and disclosed, because
the alternative (one panel everywhere) is a worse confound: it would measure
regularization mismatch rather than label efficiency.

**(c) The readout is re-selected at every rung.** Step 6's winners were selected
at n ∈ {20, 50}; the best readout at n = 2,000 may differ, and freezing the
small-n winner would under-measure both arms at the top. At each rung all
candidates are scored on `eval_a`, the best per arm is chosen, and only those two
are scored on `eval_b`. Same select-then-confirm discipline as step 6, extended
along n. Candidates are step 6's top 6 embedding and top 4 raw readouts per
component count, plus `layer12|mean|none` as a **historical reference arm** —
which quantifies, at every n, how much of the story was the readout choice rather
than the regime.

## 4. Symmetry and what is not symmetric

Unchanged from step 6: both arms share the split, the draws (paired), the panel
at each rung, and the selection procedure. Layer and pooling remain
embedding-only because raw input has no channel axis to pool; the raw arm gets
the normalizer, reduction and probe axes instead, and receives 4 candidates to
the embedding's 6 — an asymmetry that favours the embedding and so cannot
manufacture a *late* crossing.

## 5. Deliverables

1. `code/eval/label_probe/crossover.py` — ladder, n-dependent panels, candidate sets, the estimator, the driver.
2. `code/eval/label_probe/report7.py` — report and the label-efficiency figure.
3. One small additive change to `screen.score_readout`: an optional `probe_factory`.
4. `step7_REPORT.md` — the crossing per component count with its interval, the curve, the historical arm, and the disclosures from §3.
5. `step7_crossover.png` — median Δ and both arms' curves against n (log x), with the step-1 anchor.
6. `TASKS.md` T15; `LABEL_REGRESSION_EVAL_SUMMARY.md` §5 amended with the crossing.

## 6. Scope limits, named up front

- One checkpoint (Feb-25 SSL), target `parameter_0`, components 0–2, frozen backbone.
- Random draws. Active/stratified selection is a separate lever (and a separate study).
- Draws at n ≥ 1,000 overlap heavily — see §3(a).
- The panel varies across rungs — see §3(b).
- `parameter_0` is a 168-level designed grid, near-uniform, pre-standardized
  (mean 0.000, std 0.9999, range ±1.77σ). R² is therefore measured against the
  most favourable target variance it can have; a peaked deployment distribution
  would report lower R² for identical absolute error. The crossing *n* is more
  transportable than the R² values either side of it.
- 100% of `eval_b` labels also occur in the pool, and a 20-draw exactly matches
  the label of ~15% of `eval_b` rows (~33% at n = 50) — the discrete grid admits
  exact-label overlap. This inflates both arms and is not corrected here; step 1's
  `run_label_group` guard found it clean at large n. Reported as a caveat.
