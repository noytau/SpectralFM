# SpectralFM-few-shot: label-regression / few-shot-probe study — extraction before deletion

Source repo (about to be deleted): `/mnt5/home/hadar/nova/SpectralFM-few-shot`
Transcript analyzed: `/home/hadar/.claude/projects/-mnt5-home-hadar-nova-SpectralFM-few-shot/3f1d465e-2c1b-4fbb-aa8b-a1becdef4194.jsonl` (6008 lines)

**IMPORTANT — uncommitted final work rescued from /tmp.** The single most
current and well-organized synthesis of this whole study was produced in this
session *after* the last git commit (`c8354df`, 2026-09-10 00:25) and was
**never committed to the repo**. It lived only in the session's scratchpad at
`/tmp/claude-1040/-mnt5-home-hadar-nova-SpectralFM-few-shot/3f1d465e-2c1b-4fbb-aa8b-a1becdef4194/scratchpad/`.
I copied the key files out to safety at:
`/tmp/claude-1040/-mnt5-home-hadar-nova-SpectralFM-label-regression-eval-merged/a6cc197b-a5b5-4a71-b468-484689a92b79/scratchpad/fewshot_rescue/`
— most importantly **`label_regression_findings_v5.html`** (self-contained,
images embedded, 3.7MB) and its source **`body_v5.html`** (readable prose,
25KB, no images). **This should be treated as the single best deliverable of
the entire study** and copied into whatever new repo replaces this one. See
§0 below for its content in full; §1–§6 below cover the committed repo.

---

## 0. The rescued, uncommitted final synthesis (read this first)

`body_v5.html` (full text captured; reproduced in substance below) states a
**stronger and cleaner verdict** than anything committed to git:

> **NO — raw input beats the frozen embedding at every label budget measured,
> n=10 through the entire 3,716-spectrum training pool, across every one of
> 456 extraction recipes searched. There is no n at which the embedding
> wins.** The binding constraint is the representation, not the label
> budget.

Key numbers from its full 11-rung label-efficiency table (1-component,
median held-out R² on `eval_b`, one identical protocol throughout — this
table is **not** in any committed file; it comes from an uncommitted script
`measure_full_crossover.py`, log `measure_full_crossover.log`, and data
`full_crossover_measured.json`, all rescued to `fewshot_rescue/`):

| n_train | draws | raw (whitened) | embedding (whitened) | embedding (historical tap) | raw (unwhitened) | raw z-scored (unwhitened) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 300 | **+0.049** | −0.003 | −0.044 | −0.023 | −0.047 |
| 15 | 300 | **+0.131** | −0.003 | −0.018 | −0.013 | −0.042 |
| 20 | 300 | **+0.249** | +0.000 | −0.018 | −0.012 | −0.018 |
| 30 | 300 | **+0.402** | +0.049 | −0.012 | −0.010 | −0.006 |
| 50 | 300 | **+0.587** | +0.120 | −0.004 | +0.093 | +0.096 |
| 100 | 200 | **+0.761** | +0.280 | +0.109 | +0.498 | +0.268 |
| 200 | 150 | **+0.825** | +0.461 | +0.242 | +0.346 | +0.309 |
| 500 | 100 | **+0.877** | +0.650 | +0.366 | +0.701 | +0.601 |
| 1,000 | 60 | **+0.848** | +0.692 | +0.503 | +0.810 | +0.732 |
| 2,000 | 40 | **+0.853** | +0.738 | +0.596 | +0.841 | +0.769 |
| 3,716 (full pool) | 1 | **+0.856** | +0.755 | +0.654 | +0.853 | +0.786 |

Whitened raw leads at **every single rung** — this is the direct answer to
the "is there a crossover?" question the whole study chased across steps
5–7: **no, not anywhere from n=10 to n=3,716.**

**Mechanism finding (the report's own explanation):** the signal is present
in the embedding but "spread across many collinear directions"; raw input's
245 dimensions are physically meaningful and already concentrate the signal.
**Whitening — decorrelating/rescaling on the unlabeled corpus, fit with zero
labels — is "the single biggest lever measured anywhere in this study,"**
worth **+0.47 R²** at 1-comp/n=50 alone (unwhitened raw 0.10 → whitened raw
0.59, same n, same protocol). It helps raw more than it helps the embedding
because raw had more untapped structure to begin with.

**The historical extraction tap was wrong the whole time.** Every published
number before this study used the final transformer layer, mean-pooled
(`layer12`/mean). Step 6's screen found layers 0–3 are ~2–2.5× stronger, and
in the v5 ladder the historical tap is **the weakest recipe measured at
every n** — even the two long-superseded unwhitened-raw baselines beat it.
This "changes the measurement basis for every embedding claim on this
backbone, including the checkpoint ranking that guides the training
programme."

**PLS/float64 finding, resolved (see §3 for full detail):** a parallel
study's PLS-based ceiling for raw input (0.825 at n=4,716, 1-comp) is real
and reproducible, but checked directly here at small n with a shuffled-label
canary (always cleanly negative, e.g. −0.37 to −0.70), **PLS loses to this
study's existing whitened-raw best-of-panel at every n≤50** (e.g. n=50: PLS
+0.377 vs whitened-raw +0.587). **Verdict: the float64/PLS ceiling lift is
real but is a large-n phenomenon that does not transfer to the few-shot
regime.** It does not change the recommendation to ship whitening.

**Recommendations in v5 (supersedes the committed summary's "Recommendations" section):**
1. Re-measure the reconstruction-trained checkpoints under the
   whitened-recipe protocol — their existing scores use the now-known-worst
   `layer12`/mean tap, so the checkpoint ranking guiding the training
   programme is unverified against a fair recipe.
2. **Ship whitening** as part of the deployed pipeline (fit label-free,
   costs nothing extra) — not PLS, which is not a substitute at few-shot n.
3. Test active/stratified label selection (still untested; every draw in
   the whole study was i.i.d. random).
4. Re-aim the foundation-model demonstration at a task where raw features
   actually fail (multi-component disentangling, anomaly detection,
   cross-instrument transfer) — "a 245-point smooth 1-D signal with a smooth
   physical target is close to the worst case for demonstrating
   foundation-model value... that is a property of this task, not evidence
   the backbone failed at what it was trained to do."

The full HTML (`label_regression_findings_v5.html`, in `fewshot_rescue/`)
also embeds updated versions of the true-vs-pred grid (axis-fixed to
[-2, 2], restricted to 1–3 comp × 5 stages, 15 panels — see §5), the step6
confirm/screen bar and line charts, a whitening schematic (2-D toy
before/after), and the full n=10→3,716 label-efficiency chart with real IQR
error bars throughout (`step7_v3_real.png`, also rescued).

---

## 1. What was studied

**Goal:** whether a frozen embedding from the Feb-25 SSL data2vec-audio
backbone (`runai_long_train_2026-02-25_13-46-46.pt`) supports **few-shot
linear/nonlinear probing of a scalar label** (`parameter_0` in the labeled
NOVA dataset), well enough to be useful as a foundation-model probe, versus
just regressing on the raw 245-point spectrum directly.

- **Data:** `labeled_data` subset, n=4,716 usable labeled spectra (12 unique
  components; comp20/comp21 dropped as exact duplicates of comp14/comp15).
- **Methodology across 7 "steps"** (each a `code/eval/label_probe/` module +
  a `stepN_REPORT.md`):
  - **Step 1** (`study.py:run_step1`): full-n (4,716) probe panel
    (OLS/Ridge/RidgeCV) across 6 forward-pass stages (`input_raw`,
    `input_z`, `fe`, `extract_features`/post-LN, `proj`, `transformer`) ×
    5 component counts (1,2,3,7,12). This re-derived and *corrected* the
    historical "backbone barely beats input" T6 claim, which had used only
    n=1,000 and a single RidgeCV probe.
  - **Step 2–3**: nonlinear probes (XGBoost) vs. linear, preprocessing
    ablations (z-scoring, H2/H8 hypotheses on cross-component features).
  - **Step 4**: feature-vector recommendations (PLS-64 vs. concat vs. PCA) —
    **later found to have a label leak** (§3).
  - **Step 5** (`run_step5`): first genuine few-shot look, n_train ∈
    {10,15,20,...,100}, at the single historical tap (`layer12`/mean). Found
    conventional probes fail at n=20; TabPFN is the only probe that reliably
    beats predicting the mean.
  - **Step 6** (`code/eval/label_probe/study6.py`, `report6.py`,
    `screen.py`, `geometry.py`): swept the *readout* itself — 13 transformer
    layers × 6 pooling schemes × 4 unlabeled-corpus normalizers (none,
    standardize, l2, whiten) = 456 recipes, screened cheaply on `eval_a`,
    confirmed on disjoint `eval_b` with a pre-registered win rule, at
    n_train ∈ {20,50}, n_comp ∈ {1,2,3}. Verdict: **NO, 0/6 cells** — no
    embedding readout beats raw input at n≤50, under any recipe.
  - **Step 7** (`code/eval/label_probe/crossover.py`, `study.py` additions):
    laddered n itself (100→2000) at the best-so-far readout, re-selecting
    the readout on `eval_a` at each rung and scoring on `eval_b`, restricted
    to 1-comp. Found raw leads at every rung, gap narrowing but never
    crossing zero.
  - **(Uncommitted) v5 extension** (§0): extended the ladder to n=10→3,716
    (the full pool) with whitening included as a recipe on both arms —
    this is the run that finally closes the question the whole step 5–7
    arc was chasing.
- **Probe panel:** `code/eval/label_probe/regressors.py` — OLS, Ridge,
  RidgeCV (`alphas=logspace(-3,3,20)`), xgboost/HistGradientBoosting, kNN,
  MLP, PLS (`pls1`...`pls64` via `_PLSWrapper`), dummy (predict-mean).
  `geometry.py` adds kernel/graph probes: `ridge_strong`, `pca5_ridge`,
  `pca10_ridge`, `knn_cos1/3/5`, `gp_rbf`, `krr_rbf`, `graph_prop`,
  `graph_prop50`, `tabpfn_pca10/50`.
- **Splits:** disjoint `eval_a` (readout/recipe selection) and `eval_b`
  (reporting) evaluation sets carved from the 4,716 pool, remaining ~3,716
  as the training/draw pool for the label-efficiency ladders. Repeated
  random draws (hundreds at small n, down to 1 at the full pool) with
  paired comparisons (both arms scored on identical training indices at a
  given n).
- **Pooling / embedding extraction:** a "moment bank" — 10 per-channel
  statistics over 47 tokens, for both FE stages and all 13 transformer
  layers, extracted once per checkpoint on GPU (`code/eval/label_probe/
  taps.py`, `readouts.py`); every (layer, pooling) readout is then a
  cheap CPU subset/concat of that bank.

---

## 2. Cleanest, most trustworthy results

Ranked by how much scrutiny they survived:

1. **The v5 full-pool ladder (§0) is the cleanest, most complete result in
   the whole study** — one protocol, all recipes on equal footing, paired
   draws, IQR error bars at every rung, canary-checked. Raw (whitened) wins
   at n=10 through n=3,716; no crossing anywhere.
2. **Step 6's readout sweep (0/6 cells, bootstrap P(Δ>0)=0.00 in all six)**
   is well-instrumented: pre-registered win rule (written and unit-tested
   *before* measurement), disjoint screen/confirm splits, symmetric
   treatment of both arms (independently verified by a "Final whole-branch
   review" sub-agent — see transcript task-notification at line 3460,
   which reproduced `step6_verdict.json`'s six deltas exactly and found
   "Nothing Critical, nothing that could have produced or distorted the NO
   verdict").
3. **Step 1's corrected large-n numbers** (probe-panel, full n=4,716) are
   solid and reproduce the historical `legacy` protocol numbers to 4
   decimals — confirming the old pipeline wasn't buggy, just
   under-measured (n=1,000, one probe).
4. **Step 2's XGBoost-loses-to-linear finding survives**, but only once
   corrected against a properly-tuned linear baseline
   (`step2_addendum_vs_best_linear.md`) — the original `step2_REPORT.md`'s
   apparent XGBoost wins were an artifact of comparing against untuned
   Ridge(1.0), and is flagged as misleading without the addendum.

### Results that look good but are flagged as bug-driven / misleading

- **Step 4's PLS-64 recommendation ("+0.080 at 1-comp, +0.053 at 3-comp") is
  a label leak**, discovered in the *sibling* repo
  `/mnt5/home/hadar/nova/SpectralFM-label-regression` (`step6_pls_correction.md`,
  "Update 2026-09-10") and independently verified against this repo's copy
  in-session: `step4_deployment_axes.py` and `study.py`'s axis-3 both call
  `PLSRegression(n_components=k).fit_transform(X, y)` on the **entire**
  dataset before any train/test split — PLS is supervised, so the
  projection had already seen every test label. Measured inflation: 1-comp
  pls_64 leaky +0.711 vs honest +0.620 (**+0.090 inflation**); 3-comp pls_64
  leaky +0.847 vs honest +0.764 (**+0.083 inflation**). Once fit
  fold-internally, the honest PLS-64 gain over plain concat shrinks to
  "marginal" (1-comp: +0.022) or **no gain** (3-comp: −0.004) — the
  opposite of what was shipped as a recommendation.
  - **This repo's own `code/eval/label_probe/regressors.py` `_PLSWrapper`
    was independently checked in-session and confirmed clean** — it fits
    only on `X[tr], y[tr]` inside `screen.py`'s `m.fit(X[tr], y[tr])` call,
    never on the full `X_all`/`y` before splitting. So the step-6/step-7
    `pls64` numbers in *this* repo are NOT subject to that specific leak.
    But `step4_deployment_axes.md`'s PLS-64 numbers in *this* repo (the
    committed `LABEL_REGRESSION_EVAL_SUMMARY.md`'s §4 "Use PLS-64" claim
    and Recommendation #4) **are** the same leaky pattern and should be
    considered retracted — this was never fixed/flagged in the committed
    repo, only in the uncommitted v5 rewrite (§0, recommendation #2, which
    replaces "ship PLS-64" with "ship whitening; PLS is not a substitute at
    few-shot n").
- **Step 6's `whitened_topk_r2` diagnostic was initially miscounted as weak
  due to a reporting bug** (`report6.py`'s diagnostic lookup silently
  dropped all four `whitened_topk_r2_*` rows on a str/int key mismatch
  after a JSON round-trip). Recomputed correctly outside the module, it's
  actually the *strongest* diagnostic (ρ≈0.58–0.63, later refined to
  ρ=+0.631 in v5) — flagged, not patched (out of scope), but disclosed
  clearly in both the committed summary and the v5 rewrite.
- **§1's headline "embedding wins at 1-comp, 0.628 vs 0.468" (committed
  summary, section 1) is itself flagged in the same document as
  superseded** — an artifact of an under-measured `raw|z|-|none` baseline;
  once raw gets a whitened readout it's already ahead by n=50 and stays
  ahead through the full pool (§0's table makes this the final word).

---

## 3. (a) The float32/float64 precision issue

**This did not originate in the SpectralFM-few-shot repo itself** — it was
discovered in a **sibling/parallel repo**,
`/mnt5/home/hadar/nova/SpectralFM-label-regression` (still on disk, not
scheduled for deletion per this task), in its
`code/eval_outputs/label_probe/2026-09-03_full_run/step6_pls_correction.md`
("Update 2026-09-10" section) and mirrored in its
`docs/html/label-regression-report.html` (§06a). The few-shot session
(this repo) became aware of it mid-session and **directly verified whether
it affected this repo's own pipeline and conclusions** (transcript lines
~5560–5980).

**What was found (in the sibling repo):** raw-input OLS's baseline jumps
from 0.404 to 0.824 (1-comp, n=4,716) under float64 arithmetic. Initially
this was treated as a **numerical artifact** — an unstable, unregularized
OLS solve on near-degenerate (highly collinear) raw spectral features,
assumed not reachable by any practical estimator. **That conclusion was
wrong.** PLS (a fold-internal, supervised-but-honest reducer) reaches the
same number (0.825) directly, **stable to std=0.0008–0.0009 across 8
fold-seeds, identical under float32 and float64** (unlike raw OLS, which is
dtype-sensitive), and a shuffled-label canary at every k collapses to ≈0.
So: **it is real signal, not a float-precision artifact** — RidgeCV's
isotropic shrinkage structurally cannot reach it because it penalizes
low-singular-value directions hardest, and the real label signal in raw
input lives exactly in those near-degenerate, highly collinear directions.
PLS finds it directly by selecting on label-covariance instead.

**How this repo (few-shot) engaged with it:**
1. Checked its own `pls64` implementation (`regressors.py`'s `_PLSWrapper`)
   for the same leak pattern that caused the *original* float64
   over-optimism scare in the sibling repo (the leak was actually a
   separate, PLS-preprocessing-fits-on-all-data bug, not the float64 issue
   per se — see §2 above) — **confirmed clean**.
2. Ran a scoped, honest, canary-checked investigation
   (`investigate_pls_smalln.py`, rescued to `fewshot_rescue/`) asking: does
   this large-n PLS ceiling lift transfer to the few-shot regime this study
   is actually about? **Answer: no.** At n≤50, fold-internal PLS (small k)
   loses to this study's existing whitened-raw best-of-panel at every n
   (e.g. n=50: PLS +0.377 vs existing +0.587); canary always cleanly
   negative (−0.37 to −0.70), confirming no leak in the check itself.
3. **Conclusion (stated in the rescued v5 report, §0 above): the
   float64/PLS ceiling lift is real, reproducible, and a genuine finding
   about raw input's large-n potential — but it is a large-n phenomenon
   that does not transfer to, or change, the few-shot conclusion this
   study is built around.** It also does not beat this study's own
   whitened-raw numbers even at large n (whitening reaches 0.856 at the
   full pool vs. PLS's 0.825) — whitening and PLS attack the same
   collinearity problem by different routes, and whitening's route already
   subsumed it.

**Trustworthiness:** the float64/PLS finding itself (sibling repo) is
well-verified (canary-clean, dtype-invariant, cross-seed stable). Its
integration into this repo's conclusions (the small-n check) is also
canary-verified and directly measured, not inferred — this is solid,
finished work, just never committed to git.

---

## 4. (b) The "cross-over" plot(s)

There isn't one single canonical "crossover plot" — there are three
successive versions, each superseding the last, plus one that was
**deliberately never produced (out of scope) beyond n=2,000** by design:

1. **`code/eval_outputs/label_probe/2026-09-03_full_run/label_efficiency_{1,2,3,12}comp.png`**
   (step 5): the *original* label-efficiency curves, x=n_train (log), one
   line per forward-pass stage, dashed rule at each stage's full-n
   asymptote. Built at the single historical `layer12`/mean tap only — this
   is now known (step 6) to be one of the *worst* possible taps, so these
   curves understate the embedding's few-shot potential and should not be
   used as the final word.
2. **`code/eval_outputs/label_probe/step7/step7_crossover_1comp.png`**
   (step 7, committed): the pre-registered crossing estimator's output,
   n=100→2000, 1-comp only, comparing a re-selected best embedding readout,
   a matched raw readout, and the historical tap. Reports `crossed=False,
   n_cross_interp=None` — gap narrows monotonically (ΔR² −0.487 at n=100 to
   −0.113 at n=2000) but never reaches zero in the measured range. This is
   the last **committed** crossover artifact and is trustworthy as far as
   it goes, but is explicitly scoped to n≤2,000 and 1-comp only (see
   `crossover.py`'s own docstring: capped at 2000 rather than 3716 because
   draws without replacement from a 3,716-pool already share ~54% of rows
   at n=2000, understating true sampling variance — a disclosed, deliberate
   scope limit, not an oversight).
3. **`step7_v3_real.png`** (rescued to `fewshot_rescue/`, **uncommitted**):
   the actual final answer — extends the ladder to n=10→3,716 (the entire
   available pool), adds whitening as a recipe on both raw and embedding
   arms, and finds **no crossing anywhere in the full measurable range**,
   with real (not extrapolated) IQR error bars at every rung including the
   single-draw n=3,716 point (explicitly flagged in the caption as "a
   single point estimate, not a tighter measurement than the rungs before
   it," since only one draw is possible when n equals the full pool). This
   is the plot to carry forward — it directly answers the question the
   whole "crossover" arc of the study existed to ask, and closes it
   negatively. One disclosed wobble: unwhitened raw dips at n=200 (0.35)
   below its own n=100 value (0.50) — called out in the caption as "a real
   measurement, not a chart error."

**Bottom line on the crossover plots:** the committed `step7_crossover_1comp.png`
is correct but incomplete/superseded; `step7_v3_real.png` (uncommitted, in
`fewshot_rescue/`) is the clean, complete, final version and should be what
gets carried forward, not the committed one.

---

## 5. (c) The true-vs-predicted scatter plot(s)

- **`code/eval_outputs/label_probe/2026-09-03_full_run/label_reg_true_vs_pred_grid.png`**
  (committed, step 1): a 30-panel grid (`plot_true_vs_pred_grid` in
  `code/eval/label_probe/plots.py:109`) — rows = 5 component counts (1, 2,
  3, 7, 12), columns = 6 forward-pass stages, one scatter per cell, best
  cell (globally highest R²) outlined in gold. **Caveat found and fixed in
  the uncommitted rework:** the original grid does **not** fix axis limits
  across panels — each panel's axes auto-scale to its own data range, which
  makes stages/component-counts visually incomparable (a tight, good-looking
  cloud on a badly-scaled axis can look deceptively similar to a genuinely
  good fit on a properly-scaled one). This was not identified as wrong in
  the committed report, only fixed later.
- **`fixed_step1_grid.png`** (rescued to `fewshot_rescue/`, uncommitted): the
  corrected version — restricted to 1–3 components × 5 stages (15 panels,
  dropping 7/12-comp and the redundant 6th stage), **every axis fixed to
  [−2, 2]** so panels are directly comparable at a glance (a point beyond
  that range is clipped from view rather than silently rescaling the
  frame — disclosed explicitly in the caption). Best cell in this fixed
  range is 3-comp raw input, R²=0.863, boxed. This is the version that
  should be carried forward; the original 30-panel grid should be treated
  as superseded/visually misleading without the axis-fixing.
- No other true-vs-predicted plots exist for the few-shot regime
  specifically (steps 5–7 report scalar R²/curves, not per-point scatters,
  at small n) — this scatter grid is exclusively a large-n (full pool)
  diagnostic, at the input/stage level, not at the label-efficiency-ladder
  level.

---

## 6. Files worth copying forward into a new clean repo

**From the committed repo** (`/mnt5/home/hadar/nova/SpectralFM-few-shot`):
- `LABEL_REGRESSION_EVAL_SUMMARY.md` — the committed final report (superseded in tone/scope by the rescued v5, but has useful step-by-step narrative and the recommendations list, with corrections flagged inline).
- `code/eval/label_probe/` (whole package) — `taps.py`, `readouts.py`, `features.py`, `cache.py`, `protocol.py`, `regressors.py`, `geometry.py`, `screen.py`, `study.py`, `study6.py`, `crossover.py`, `report6.py`, `report7.py`, `plots.py`, `__main__.py`, and `tests/` — this is the actual reusable evaluation machinery (moment-bank extraction, readout/pooling/normalizer sweep, probe registry, pre-registered win-rule/bootstrap crossing estimator). All under active test coverage (88 tests per the v5 footer).
- `code/eval_outputs/label_probe/2026-09-03_full_run/step{1,2,3,4,5}_REPORT.md` + `step2_addendum_vs_best_linear.md`, `step4_deployment_axes.md` (but flag PLS-64 recommendation as retracted — see §2), `step5_tabpfn_pca50.md` — plus their `*_results.json`.
- `code/eval_outputs/label_probe/step6/` and `step7/` — `step6_REPORT.md`, `step7_REPORT.md`, `*_results.json`, `*_verdict.json`, `*_confirm.json`, `*_screen.json`, and the PNGs (`step6_confirm.png`, `step6_screen_{1,2,3}comp.png`, `step7_crossover_1comp.png`).
- `docs/superpowers/specs/2026-09-09-few-shot-readout-geometry-design.md` and `2026-09-09-label-efficiency-crossover-design.md`, plus the matching `plans/` — these document the pre-registered win rules and estimator design *before* measurement, which is the backbone of why the NO verdicts are trustworthy.
- `docs/html/step6-readout-geometry.html` — committed HTML version of the step 6 verdict.
- `TASKS.md` sections T13, T14, T15 — the ledger entries with dates and status.

**From the rescued /tmp scratchpad** (`fewshot_rescue/` in this session's scratchpad — **copy these somewhere permanent immediately, they exist nowhere else**):
- `label_regression_findings_v5.html` — the single best final deliverable of the whole study (self-contained, all images embedded).
- `body_v5.html` — its readable source (no images), useful for future edits.
- `report_template_v5.html` — the style/scaffold used to build it.
- `stages_1_2.png`, `full_crossover_measured.json`, `measure_full_crossover.py`, `investigate_pls_smalln.py` + its logs (`investigate_pls.log`, `investigate_pls2.log`) — the actual scripts/data behind the final n=10→3,716 ladder and the small-n PLS check; **not reproducible from the committed repo alone**.
- `step6-ledger-archive.md`, `step7-ledger-archive.md` — archived ledger snapshots referenced by the rewrite.

**Cross-reference, not in scope for copying but worth knowing about:**
`/mnt5/home/hadar/nova/SpectralFM-label-regression` — a separate, still-extant
sibling repo where the original float64/PLS-ceiling and PLS-label-leak
findings were made (`code/eval_outputs/label_probe/2026-09-03_full_run/step6_pls_correction.md`,
`docs/html/label-regression-report.html` §06a). Not being deleted per this
task's scope, but if it later is, its `step6_pls_correction.md` is the
primary source for §3 above and should be rescued too.

---

## 7. What was abandoned, superseded, or wrong

- **Step 4's "use PLS-64" recommendation** — label leak (PLS fit on all
  labels before CV split, in `step4_deployment_axes.py`/`study.py` axis 3).
  Honest fold-internal numbers show marginal-to-no gain over plain concat.
  Superseded by "ship whitening instead" (v5, §0).
- **The historical `layer12`/mean extraction tap**, used for every
  pre-step-6 measurement of this backbone (including the original T6 claim
  this whole study was launched to re-examine) — shown by step 6 to be
  among the *worst* available taps (layers 0–3 are 2–2.5× stronger). Any
  number computed at this tap (which is most of the project's prior
  history, including recon-trained-checkpoint comparisons per
  Recommendation #2 in TASKS.md and the committed summary) should be
  treated as measured at an artificially weak recipe.
- **The original step-1 "embedding wins at 1-comp, 0.628 vs 0.468" framing**
  — the committed summary itself already flags this as superseded (its own
  §1, "Superseded by step 7" paragraph); the rescued v5 ladder confirms
  raw wins at every n once given a fair (whitened) readout, closing the
  question definitively.
- **XGBoost's apparent wins in `step2_REPORT.md`** — artifact of an
  untuned Ridge(1.0) comparison baseline; XGBoost loses to the best linear
  probe in all 30 stage×component cells once corrected
  (`step2_addendum_vs_best_linear.md`).
- **H8 (explicit cross-component features as "the dominant carrier")** —
  tested and rejected; recovers only a small fraction of either the raw
  input's or the transformer's signal. Signal is distributed across the
  full spectral shape, not summarizable into a few engineered ratios.
- **"Exchangeability" over-generalization** — the ≈0 R² collapse from
  pooling raw signal into one shared model does *not* extend to pooling
  *embeddings*, which lose less (0.661 at 3-comp) — an initially
  over-broad claim, narrowed.
- **`report6.py`'s diagnostic lookup bug** — silently dropped
  `whitened_topk_r2_*` rows on a str/int key mismatch after a JSON
  round-trip, initially making the study's own best-predicted diagnostic
  look like one of the weakest. Flagged, not patched (deliberately out of
  scope for this eval-focused work), fixed only in the manual
  recomputation used for the v5 rewrite.
- **20-label-constraint framing (step 5's headline, "TabPFN makes 20
  labels marginally viable")** — the committed summary's own §5 walks this
  back at length: once a fair (whitened) readout is used, conventional
  probes beat TabPFN even at n=50 (GP-RBF +0.58 vs TabPFN +0.37 on the
  same whitened raw input, per v5). TabPFN's apparent edge was itself an
  artifact of testing only unwhitened recipes.

---

## 8. Open questions / next steps relevant to the FM-probe end goal

From both the committed summary's "Scope / not done" section and the
rescued v5's "Recommendations":

1. **Re-measure recon-trained backbones under the whitened-recipe, correct-tap
   protocol.** All existing "≈0" scores for those checkpoints (referenced in
   `SpectralFM/TASKS.md` T6) were taken at the worst-known tap
   (`layer12`/mean) with an under-powered protocol — the checkpoint ranking
   that guides the whole training programme (SSL vs. reconstruction-trained
   backbones) is currently **unverified** against a fair recipe. This is
   flagged as the single highest-value follow-up in both the committed and
   uncommitted reports.
2. **Ship whitening, not PLS, as the default deployed feature-vector
   recipe** for any few-shot probe built on this backbone or on raw input —
   it is unlabeled-fit (free), and is the single largest lever found in the
   entire study (+0.47 R² at 1-comp/n=50).
3. **Active/stratified label selection is untested end to end.** Every draw
   in every step was i.i.d. random from the pool. This is called out
   repeatedly (committed summary's scope section, v5's recommendations) as
   the most likely remaining source of leverage specifically for the
   embedding side — selection is a geometry problem, and geometry is
   exactly where an embedding's structure could still matter even though
   its raw per-point features don't currently win.
4. **2-/3-component behavior across the full n=100→3,716 ladder is
   untested** — step 7 and the v5 extension both restricted the full ladder
   to 1-component only (justified by step 1's large-n anchors showing
   embedding-leads-raw only at 1-comp, historically — now itself
   superseded, so this restriction's justification is weaker than when it
   was made).
5. **Distribution shift and calibration are completely untested** anywhere
   in this study — relevant to any real FM-probe deployment where the
   labeling distribution at inference time may differ from the training
   pool's.
6. **The task itself may be a poor demonstration vehicle for FM value.**
   The v5 report's own mechanism section argues a smooth, low-dimensional,
   already-well-conditioned 245-point signal with a smooth scalar target is
   close to a worst case for showing a representation-learning advantage —
   raw features are already near-optimal. If the goal is to demonstrate
   the backbone's value as an FM probe, a task where raw features
   demonstrably fail (multi-component disentangling, anomaly/outlier
   detection, cross-instrument or cross-distribution transfer) is
   recommended over further work on this specific label-regression setup.
7. **No trained (as opposed to frozen, hand-designed) pooling/projection
   head was ever tried** — explicitly noted as untested in both reports
   (the "attention-pooling head is still not run").
