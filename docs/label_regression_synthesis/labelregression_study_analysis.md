# Label-Regression Study — Extraction Before Deletion

Source repo: `/mnt5/home/hadar/nova/SpectralFM-label-regression` (branch
`label-regression-dev`, HEAD `e6dd3e9`, plus uncommitted working-tree edits
to `SUMMARY.md`, `step6_pls_correction.md`, `step6_reducers_c{1,3}.json`,
`docs/html/label-regression-report.html` — these uncommitted edits are the
**latest and most-corrected** version of the findings; read them, don't rely
on the last commit alone).

Session transcript analyzed:
`/home/hadar/.claude/projects/-mnt5-home-hadar-nova-SpectralFM-label-regression/a6fb3771-4970-45ad-a15d-a6cca98d5c98.jsonl`

---

## 1. What was specifically studied

**Label:** a single scalar float label (`parameter_0` in `labeled_data`'s
`labels.tsv`) associated with each of 12 unique spectral "components" per
spectrum, on the `nova_data/labeled_data` dataset (n=4,716 spectra after
dropping 2 exact-duplicate components — comp20≡comp14, comp21≡comp15 on
100% of spectra).

**Backbone:** the Feb-25 SSL checkpoint,
`/mnt5/noy/SpectralFM/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt`
— the same one T6/ARCHITECTURE.md's headline claim rests on. No
reconstruction-trained ("3AE") checkpoints were re-evaluated (they live on
the RunAI PVC, out of scope for this Geoffrey-only study — flagged as an
open gap, see §6).

**Methodology — a 6-step, multi-probe, multi-stage rebuild** of the
project's existing (and, this study found, under-powered) label-regression
eval. Code: `code/eval/label_probe/` (new, ~2,400 lines: `study.py`,
`reducers.py`, `regressors.py`, `taps.py`, `features.py`, `protocol.py`,
`fewshot_curve.py`, `plots.py`, `cache.py`, `fewshot_report.py`,
`__main__.py`). The old `code/eval/evaluations/label_regression.py` and
`label_regression_sweeps.py` were left untouched deliberately, for lineage
comparison, not extended.

- **Taps (pipeline stages):** raw input, input (z-scored), post-FE
  (pre-LayerNorm), post-FE (post-LayerNorm), post-projection,
  post-transformer. (`code/eval/label_probe/taps.py`)
- **Component counts:** 1, 2, 3, 7, 12 (concatenated feature vectors across
  components — "components are not exchangeable," see §5 below, so
  per-component pooling into one shared model was explicitly rejected).
- **Probes swept:** OLS, Ridge(1.0), RidgeCV, XGBoost/HistGradientBoosting,
  PCA+Ridge, PLS, Lasso, ElasticNet, MLP, kNN, TabPFN, plus a `dummy`
  (predict-the-mean) floor.
- **Splits:** primary protocol = repeated shuffled 5-fold CV at full n
  (n=4,716), 3 repeats; a `legacy` protocol reproducing the historical
  n=1,000/seed=42/unshuffled-KFold(5) setup exactly (to 4 decimals) for
  lineage; a separate **few-shot protocol** (Step 5) using repeated random
  subsampling, 20–100 draws per cell, n_train ∈ {20, 50, 100, ...}, both
  transductive (PCA basis fit on all unlabeled data — legitimate, since
  deployment has unlabeled data) and inductive variants.
- **Difference from the sibling few-shot study** (implied by the SpectralFM
  eval package's `code/eval/label_probe/fewshot_*` vs the general eval
  package's few-shot evaluator): this study's Step 5 is purpose-built for a
  **client-stated constraint of ~20 labeled samples**, reports full
  distributions (never a point estimate) and `frac_positive_r2` (how often
  a probe beats the mean) as the actionable number, and explicitly separates
  "asymptotic representation quality" (Steps 1–4, n≈4,716) from "sample
  efficiency" (Step 5) as two different questions that do not extrapolate
  from one another.

---

## 2. The cleanest, most trustworthy results

**Trust the 2026-09-10 corrected numbers, not the original Step-1/4 headline.**
Two rounds of correction happened, both documented and both retracting
publicized numbers — this is a good-faith, self-correcting study; the last
version standing is the trustworthy one.

### Final corrected picture (n=4,716, 5-fold CV, PLS fit fold-internally, canary-verified)

| n-comp | raw input (PLS, corrected) | post-transformer (best of RidgeCV/PLS) | recovery |
|---:|---:|---:|---:|
| 1 | **0.825** | 0.645 | 78% |
| 2 | **0.939** | 0.763 | 81% |
| 3 | **0.969** | 0.793 | 82% |
| 7 | **0.993** | 0.866 | 87% |
| 12 | **0.999** | 0.945 | 95% |

**There is no component count at which the SSL backbone beats correctly-probed
raw input.** This *reverses* the project's long-standing headline claim
(`TASKS.md` T6 / `ARCHITECTURE.md`: "emb R²=0.44 beats input 0.38", and the
intermediate, also-superseded claim from this same study that "embedding wins
at 1-comp, 0.628 vs 0.468").

Other results that hold up (not touched by either correction):
- **XGBoost never beats the best linear probe** — loses in all 30 stage ×
  component cells by 0.165–0.410 R² (`step2_addendum_vs_best_linear.md`).
  Its apparent win in the original `step2_REPORT.md` was an artifact of
  comparing against a fixed, untuned `Ridge(alpha=1.0)` rather than the
  best-of-panel linear probe.
- **Split/resampling noise is negligible** (SD≈0.0005, repeated 5-fold),
  three orders of magnitude below systematics from probe choice (0.27 R²),
  sample size (0.07 R²), and normalization (0.06–0.17 R²). This retroactively
  overturns an earlier project assumption (T8's A/B at ΔR²≈0.03 was
  originally assumed "decided inside the noise" — it wasn't, split noise is
  ~30x smaller than that margin).
- **Per-component z-scoring costs R² at 1-comp** (−0.06) but *helps* at
  2–12-comp (+0.05 to +0.07) — because z-scoring only damages
  cross-component scale ratios, which don't exist at 1-comp.
- **Components are not exchangeable for raw signal**: pooling per-component
  rows into one shared model collapses to R²≈0.006 regardless of
  `GroupKFold` — this reproduces a historical bug (commit `bf94422`) from
  first principles. But it does **not** generalize to embeddings: averaging
  embeddings across components gives R²=0.661 at 3-comp — lossy, not
  collapsed. ("Exchangeability over-generalized" — one of 4 of the
  investigator's own hypotheses that were explicitly falsified; see
  `docs/html/label-regression-report.html` §07.)
- **Few-shot (n_train≤50-100) inverts the large-n picture even more sharply
  in raw input's favor**: raw input beats the transformer embedding ~6× at
  n_train=50 (3-comp, matched PCA-50 reduction: 0.458 vs 0.076), and this was
  tested specifically to rule out "unfair reduction" as a confound (PCA-10
  gives the embedding only ~11% of its label signal vs 77% for raw input;
  re-tested at a fair PCA-50 reduction where the embedding ceiling rises to
  0.333 — the gap still doesn't close). At n_train=20, **no conventional
  probe beats predicting the mean**; only TabPFN does, reliably (~90% of
  draws, R²≈0.08–0.13), and only on raw 1–3-comp input, not on embeddings.

**Flagged as NOT fully re-verified**: `input(z)`/FE/proj columns under the
PLS-correction protocol are only spot-checked at 1- and 3-comp (both showed
comparably large corrections, e.g. z-scored input 1-comp: 0.468 → 0.771) —
a full re-sweep across all 5 component counts was not done. Whether the PLS
correction also changes the few-shot (Step 5) numbers is explicitly
untested.

---

## 3. Special-attention items

### (a) float32 vs float64 precision issue — full diagnostic arc

**What broke:** `code/eval/data_loader.py` and the legacy
`code/eval_label_regression.py` cache feature matrices as `float32`
(`.astype(np.float32)`), and this float32-ness propagates through every
downstream regressor — including OLS via normal-equations solve. Raw-input
feature matrices are severely ill-conditioned (condition number ≈10⁹–10¹⁰,
from highly collinear adjacent spectral bins). At `cond × eps(float32)≈875
≫ 1` the OLS solve is numerically destroyed; at `cond × eps(float64)≈1.6e-6
≪ 1` it is fine. This made OLS on raw input silently return R²≈0.41 instead
of the true ≈0.82 at 1-comp.

**How diagnosed:** the investigator noticed a run using
`LinearRegression` (with `.astype(np.float64)` explicitly added for a
side-check) returned 0.8519/0.8144 (train/test), identical to `pinv` and to
PLS — vs. an earlier float32 run returning 0.408. Traced the discrepancy to
dtype, confirmed via a dtype sweep across configs
(1c OLS: 0.4042→0.8237 = catastrophic change; 1c RidgeCV: 0.4095→0.4095 =
**zero** change; **RidgeCV is immune** because ridge's `X^TX+αI`
regularization is structurally the same fix as the numerical-stability fix).

**Self-correction the investigator made, worth flagging explicitly:** the
session *first* concluded (transcript ~line 4256–4372, and an earlier
published report revision) that the float64 OLS 0.82 number was **numerical
mush, not real signal** — based on: (B) stability across 8 fold-seeds
(SD=0.0009, looked "too stable" to be overfit noise, but was initially
mis-read), and (C) sensitivity to a tiny ridge penalty (α=1e-6 drops R² from
0.824→0.738; α=1e-4 → 0.517) — reasoning "a real signal wouldn't be this
fragile to negligible regularization." This was published as "artifact,
closing the investigation, no headline numbers change."

**That verdict was then explicitly retracted** (transcript line 4662,
`step6_pls_correction.md` "Update 2026-09-10" section): running **PLS**
(fit correctly inside each CV fold — an ordinary, practical, well-regularized
linear method, NOT a raw unregularized solve) reaches the **identical**
number: 0.8245±0.0008 at 1-comp, 0.9690±0.0002 at 3-comp, stable across 8
fold-seeds, **dtype-invariant (identical under float32 and float64, unlike
OLS)**, and passes a shuffled-label canary (shuffled R²≈−0.08 to −0.28,
i.e. collapses to noise as it should). This is the correct, final diagnosis:
**it is real signal that RidgeCV structurally cannot reach** (ridge's
isotropic shrinkage penalizes exactly the low-singular-value/near-degenerate
directions this signal lives in; PLS selects directions by label-covariance
instead and finds it directly), not a float32 bug and not OLS numerical mush.
The float32/64 sensitivity was real but was a red herring about *how* you
reach the signal (OLS needs float64 and is unstable even then), not about
*whether* the signal is real.

**Trustworthiness of the fix:** high. Verified three independent ways
(dtype invariance, 8-seed resampling stability, shuffled-label canary), and
a canary tool was productized (`python -m eval.label_probe.reducers --canary`)
specifically so this class of issue gets caught automatically in future runs.

### (b) "Cross-over" — not a saved plot, but a documented crossing phenomenon

There is **no single artifact literally named "cross-over plot."** The
"crossing" referred to in the study is a data pattern, stated in prose and a
small table (not its own figure): **OLS degrades monotonically with sample
size while RidgeCV improves, and the two curves cross.** Measured (2-comp,
n=500→4,716): OLS 0.735→0.665 (down), RidgeCV 0.706→0.756 (up). Cited in
`plan.md` (pre-implementation diagnostics), `step3_REPORT.md` line 43 (H1),
and `docs/html/label-regression-report.html` §07 ("More data is free R² —
refuted... They cross. Any single-n number is a point on a surface, not a
constant.").
- Correctness: this is clean and uncontested by either later correction —
  it's about sample-size × probe-choice interaction, not about the PLS/OLS
  precision issue in (a). Safe to carry forward.
- The closest thing to a visual "crossing" plot that *does* exist is
  `label_reg_stages.png`'s right panel ("Regressor comparison at 12-comp"),
  which plots OLS/Ridge/RidgeCV lines across the 6 pipeline stages (not
  across n) — Ridge visibly dips and crosses OLS/RidgeCV around
  post-FE(pre-LN). This is a genuine, uncorrected observation but is a
  different axis (pipeline stage, not sample size) from the n-crossing
  described above. Don't conflate the two when reusing this material.

### (c) True-vs-predicted scatter — `label_reg_true_vs_pred_grid.png`

30-panel grid (6 stages × 5 component counts), viewed directly. Assessment:
- **Generally good and informative.** Clean diagonal alignment, tightening
  visibly as component count increases (1-comp: loose scatter around the
  diagonal; 12-comp: near-perfect diagonal, R²=0.989 best cell highlighted
  in the figure with a gold border).
- **No degenerate/collapsed predictions** (e.g. all-same-value predictions)
  anywhere in the grid — predictions have healthy spread (`pred σ` reported
  in each panel title, close to 1.0 in most cells, tightening toward the
  true σ as component count grows).
- **One real visual artifact worth flagging**: several panels (especially
  post-FE pre-LN / post-LN columns at 2–3-comp, and to a lesser extent
  post-projection) show a **vertical-banding pattern** — predictions
  clustering into a small number of discrete vertical strips rather than a
  smooth cloud. This looks like a discretization/degenerate-direction
  artifact in those intermediate representations, not something the study
  called out explicitly in text — worth a second look if this plot is
  reused, since it might indicate those pooled FE-stage features have much
  lower effective rank than the raw 512-dim vector suggests.
- **Important caveat for reuse**: this grid's raw-input and best-of-panel
  numbers are the **pre-PLS-correction** numbers (best-of-panel
  OLS/Ridge/RidgeCV only). E.g. its "Raw input, 1-comp, R²=0.410" panel is
  now known to understate the true raw-input ceiling (0.825, per §2 above).
  The plot is still useful for shape/calibration/no-clipping checks, but its
  raw-input R² annotations should not be quoted without the PLS correction
  applied alongside.

Other figures, sanity-checked at same level: `label_reg_stages.png`
(headline, described above — left panel's dashed "input ceiling" reference
lines are similarly pre-PLS-correction and understate raw input, especially
at 1-comp: dashed line ≈0.47 vs corrected 0.825). `label_efficiency_{1,2,3,12}comp.png`
(few-shot curves, Step 5) were not visually inspected in this pass but are
described quantitatively and consistently in `step5_REPORT.md`/`SUMMARY.md`.

---

## 4. Files worth copying into a new clean repo

**Code (all in `code/eval/label_probe/`, fairseq-free per project convention):**
- `study.py` (883 lines) — the core multi-stage, multi-probe orchestration.
- `reducers.py` (250 lines) — PCA/PLS/reducer sweep + the productized
  `--canary` shuffled-label leak check. **High value** — reuse the canary
  tool as a standing check for any future supervised-transform work.
- `regressors.py`, `taps.py`, `features.py`, `protocol.py`, `cache.py`,
  `fewshot_curve.py`, `fewshot_report.py`, `plots.py`, `__main__.py`.

**Docs (read these before re-deriving anything):**
- `code/eval_outputs/label_probe/2026-09-03_full_run/SUMMARY.md` — best
  single entry point, already reflects both corrections (working-tree
  version, not last commit — copy the working tree, not `git show HEAD:`).
- `step6_pls_correction.md` (working-tree version) — the float32/PLS/leak
  diagnostic writeup, the single most important supporting doc.
- `step1_REPORT.md`, `step2_REPORT.md` + `step2_addendum_vs_best_linear.md`,
  `step3_REPORT.md`, `step4_REPORT.md` + `step4_deployment_axes.md`,
  `step5_REPORT.md` + `step5_tabpfn_pca50.md`.
- `docs/html/label-regression-report.html` (working-tree version) — polished
  narrative report with all corrections folded in; good source for a
  write-up but re-read it (not the committed HEAD) since it has uncommitted
  fixes.
- `plan.md` (repo root) — the pre-implementation diagnostic session that
  first found the 0.27 R² probe-choice problem; good methodological
  background, not itself a set of final numbers.
- `TASKS.md` T13 section (search `T13.` — line ~319) — the ledger entry;
  cross-reference for how this connects to T6/T8/T12 decisions.

**Results/data:**
- `code/eval_outputs/label_probe/2026-09-03_full_run/*.json` — the raw
  per-step numeric results (`step{1,2,3,4,5,5b,6}*_results.json` /
  `*_reducers_*.json`) — useful if re-deriving tables or plots later.
  Note two of the `step6_reducers_c{1,3}.json` files have working-tree edits
  not yet committed; grab the working-tree version.
- The 6 PNGs in that directory (`label_reg_stages.png`,
  `label_reg_true_vs_pred_grid.png`, `label_efficiency_{1,2,3,12}comp.png`)
  — final, corrected-protocol figures (modulo the raw-input-ceiling caveat
  in §3c above for the two headline plots).
- Checkpoint pointer (not a file to copy, just note it): Feb-25 SSL backbone
  at `/mnt5/noy/SpectralFM/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt`.

**Not worth copying:** `code/eval/evaluations/label_regression.py` and
`label_regression_sweeps.py` (the pre-existing, narrower single-probe
implementation) — kept only for lineage/reproduction checks in this study,
now fully superseded functionally by `code/eval/label_probe/`. Also skip
`debug_recon_encoding.py` and other repo-root one-off debug scripts — unrelated
to label regression.

---

## 5. Abandoned / superseded / wrong — do NOT carry forward as-is

1. **"Embedding wins at 1-comp" (0.628 vs 0.468)** — the study's own
   Step-1 headline, published and then explicitly retracted 2026-09-10. Root
   cause: RidgeCV structurally cannot reach the raw-input signal that PLS
   can (see §3a). **Do not cite this number without the correction.**
2. **"Use PLS-64 (not concatenation) for the embedding feature vector"** —
   retracted (`79af610` "Retract the PLS-64 recommendation: it rested on a
   label leak"). Root cause: `PLSRegression.fit_transform(X, y)` was called
   on the *whole* dataset (train+test) before cross-validating a downstream
   RidgeCV — a supervised projection fit on labels the CV then "held out."
   Detected via a shuffled-label canary: leaky pipeline scored **R²=+0.18 on
   permuted labels** (should be ≈0); fixed (fold-internal) pipeline scored
   −0.28. Honest gain from PLS on the *embedding* is marginal at best
   (+0.022 at 1-comp, ~nil at 3-comp vs plain concatenation) — concatenation
   is fine for the embedding. (PLS *is* still warranted and non-optional on
   *raw input*, per §2 — the leak only ever affected the supervised-fit
   procedure, not PLS's applicability in general.)
3. **XGBoost "beating" input at raw input (+0.173 at 2-comp)** — an artifact
   of comparing against fixed `Ridge(alpha=1.0)` rather than best-of-panel
   linear. Once compared fairly, XGBoost loses everywhere (§2). Don't reuse
   the original `step2_REPORT.md` framing without the addendum.
4. **"Cross-component ratios are the dominant carrier" (investigator's own
   hypothesis, H8)** — falsified: hand-engineered per-component
   moments/pairwise-diff/log-energy-ratio features recover only 0.398 of raw
   input's ceiling and 0.057 of the transformer's — the signal is
   distributed across the whole spectral shape, not summarizable this way.
5. **"Components are never exchangeable" over-generalized** — true for raw
   signal (collapses to R²≈0.006 if pooled), false for embeddings (0.661 at
   3-comp when averaged) — don't assume the raw-signal finding transfers to
   embedding-space pooling in future work.
6. **"PCA-10 explains why TabPFN fails on embeddings"** — tested directly at
   a fair PCA-50 reduction and refuted; the embedding is genuinely less
   label-efficient few-shot, not merely under-reduced (§2).
7. **The original 2026-08-21 numbers "orphaned...only in a YAML comment"**
   (per SUMMARY.md recommendation #5) — never folded into TASKS.md; flagged
   as a loose end, not a validated result — don't treat as settled.
8. **The `label_reg_stages.png` / `label_reg_true_vs_pred_grid.png` dashed
   "input ceiling" references** — visually still the pre-PLS-correction
   ceiling; don't quote those specific overlay numbers without adjusting per
   §3c.

---

## 6. Open questions / next steps relevant to "few-shot regressor over the embedding" (the end goal)

This is the single most load-bearing section for the project's stated end
goal, since the study's final verdict is uncomfortable for that goal:

- **Central unresolved tension for the FM-probe goal**: at n≈4,716 the
  backbone's embedding recovers 78–95% of a (corrected) raw-input ceiling —
  informative, but never *better* than properly-probed raw input at any
  component count. At the few-shot regime the project actually cares about
  (n_train≤50–100), raw input beats the embedding by ~6×, and the gap is
  **not** explained by an unfair dimensionality reduction (ruled out at
  matched PCA-50). **This directly challenges the "few-shot regressor over
  the embedding as an FM probe" framing** — as currently trained, this SSL
  backbone's embedding is a worse few-shot regression substrate than the raw
  245-point signal itself, at every tested regime. Any few-shot-probe work
  building on this backbone should confront this finding head-on rather than
  assume the embedding is the right substrate by default.
- **Re-measure recon-trained ("3AE") backbones under this corrected
  protocol** — their historical "≈0" label-regression scores came from the
  same under-powered (single-RidgeCV, n=1,000) protocol that understated
  even the SSL backbone by 0.2–0.3 R². The T6 *ranking* (SSL >> recon-trained)
  is therefore unverified under the new protocol — could be smaller than
  claimed, or could hold; genuinely unknown. Blocked on: those checkpoints
  live on the RunAI PVC, not Geoffrey.
   (Consistent with the parent project's active thrust — reconstruction
  decoders on a frozen, already-good SSL backbone rather than training
  through it — see the outer `nova/CLAUDE.md`.)
- **Which 20 (or 50, 100) spectra get labeled is untested and probably the
  biggest remaining win** — all few-shot draws in Step 5 were random;
  stratified or active-learning selection of the labeled set would very
  plausibly beat random draws and was explicitly flagged as the "obvious
  follow-up," not attempted.
- **Distribution shift and calibration are entirely untested** — the whole
  study is in-distribution, same-source spectra.
- **H4 (pooling schemes) and a trained attention-pooling head were never
  run** — would need re-extraction of embeddings at other pooling choices;
  currently the study only used one (unspecified, presumably mean) pooling.
- **TabPFN excluded at 12-comp by compute budget** (~29 GPU-hours for the
  full grid) — can't compare TabPFN vs linear at the highest component
  count.
- **RRR (reduced-rank regression) flagged as inapplicable only because the
  label is scalar** (`parameter_0`) — if the upstream data-generating
  pipeline's other parameters (name implies `parameter_1`, etc. exist) could
  be recovered to make the label multivariate, RRR could apply and was
  flagged as "plausibly a bigger win than any reducer choice examined here."
  Worth asking the data owner — an open, unexplored lead.
- **Whether the PLS correction changes the few-shot (Step 5) numbers is
  itself untested** — Step 5's few-shot curves currently only used the
  pre-correction protocol; if PLS also recovers more raw-input signal in the
  n≤100 regime, the "raw input wins ~6×" gap could be even larger (or, less
  likely, could shift) — not yet re-run.
