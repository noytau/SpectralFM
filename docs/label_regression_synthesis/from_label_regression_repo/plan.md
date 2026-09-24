# Label-Regression Evaluation, Rebuilt

Branch: **`label-regression-dev`** (created at execution time, not now).
Mirror this file to `plan.md` in the repo root; do not commit it.

---

## Context

Label regression is the project's decisive metric. `TASKS.md` T6 and
`ARCHITECTURE.md` both rest on one number: the Feb-25 SSL backbone scores
emb R²=0.44 against a raw-input baseline of 0.38, therefore "SSL
pretraining produces label-informative embeddings, reconstruction-trained
backbones don't." Every decision in the T8→T12 ladder was made on this
metric, including T8's A-vs-B (ΔR²≈0.03) and step 2's mlp768-over-mlp2048
(ΔR²≈0.07).

The current implementation (`code/eval/evaluations/label_regression.py`)
answers a narrower question than the program needs:

1. **It probes only the transformer output.** FE and projection taps exist
   (`structured_similarity.py:189-236`) but no label probe uses them, so
   when embeddings underperform there is no way to tell *where* the signal
   was lost.
2. **The probe is a single linear RidgeCV**, so "the signal did not
   survive" and "the signal survived non-linearly" are indistinguishable.
3. **The feature vector is 1–3 mean-pooled component embeddings
   concatenated**, mirroring a legacy script and never validated.

Four steps, each with a short report. The headline deliverable is one
figure: raw input, post-FE, post-projection, post-transformer on one axis.

**Scope decisions confirmed with the user:** dedicated venv for xgboost
(noy's env is not writable by `hadar`); Feb-25 SSL backbone only
(`step1`–`step4` checkpoints live on the RunAI PVC); new CV protocol plus a
legacy-compatible row; DS pass first, then a trained pooling/MLP head on
frozen embeddings, no backbone training.

---

## Measured evidence (this session)

### The original numbers reproduce exactly

Running the repo's own `load_labeled_data` → `_normalize_like_fairseq` →
`compute_linear_probing_metrics` at the original settings (n=1000, seed 42,
`KFold(5, shuffle=False)`) returns **0.3772 / 0.7060 / 0.7568** for 1/2/3
components — the recorded values to four decimals. The historical pipeline
is sound. Everything below concerns what it *chooses to measure*.

### The probe choice is an uncontrolled variable worth up to 0.27 R²

Same data, same folds, only the regressor swapped (n=1000, z-scored):

| Probe | 1-comp | 2-comp | 3-comp |
|---|---:|---:|---:|
| Ridge(α=1.0) | 0.2376 | 0.5414 | 0.5792 |
| **RidgeCV(−3..3, cv=5) — used everywhere** | **0.3772** | **0.7060** | **0.7568** |
| RidgeCV(−3..4, LOO) | 0.3773 | 0.7060 | 0.7725 |
| LinearRegression (OLS) | **0.5079** | **0.7421** | 0.7698 |

At n=1000 the reported baseline is not the best a linear model can do: the
honest linear baseline at 1-comp is 0.5079, not 0.3772 — a 0.13 R² gap.
Across the panel the input baseline spans **0.238 → 0.508 at 1-comp**, a
0.27 R² range set purely by the probe.

**This undercuts the program's headline claim — but the direction is not
yet settled.** T6 reports the Feb-25 SSL backbone at emb R²=0.4412 vs input
0.3772 and concludes SSL embeddings are label-informative. At n=1000 the
OLS input baseline (0.5079) is *above* the embedding, which would reverse
the conclusion. **However, OLS degrades with n (see the learning curve
below), so whether 0.5079 survives at n=4,716 is unverified.** The claim
that stands today is the weaker one: a single arbitrarily-regularized probe
number cannot support the T6 conclusion in either direction. Settling it at
full n is step 1's first job.

### Per-component z-scoring destroys label information

All 4,716 spectra, 2-comp:

| Probe | raw | z-scored (as the eval does) | cost |
|---|---:|---:|---:|
| OLS | **0.8318** | 0.6670 | **−0.165** |
| RidgeCV | 0.8179 | 0.7577 | −0.061 |

`_normalize_like_fairseq` z-scores each component independently, so
per-component amplitude *and* every cross-component scale ratio vanish
before any probe runs. No regressor recovers it afterward.

**The fix is asymmetric, and that matters for fairness.** The backbone was
*trained* on layer-normalized input, so feeding it raw signal is
off-distribution. Honest design: report the input baseline on **raw**
signal (its best form) while embeddings keep the normalization the backbone
expects. This *widens* the input-vs-embedding gap. Report both variants so
the comparison is not rigged in either direction.

### The input ceiling

All 12 unique components, 2,940-dim, all 4,716 spectra:

| | raw | z-scored |
|---|---:|---:|
| RidgeCV | **0.9886** | **0.9799** |
| OLS | — | 0.9448 |

**`parameter_0` is essentially solved from the raw multi-component input,
under either normalization.** The z-scoring penalty shrinks with component
count (−0.165 at 2 comps, ~−0.01 to −0.04 here): with 12 components there
is enough redundancy that shape alone recovers most of what the lost
amplitude carried.

Against a ~0.98 input ceiling, the best embedding number in the entire
program is ~0.51 — suggesting the backbone discards a large fraction of a
signal a linear model reads almost perfectly off the input. **That gap, not
the 0.44-vs-0.38 margin, is what this eval should be reporting.**

Two honesty caveats on that sentence, both of which step 1 must discharge
before it is promoted from suggestive to established:

- **It is not apples-to-apples.** The 0.98 ceiling is 12 components; the
  ~0.51 embedding number is 3 components. The matched comparison
  (12-component embedding vs 12-component input) has never been run.
- **The embedding side is historical.** ~0.51 was measured under the old
  protocol — n=1000, unshuffled folds, single RidgeCV. Every input-side
  number above moved substantially when those were changed, and the
  embedding numbers may move too.

Everything measured this session is **input-side only**. The four-stage
comparison requires the backbone forward passes, which is step 1.

The reframe survives either way: not "did the signal survive?" but **"what
fraction of a near-perfectly-recoverable signal does the backbone
preserve?"** Report every stage as a **recovery ratio against the measured
input ceiling**.

### Protocol effects, decomposed

An earlier draft attributed +0.11 on the 2-comp baseline to "protocol".
That conflated two changes. Correctly separated:

| 2-comp input baseline | R² | attributable to |
|---|---:|---|
| recorded (n=1000, z-scored, RidgeCV cv=5) | 0.7060 | — |
| all 4,716 spectra | 0.7556 | +0.05 sample size |
| drop the z-scoring | 0.8179 | +0.06 normalization |

### Diagnostics that came back clean

- **The 168 repeated label levels are not leakage.** Same-label spectra are
  no more similar than different-label ones (cosine 0.9848 vs 0.9842), and
  `GroupKFold` on label value reproduces random-KFold R² to three decimals
  (0.401/0.409, 0.816/0.817, 0.9886/0.9886). Keep as a cheap guard.
- **No block structure in the spectrum index.** Shuffled vs contiguous
  folds agree to ≤0.002 across four independent checks (z-scored OLS
  0.6649/0.6670, RidgeCV 0.7556/0.7577; raw OLS 0.8307/0.8318, RidgeCV
  0.8168/0.8179). Fold shuffling is hygiene, not correctness.
- **comp14 ≡ comp20 and comp15 ≡ comp21 on 100% of spectra.** 12 unique
  components, not 14.

### More data is NOT free R² — the learning curve inverts by probe

2-comp, z-scored, 5-fold OOF, 5 independent draws per size:

| n | OLS | RidgeCV |
|---:|---:|---:|
| 500 | 0.7352 ± 0.027 | — |
| 1000 | 0.7205 ± 0.010 | 0.7060 |
| 2000 | 0.6800 ± 0.011 | — |
| 3000 | 0.6709 ± 0.004 | — |
| 4716 | **0.6649** | **0.7556** |

**OLS degrades monotonically with more data**, well outside the error bars,
while RidgeCV *improves* over the same range. They cross. A single linear
map fitting the whole population worse as more of it arrives points at a
heterogeneous or outlier-heavy population — which is a concrete target for
steps 2–3 (non-linear / mixture / robust models), and a warning that any
number quoted at one n is a point on a curve.

Two consequences for this plan:

- **H1 as originally written ("more data is free R²") is false.** Sample
  size interacts with probe choice; step 1 reports the curve rather than a
  direction.
- **The OLS-beats-embedding claim is unverified at full n.** 0.5079 at
  1-comp was measured at n=1000; OLS falls with n. Step 1 must re-measure
  the whole probe panel at n=4,716 before any claim about T6 is made.

Worth testing in step 3: robust loss / outlier trimming, and whether the
decline tracks specific spectra (a heterogeneity diagnostic).

---

## Split design

Settled empirically, and it answers both halves of the question.

### The split

**Sample unit is the spectrum**, so n=4,716 — not 66,024 wavs.

**Never a single holdout.** 20 random 80/20 splits at 2-comp give
R² = 0.6596 ± 0.0115, range 0.6368–0.6799 — a **0.043 spread from the seed
alone**, larger than the ΔR²≈0.03 margins this metric decides. (The
existing code already uses pooled K-fold, not a holdout, so this is a
guard-rail for the new code, not a criticism of the old.)

**Pooled K-fold OOF is 23× tighter, and fold count / stratification barely
matter** (2-comp OLS, 5 repeats each):

| Splitter | R² | repeat SD |
|---|---:|---:|
| single 80/20 holdout | 0.6596 | 0.0115 |
| KFold(5) shuffled | 0.6652 | **0.0005** |
| KFold(10) shuffled | 0.6655 | 0.0004 |
| StratKFold(5) on y-decile | 0.6652 | 0.0009 |

Stratifying on y-deciles *increases* variance slightly and 10-fold buys
nothing over 5-fold, so **neither is adopted** — plain repeated shuffled
5-fold it is.

**Protocol: repeated shuffled 5-fold, paired across models.**

- `KFold(5, shuffle=True)`, **5 repeats** with different seeds. Report mean
  ± SD across repeats.
- **Pair comparisons on identical folds** and difference per-repeat.
  Measured benefit: resolvable ΔR² goes 0.0016 → 0.0010, a ~1.6× gain.
  Cheap and worth doing, but *not* the big win — the 23× win is pooled
  K-fold over a single holdout.
- **Report two uncertainties, labelled.** The ±0.0005 repeat SD measures
  *split-assignment noise only* on this fixed dataset — right for "model A
  beats model B here". For "this R² generalizes to another 4,716 spectra",
  bootstrap over spectra; that interval is nearer ±0.011. Never quote the
  tight number for the broad claim.
- **Nested tuning.** Any hyperparameter chosen on the inner training fold
  only. `RidgeCV`-inside-`cross_val_predict` already does this; XGBoost
  must follow or its R² is optimistically biased.
- **Always sweep n and probe.** The learning curve is probe-dependent
  (below), so any single-(n, probe) number is a point on a surface.

### Statistical noise was never the problem — systematics are

| Source of variation | magnitude (2-comp unless noted) |
|---|---:|
| probe choice (1-comp) | **0.27 R²** |
| sample size n (OLS) | 0.07 R² |
| normalization | 0.06–0.17 R² |
| split noise, pooled repeated 5-fold | 0.0005 R² |

Systematics dominate split noise by two to three orders of magnitude.
**This corrects an earlier assumption in this plan:** T8's A-vs-B at
ΔR²≈0.03 was *not* "decided inside the noise" — with resolvable ΔR²≈0.001
that margin is ~30× above split noise, and the systematics largely cancel
within one A/B run at fixed probe/n/normalization. T8's internal comparison
is probably sound.

Where systematics *do* bite is comparing **across** measurements taken
under different settings — precisely the documented T8 inconsistency (3c
flipping between 0.186 and 0.240 between two different probes), and any
comparison of an embedding number against an input baseline drawn from a
different probe regime.

So the value of this work is **not** error bars. It is **pinning one
protocol** so numbers become comparable at all, and reporting the
probe/n/normalization surface instead of one arbitrary point on it.

### The components

**The wide layout already makes grouping automatic.** A spectrum is one row
with components as concatenated columns, so components physically *cannot*
straddle a split. This is what fixed the historical `input R² ~0.006` bug
(commit `bf94422`, which had treated component files as independent
samples). Preserve it.

1. **Drop comp20 and comp21** — exact duplicates of comp14/comp15 on 100%
   of spectra. Zero information, +490 dims in an already p-heavy problem.
2. **If the long layout is ever used** (component = row, 56,592 samples),
   `GroupKFold(groups=spectrum_id)` is mandatory — otherwise a sibling
   component with the identical label sits in training.

   Caveat worth recording, because it is counterintuitive: grouped and
   ungrouped both score 0.0027 with pooled OLS. The leak is real but
   **undetectable with a model too weak to exploit it** — the pooled linear
   map cannot fit non-exchangeable components at all. So the absence of
   inflation here is not evidence that grouping is unnecessary. Verify the
   guard with a model that can actually fit (per-component models,
   non-linear probes, embeddings).
3. **Components are NOT exchangeable — measured, and it is decisive:**

   | 12-component formulation | R² |
   |---|---:|
   | **WIDE** (spectrum = 1 row, components concatenated) | **0.9448** |
   | LONG, per-component rows, one shared model | 0.0027 |
   | LONG, predictions averaged per spectrum | **0.0058** |

   **That 0.0058 reproduces the historical bug exactly** — `TASKS.md`:
   *"the previous loader sampled component files as independent samples —
   input R² ~0.006."* Both the bug and the fix now reproduce from first
   principles, and the mechanism is finally explained: each component has
   its own distribution and its own relation to the label, so no single
   shared map from "some component's 245-vector" to the label exists. comp0
   alone scores 0.41–0.51 when it is the only input; pooling all 12 into
   one model collapses to zero. The label lives in *which component is
   which* and in cross-component structure.

   Therefore: **the naive long layout is dropped from step 4.** The
   salvageable variant is *12 separate per-component models, stacked*,
   which preserves component identity. The expected winner remains wide +
   explicit cross-component ratio features + dimension reduction.

   **This also raises the bar on the embedding side.** Embedding each
   component independently and concatenating is the wide layout and is
   fine. But any pooling that averages *across* components is the long
   failure mode in disguise — it would look like "the backbone destroyed
   the signal" when it is really the feature construction. Step 4 checks
   this explicitly.
4. **Component availability.** Report the R²-vs-component-count curve
   rather than one number, plus a component-dropout robustness check.

---

## Design

New self-contained subpackage `code/eval/label_probe/`, rather than edits
to `label_regression.py` — the historical eval stays byte-identical so the
R²=0.44 lineage remains reproducible.

```
code/eval/label_probe/
  __init__.py
  taps.py        four stage taps in ONE forward pass, with pooling options
  cache.py       .npz cache of extracted reps, keyed by ckpt+n+comps+pooling
  protocol.py    split schemes + paired fold-wise aggregation
  regressors.py  registry: ols | ridge | ridgecv | xgb | hgb | knn | mlp | pls | dummy
  features.py    DS features (step 3) + feature-vector variants (step 4)
  study.py       run_step1..run_step4, each writing its own REPORT.md
  plots.py       the four-stage figure + companions
  __main__.py    CLI: python -m eval.label_probe --step N ...
```

Constraint from `EVAL_OVERVIEW.md:461-468`: **everything under `code/eval/`
must import without fairseq.** Use `CheckpointLoader`, never
`code/model_loader.py`.

### Reuse (do not re-implement)

| Need | Existing code |
|---|---|
| Load labelled spectra grouped by component | `code/eval/data_loader.py:154` `load_labeled_data(..., comps=)` — returns `[N,k,245]` + `[N]`, groups by `(dataset, spec)` |
| Load the checkpoint | `code/eval/checkpoint_loader.py:188` `CheckpointLoader.from_file` |
| Stage-tap reference | `structured_similarity.py:189-236` `_extract_all_representations` — costs **two** FE passes/batch; the hook version needs one |
| Hook pattern | `signal_reconstruction.py:401-413` |
| fairseq-style normalization | `label_regression.py:32-35` `_normalize_like_fairseq` |
| Probe + metric set | `code/eval/metrics.py:181-278` — extend, don't fork |
| Figure saving | `code/eval/report.py:24` `_save_fig`; `report.py:724-781` for idiom |

### Files modified (small, backwards-compatible)

- `code/eval/metrics.py` — add `ols`, `xgb`, `hgb` branches. `ridge`
  untouched.
- `code/eval/requirements.txt` — `xgboost>=2.0` as an optional extra.
- `code/eval/EVAL_OVERVIEW.md` — document the new method as #8.
- `TASKS.md` — ledger entry; record the orphaned 2026-08-21 numbers, the
  `valid.tsv ⊃ train.tsv` hazard, and the probe/normalization findings.

---

## Step 0 — Branch and environment

```bash
git checkout -b label-regression-dev
/mnt5/noy/miniconda3/envs/spectralfm_env/bin/python3 -m venv \
    --system-site-packages /mnt5/home/hadar/nova/.venv-labelprobe
/mnt5/home/hadar/nova/.venv-labelprobe/bin/pip install xgboost
```

noy's env is not writable by `hadar`, so a venv is the only option;
`--system-site-packages` inherits torch 2.8 / transformers 4.57 / sklearn
1.7.2 read-only and adds only the 132 MB xgboost wheel (verified: 3.2.0
reachable). `regressors.py` falls back to `HistGradientBoostingRegressor`
if `import xgboost` fails.

---

## Step 1 — Four-stage baseline

`taps.py::extract_stage_reps` registers hooks on `model.feature_extractor`
and `model.feature_projection.projection`, then makes **one**
`model(input_values=...)` call per batch:

| Stage | Source | Shape (per comp) | Pooled |
|---|---|---|---|
| `input` | the signal itself | `[245]` | `[245]` |
| `fe` | `feature_extractor` hook | `[512, 47]` | `[512]` |
| `proj` | `feature_projection.projection` hook | `[47, 768]` | `[768]` |
| `transformer` | `out.last_hidden_state` | `[47, 768]` | `[768]` |

`out.extract_features` (post-LN FE, `[47,512]`) captured as a fifth
diagnostic column, since `structured_similarity` taps the *pre*-LN output.

Reps for 4,716 spectra × 12 components extracted once, cached to `.npz`;
steps 2–4 become CPU-only re-runs.

**Grid:** 4 stages × {1c, 2c, 3c, 12c} × {OLS, RidgeCV, Ridge(1.0)} ×
{raw, z-scored} × protocol {primary, legacy, label_group}.

The probe panel is not optional — the evidence above shows a single probe
choice moves the input baseline by up to 0.27 R². **Always report the best
linear baseline**, never one arbitrarily-regularized number.

**Report:** stage table with paired error bars; recovery ratio vs the
measured input ceiling; legacy-vs-primary delta; the learning curve; and an
explicit verdict on whether emb R²=0.4412 still beats the honest input
baseline.

### The headline figure — `label_reg_stages.png`

- **Left (headline):** x = four stages in forward order, y = R², one line
  per component config, error bars = ±1 SD over folds×repeats, dashed rule
  at each config's best input R².
- **Right:** same x, one line per regressor family (OLS / RidgeCV /
  XGBoost), at the best component config.

Regenerated at the end of steps 2 and 4. Load the `dataviz` skill before
writing plotting code.

---

## Step 2 — XGBoost regressor

Add `xgb` and `hgb` branches to `compute_linear_probing_metrics`; both flow
through the existing `cross_val_predict` path and inherit the full metric
set.

State the caveat up front: at 3 comps the embedding vector is 2,304-dim, at
12 comps 9,216-dim, against n=4,716. **p ≫ n is XGBoost's weakest regime
and Ridge's strongest** — and linear already reaches 0.9886 at the raw
input, leaving little headroom there. XGBoost's real value is at the
*embedding* stages, where the linear probe is far from ceiling.

Search inside the training fold only (`max_depth ∈ {3,6,10}`,
`n_estimators ∈ {300,1000}` with early stopping, `lr ∈ {0.03,0.1}`);
report on the full vector and on PCA-256.

**Report:** paired ΔR² (xgb − ridge) per stage per config, the p≫n effect,
and FE-stage feature importances (512 interpretable channels).

---

## Step 3 — Investigate and crack the problem

Given the near-perfect input recoverability, this narrows from "can it be
cracked" to "why does the backbone discard a signal sitting in plain
sight." Report as "hypothesis → paired ΔR² → verdict".

**Data-science pass:**

- **H1 — sample size × probe interaction.** *Measured at the input: OLS
  falls 0.735→0.665 as n goes 500→4,716 while RidgeCV rises 0.706→0.756.*
  Repeat per stage. Then chase the cause: robust loss / outlier trimming,
  and whether the decline concentrates in identifiable spectra
  (heterogeneity diagnostic). If the population is a mixture, that alone
  argues for the non-linear models in H7.
- **H2 — normalization.** *Already measured: −0.165 R² for OLS at 2-comp.*
  Quantify per stage; add back `(mean, std, log-energy, max, argmax)` per
  component and measure recovery.
- **H3 — component budget.** Ladder 1/2/3/7/12, comp20/21 dropped. Carries
  the input from 0.41 to 0.99.
- **H4 — mean-pooling is the bottleneck.** Raw input keeps 245 ordered
  points; the embedding keeps a 768-dim time-average. Compare `mean`,
  `mean+std`, `mean+max+min`, 4-segment (47→4×768), first/last token.
- **H5 — 168 discrete levels.** Leakage ruled out; what remains is the
  modelling reframe — ordinal classification with expected-value decoding,
  k-NN among levels, rank/quantile target transforms.
- **H6 — comparability, not error bars.** *Measured: split noise is 0.0005,
  three orders below the systematics.* So the job is not "add error bars"
  but "report the probe × n × normalization surface", and re-judge which
  recorded cross-measurement comparisons (e.g. T8's 3c flip between two
  probes) were comparing like with like at all.
- **H7 — non-linearity.** XGBoost, kernel ridge (RBF), PLS, small MLP, to
  separate "no signal" from "no *linear* signal" at the embedding stages.
- **H8 — cross-component structure is the carrier.** 1→2 components takes
  raw R² from 0.41 to 0.83. Test explicit pairwise features: `c_i − c_j`,
  `log(E_i/E_j)`, inter-component correlations — and whether the same
  features built from *embeddings* recover what concatenation misses.
- **H9 — the ceiling.** Report every stage as a fraction of the measured
  input ceiling. Estimate the label's own quantization floor (168 levels,
  ~28 spectra each).

**Then one deep method** (gated on DS results): a trained pooling head on
the **frozen** 47×768 sequence — attention pooling + 2-layer MLP, same fold
structure, early-stopped on an inner split. No backbone training: the
question is whether the frozen representation kept the signal, and
fine-tuning would erase the question. One 2080 Ti.

---

## Step 4 — Feature-vector expressivity

| Axis | Levels |
|---|---|
| Component count | 1, 2, 3, 7, 12 |
| Cross-component combination | concat · concat(mean,std) · pairwise-diff augmented · per-component models stacked |
| Temporal pooling | mean · mean+std · mean+max+min · 4-segment · attention |
| Dimensionality | full · PCA-256 · PLS-64 |

**Excluded by measurement:** the naive long layout (component = row, one
shared model) collapses to R²≈0.006 — components are not exchangeable.
Include it as a deliberate negative control so the report shows the failure
mode rather than asserting it.

**Correction (measured 2026-09-08):** an earlier draft of this section also
lumped *mean-over-components* into that collapse. That was an
over-generalization. The R²≈0.006 result is for pooling **raw signal** into
one shared model; averaging **embeddings** across components gives 0.661 at
3-comp (vs 0.793 concat) — lossy but nowhere near collapse. The
exchangeability finding does not transfer from raw signal to embeddings.

Staged, not a full cross-product: fix the others at the step-1 default,
sweep one axis, keep the winner, move on. ~50 fits on cached features.

**Report:** the winning feature vector, paired ΔR² vs the current
`(0,1)`-concat default, and a recommendation for `COMP_CONFIGS` — applied
only if you want the historical eval changed.

---

## Step 5 — Few-shot / label-efficiency regime (added 2026-09-08)

**Why this exists.** The client's deployment constraint is **~20 labeled
samples at most**. Steps 1-4 all measure something different: how much
label information a probe recovers given ~4,716 labels. That is an
*asymptotic representation-quality* question. "Can a deployment with 20
labels predict this?" is a *sample-efficiency* question, and the two do not
have to agree.

Concretely, extrapolating step 1's numbers to n=20 is not defensible:

- A single 80/20 holdout at n=4,716 already swings R² by ±0.043 across
  seeds (measured, see "Split design"). At n=20 the estimator variance
  dominates completely.
- At n_train≈16 the feature vector is 245-dim (raw 1-comp) to 9,216-dim
  (12-comp transformer). Every probe in steps 1-4 is far past the point of
  being identifiable at that n, so the step 1-4 ranking carries no
  information about the n=20 ranking.
- k-fold CV itself degenerates (5-fold on 20 samples = train on 16, test
  on 4). The protocol has to change, not just the sample size.

A high-dimensional representation is **not automatically sample-efficient**
just because it is informative in aggregate. If the label-relevant signal
sits in a few effective directions, the embedding could need far fewer
labels than raw input; if it is spread diffusely, the embedding could be
*worse* at n=20 while still winning decisively at n=4,716. That is the
open question, and it is testable.

### Protocol (deliberately different from steps 1-4)

- **Repeated random subsampling, not k-fold.** For each
  `n_train ∈ {10, 15, 20, 30, 50, 100}`: draw `n_train` spectra at random,
  fit, evaluate on a large fixed held-out set (1,000 spectra, disjoint from
  every train draw). **Many draws per n (default 100)** -> report the
  *distribution* (median, IQR, 10th/90th pct), never a point estimate.
- **Capacity matched to n.** Probes at this scale must be low-capacity:
  - `ridge_strong` — RidgeCV with an alpha grid biased high, selected by
    LOO within the tiny training set.
  - `pca{k}_ridge` for k ∈ {1,2,3,5} — PCA fitted on **unlabeled** data
    (all 4,716 spectra's representations; PCA uses no labels, and in
    deployment you *do* have unlabeled spectra), then a tiny probe on k
    components. This is the main "use the unlabeled corpus" lever.
  - `pls{k}` for k ∈ {1,2} — supervised, so fitted inside the train draw only.
  - `knn{k}` for k ∈ {1,3,5} — uses representation geometry directly rather
    than fitting coefficients.
  - `dummy` — mean predictor, the floor. At these n it is a real contender
    and must be shown.
- **Inductive vs transductive, both reported.** PCA-on-all-unlabeled is
  legitimate and deployment-realistic, but it is a different assumption
  from PCA-on-the-20-training-points. Report both; label which is which.
  Never quietly use unlabeled data and call it a 20-sample result.
- **Metrics.** Median held-out R², plus MAE and Spearman (R² is hard to
  read when it goes sharply negative, which it will at n=10), plus
  **`frac_positive_r2`** — the fraction of draws beating the mean
  predictor. That reliability number is what a client actually needs:
  "does this work at all, how often".
- **Symmetry.** Raw input and every backbone stage get the identical probe
  family and identical draws. Otherwise the comparison is rigged.

### Headline output

`label_efficiency.png` — x = `n_train` (log), y = median held-out R² with
an IQR band, one line per stage, horizontal rule at each stage's
asymptotic (step 1) R². Plus the number the client actually wants:
**labels needed to reach 50% / 80% of each stage's asymptotic ceiling.**

`step5_REPORT.md` states plainly whether, at n=20, any backbone stage beats
raw input — and if the answer is "they're all within noise of the mean
predictor", it says that too rather than reporting a flattering median.

### What this can and cannot settle

It settles which representation is more label-efficient under a fixed,
honest small-n protocol. It does **not** validate a deployment pipeline:
real deployment would also need to handle which 20 samples get labeled
(active/stratified selection beats random draws, and would be the obvious
follow-up), distribution shift, and calibration. Those are out of scope
here and should be named as such rather than implied.

---

## Verification

```bash
cd /mnt5/home/hadar/nova/SpectralFM-label-regression/code   # `eval` is the package root
V=/mnt5/home/hadar/nova/.venv-labelprobe/bin/python
CKPT=/mnt5/noy/SpectralFM/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt
LAB=/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data

$V -m eval.label_probe --step 1 --checkpoint $CKPT --labeled_data_dir $LAB \
   --max_samples 200 --comps 0 1 --seeds 0 --device cuda      # smoke, ~1 min
$V -m eval.label_probe --step 1 --checkpoint $CKPT --labeled_data_dir $LAB \
   --output_dir /mnt5/home/hadar/nova/SpectralFM-label-regression/code/eval_outputs/label_probe/
# NOTE (found at execution time): /mnt5/noy/ is NOT writable by non-noy users
# despite rwxrwxr-x -- it's group-owned by group 'noy', and hadar is not a
# member. Outputs live under the repo's own code/eval_outputs/ instead.
```

Gates, in order:

1. **Legacy row reproduces history.** `--protocol legacy --probe ridgecv`
   must give input 0.3772 / 0.7060 / 0.7568 and transformer ≈0.441 / 0.475
   / 0.506. Verified reproducible this session. If it fails, the tap or
   normalization is wrong — stop.
2. **Probe panel reproduces this session.** n=1000 z-scored OLS must give
   0.5079 / 0.7421 / 0.7698; Ridge(1.0) 0.2376 / 0.5414 / 0.5792.
3. **Full-set anchors.** 2-comp raw OLS ≈0.8318, RidgeCV ≈0.8179; 12-comp
   raw RidgeCV ≈0.9886.
4. **Tap shapes** `[B,512,47]`, `[B,47,768]`, `[B,47,768]`, and one-pass
   extraction agrees with `_extract_all_representations` to <1e-5.
5. **`dummy` regressor R² ≈ 0** under every protocol.
6. **`label_group` ≈ `primary`** at the input stage (0.9886 vs 0.9886).
7. **Long layout reproduces the historical bug: R² ≈ 0.006.** Note that
   grouped and ungrouped both give 0.0027 — pooled OLS across
   non-exchangeable components cannot fit, so it cannot exploit the sibling
   leak either. Do **not** expect ungrouped to score higher here; that is
   not evidence of a broken guard. Grouping still becomes mandatory the
   moment a model that *can* fit is used (per-component models, non-linear
   probes, embeddings) — test the guard there, not with pooled OLS.
8. **No fairseq import.**
9. `git diff` on `label_regression.py` is empty.

## Deliverables

- Branch `label-regression-dev`; `code/eval/label_probe/`; four small
  modifications above.
- Four short `REPORT.md` files + a top-level `SUMMARY.md`.
- `label_reg_stages.png`, regenerated at steps 1, 2 and 4.
