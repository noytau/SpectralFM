# Label Regression over the SSL Embedding — Findings

## Question

Does a frozen embedding from the SSL backbone (`data2vec-audio`, checkpoint
`runai_long_train_2026-02-25_13-46-46.pt`) support regression of a scalar
label (`parameter_0`, `labeled_data` subset, n=4,716 spectra) well enough to
serve as a **few-shot foundation-model probe** — measured against regressing
directly on the raw 245-point spectrum?

## Method

- **Data:** `labeled_data` subset, n=4,716 usable spectra (12 unique
  components; 2 exact-duplicate components dropped).
- **Backbone:** frozen SSL checkpoint, embeddings extracted once as a
  per-layer, mean-pooled feature bank (all 13 transformer layers plus the
  feature-extractor stage).
- **Readout selection:** the embedding readout (which layer, which
  normalization) is not assumed — it is selected by a large-n RidgeCV screen
  over candidate (layer, normalizer) pairs, and the winner is reported
  alongside a fixed reference point: the layer-12 (final layer), mean-pooled,
  standardized readout, which is the conventional choice elsewhere in this
  codebase's evaluation tooling.
- **Raw input** is compared after whitening (decorrelating/rescaling the
  245-dimensional feature space; label-free, fit only on the inputs).
- **Probe:** RidgeCV, 5-fold cross-validation.
- **Label-efficiency ladder:** the comparison is run not just at the full
  pool but at n_train ∈ {10, 20, 50, 100, 200, 500, 1,000, 2,000}, with
  repeated random draws at each n (100 draws at n≤50, down to 3 at n=2,000)
  to report a median and interquartile range rather than a single point
  estimate.
- **Leak check:** every reported R² is paired with a shuffled-label canary —
  the identical pipeline run with permuted labels — which must collapse to
  ≈0. All canaries passed (see Results).
- **Components:** 1, 2, and 3-component concatenated feature vectors were
  tested, since results differ meaningfully by component count.

### Why raw input needs a whitened readout to measure fairly

Raw input's 245 features are highly collinear (adjacent spectral bins carry
almost the same information). A standard shrinkage-based linear probe
(RidgeCV) applies uniform shrinkage across all feature directions — which
happens to wash out signal that lives specifically in the thin, near-
degenerate directions created by that collinearity. Whitening addresses this
directly: it decorrelates and rescales the feature space, label-free, before
the probe ever sees it, so RidgeCV's shrinkage no longer falls on exactly the
directions where the signal lives.

A supervised alternative, partial least squares (PLS), solves the same
underlying problem by a different route worth understanding even though it
is not the recipe used here: PLS looks at features and labels together and
constructs a small number of new axes explicitly chosen to correlate with the
label, which are no longer collinear with each other. Ordinary linear
regression on those few well-behaved axes then recovers signal RidgeCV alone
never could — not because the regression got smarter, but because the hard
geometric problem (collinearity) was already solved by the projection before
regression saw the data. This effect is real and large at this sample size
(PLS and whitening reach comparable raw-input ceilings at n≈4,716); whitening
is preferred here because it is unsupervised and therefore free of any risk
of a supervised-projection leak, and it is simpler to apply consistently
across every sample size in the ladder.

## Results

### Readout screen (1-component, full pool)

| Layer | Standardized | Whitened |
|---|---:|---:|
| 0 | **0.858** | 0.736 |
| 1 | 0.850 | 0.736 |
| 2 | 0.854 | 0.735 |
| 3 | 0.828 | 0.723 |
| 12 (final layer) | 0.680 | 0.578 |

The best embedding readout (layer 0, mean-pooled, standardized) is
substantially stronger than the conventional final-layer/mean-pool tap used
elsewhere in this codebase — a 0.18 R² gap at 1-component. **Readout choice
matters as much as the raw-vs-embedding comparison itself**; every result
below uses this selected best readout for the embedding, plus the
conventional final-layer readout as a labeled reference line.

### Label efficiency (median held-out R², repeated draws)

**1 component:**

| n_train | Raw input (whitened) | Embedding (best readout) | Embedding (final-layer, conventional) |
|---:|---:|---:|---:|
| 10 | 0.039 | −0.236 | −0.208 |
| 20 | 0.210 | −0.172 | −0.146 |
| 50 | 0.535 | −0.099 | −0.073 |
| 100 | 0.699 | 0.108 | 0.068 |
| 200 | 0.762 | 0.266 | 0.169 |
| 500 | 0.797 | 0.537 | 0.336 |
| 1,000 | 0.810 | 0.688 | 0.511 |
| 2,000 | 0.812 | 0.802 | 0.563 |
| 4,716 (full pool) | 0.833 | **0.854** | 0.680 |

**2 components:**

| n_train | Raw input (whitened) | Embedding (best readout) | Embedding (final-layer, conventional) |
|---:|---:|---:|---:|
| 10 | 0.001 | −0.209 | −0.232 |
| 50 | 0.456 | 0.180 | 0.111 |
| 200 | 0.872 | 0.399 | 0.336 |
| 1,000 | 0.924 | 0.747 | 0.615 |
| 4,716 (full pool) | **0.937** | 0.880 | 0.790 |

**3 components:**

| n_train | Raw input (whitened) | Embedding (best readout) | Embedding (final-layer, conventional) |
|---:|---:|---:|---:|
| 10 | −0.021 | −0.237 | −0.220 |
| 50 | 0.353 | 0.161 | 0.098 |
| 200 | 0.902 | 0.409 | 0.338 |
| 1,000 | 0.956 | 0.752 | 0.638 |
| 4,716 (full pool) | **0.964** | 0.881 | 0.817 |

(Intermediate rungs n=20/100/500/2,000 follow the same pattern and are in the
full results file; the tables above show the shape of the curve.)

Full numeric results (every rung, every draw's median/IQR/fraction-positive):
[`code/eval_outputs/label_probe/run_2026-09-11/label_probe_results.json`](../code/eval_outputs/label_probe/run_2026-09-11/label_probe_results.json).
Plots: [`label_efficiency.png`](../code/eval_outputs/label_probe/run_2026-09-11/label_efficiency.png)
(label-efficiency curves with IQR bands) and
[`true_vs_pred_grid.png`](../code/eval_outputs/label_probe/run_2026-09-11/true_vs_pred_grid.png)
(true-vs-predicted scatter, axes fixed to the same [−2, 2] range across every
panel so cells are directly, visually comparable).

### Leak check

Every configuration above was paired with a shuffled-label canary: refit the
identical pipeline on randomly permuted labels. Real R² is large and
positive in every case; shuffled R² is consistently ≈0 (typically between
−0.02 and 0.00), confirming none of the reported numbers are measurement
artifacts.

## What this means

**At 2 and 3 components, raw input beats the embedding at every sample size
tested, from n=10 through the full pool of 4,716.** The gap is largest in the
few-shot regime (e.g. 3-comp, n=50: raw 0.353 vs. embedding 0.161) and never
closes, even at the full pool (raw 0.964 vs. embedding 0.881).

**At 1 component, the picture is more nuanced: raw input leads throughout the
few-shot regime and up to n=2,000, but the embedding overtakes it at the full
pool** (raw 0.833 vs. embedding 0.854 — a real, if modest, crossover). This
is the one regime in this study where the embedding is not simply dominated
by raw input — but it only appears at 1-component and only once labeled data
is plentiful (thousands of samples), which is the opposite of the few-shot
regime the foundation-model framing is meant to target.

**For the stated goal — a few-shot regressor over the embedding as a
foundation-model probe — the practical conclusion is that raw input is the
stronger substrate in the regime that matters (small labeled sets, n≤500),
across every component count tested.** The embedding only closes or reverses
the gap once far more labeled data is available than "few-shot" implies, and
even then only at 1-component.

A secondary, load-bearing finding: **readout choice (which layer, which
normalization) changes the embedding's apparent few-shot performance by a
wide margin.** The conventional final-layer/mean-pool tap consistently
underperforms an early-layer readout (e.g. at n=500, 1-comp: 0.336 vs.
0.537) — any comparison that fixes the readout to the conventional choice
without checking alternatives risks understating the embedding's real
ceiling.

## Recommendations

1. **Use a whitened raw-input baseline and a screened (not assumed) embedding
   readout in any future comparison on this backbone** — both materially
   change the measured gap, and skipping either understates one side unfairly.
2. **Treat the embedding's few-shot performance as currently unproven for this
   task.** Any few-shot-probe work built on this backbone should either
   demonstrate an advantage directly (e.g. via better label selection —
   stratified or active sampling was not tested here; every draw was i.i.d.
   random) or target a task better suited to showing representation-learning
   value. A smooth, low-dimensional, already-well-conditioned 245-point
   signal with a smooth scalar target is a difficult case for demonstrating
   an embedding's advantage over raw features; a task where raw features
   demonstrably fail (multi-component disentangling, anomaly detection,
   cross-instrument or cross-distribution transfer) is a stronger candidate.
3. **Re-run this same protocol on other backbone checkpoints** (in particular
   reconstruction-trained ones) before drawing any comparative conclusions
   about which checkpoint is "better" for this or a similar probing task —
   this study only evaluated the one SSL checkpoint named above.
4. **Distribution shift and calibration are untested** — this study is
   entirely in-distribution (train and eval draws from the same pool).

## Reproducing this

```bash
cd code
/mnt5/noy/miniconda3/envs/spectralfm_env/bin/python3 -m eval.label_probe \
  --checkpoint /mnt5/noy/SpectralFM/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt \
  --data /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data \
  --out_dir eval_outputs/label_probe/<run_name> \
  --device cuda --comps 1 2 3
```

The embedding-feature bank (`bank.npz`, several GB) is cached in `out_dir`
and reused on a rerun with the same `out_dir` — delete it to force
re-extraction. Runtime on one GPU: embedding extraction is a single forward
pass over the full dataset (a few minutes); the readout screen and
label-efficiency ladder together take on the order of a few hours, dominated
by repeated RidgeCV fits at the smaller sample sizes.

Code: [`code/eval/label_probe/`](../code/eval/label_probe/) — fairseq-free,
tested (`pytest eval/label_probe/tests/ --import-mode=importlib` from
`code/`).
