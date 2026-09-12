# Label Regression over the SSL Embedding — Findings

## Question

Does a frozen embedding from the SSL backbone (`data2vec-audio`, checkpoint
`runai_long_train_2026-02-25_13-46-46.pt`) support regression of a scalar
label (`parameter_0`, `labeled_data` subset, n=4,716 spectra) well enough to
serve as a **few-shot foundation-model probe** — measured against regressing
directly on the raw 245-point spectrum?

## The model's blocks, and the terms used below

The backbone has three kinds of block, and every result below is labeled by
which one it reads from:

- **FE** — the 5-layer convolutional feature extractor. Two taps exist:
  **FE (pre-LN)**, its raw output, and **FE (post-LN)**, the same output
  after `feature_projection`'s LayerNorm (not yet through its Linear layer).
- **Projector** — `feature_projection`'s Linear layer (512→768) plus the
  positional convolution and the encoder's own pre-block LayerNorm. This is
  exactly what Transformer layer 1 receives as input.
- **Transformer layer 1 – 12** — the output of each of the 12 Transformer
  blocks, in order. "Transformer layer 12" is the final block's output —
  the conventional tap used elsewhere in this codebase's evaluation tooling.

## Method

- **Data:** `labeled_data` subset, n=4,716 usable spectra (12 unique
  components; 2 exact-duplicate components dropped).
- **Backbone:** frozen SSL checkpoint; a moment-bank of pooled statistics is
  extracted once, per block, per component, covering every block listed
  above.
- **Raw-input baselines (three, all checked):** whitened (PCA-whitening,
  fit label-free on the unlabeled corpus), z-scored, and z-scored + PLS-64
  (partial least squares, fit strictly inside each cross-validation fold —
  see "Why raw input needs care" below for what PLS buys and why).
- **Embedding recipe — searched, not assumed** (this is the part we pushed
  hardest on, per instruction to squeeze out as much signal as the
  embedding can give before comparing it to raw input). Four phases, each
  narrowing from the last:
  1. **Which block** (mean-pooled, both normalizers) — the full scoreboard,
     all 15 blocks, is in Results below.
  2. **Which pooling scheme**, tried only on the phase-1 winner: mean,
     mean+std, mean/max/min, four-segment (the FE's 47 output tokens split
     into 4 contiguous chunks, each pooled separately), first+last token.
  3. **Concatenating the top-2 blocks from phase 1** — does combining beat
     the single best block?
  4. **Which probe**, tried only on the phase 1–3 winner: RidgeCV, PLS at 4
     ranks (8/16/32/64), PCA+Ridge at 3 ranks (10/32/64), gradient
     boosting, k-NN.

  The raw-input side was not searched this deeply — whitening, z-scoring,
  and PLS-64 already cover the levers that matter there (see below).
- **Probe protocol:** RidgeCV, 5-fold cross-validation, unless a phase-4
  search says otherwise for the embedding.
- **Label-efficiency ladder:** every comparison is run not just at the full
  pool but at n_train ∈ {10, 20, 50, 100, 200, 500, 1,000, 2,000, 4,716},
  with repeated random draws at each n (100 draws at n≤50, down to a single
  draw at the full pool) to report a median rather than a point estimate.
- **Leak check:** every reported R² is paired with a shuffled-label canary
  — the identical pipeline run with permuted labels, which must collapse to
  ≈0. All canaries passed.
- **Components:** 1, 2, and 3-component concatenated feature vectors (see
  `code/eval/label_probe/features.py` for why components are concatenated,
  not pooled, on the raw-input side).

**Scope note:** the 4-phase embedding search above was run once, at
1-component, full pool. The winning recipe (block, pooling, normalizer,
probe) is then reused unchanged at 2- and 3-components — it was not
independently re-optimized per component count. This is a reasonable
simplification (the search is expensive) but means the 2–3-component
embedding numbers below are a lower bound on what a per-component-count
search might find, not a fully independent optimum.

### Why raw input needs care to measure fairly

Raw input's 245 features are highly collinear (adjacent spectral bins carry
almost the same information). A standard shrinkage-based linear probe
(RidgeCV) applies uniform shrinkage across all feature directions — which
happens to wash out signal that lives specifically in the thin, near-
degenerate directions created by that collinearity. Whitening addresses
this directly: it decorrelates and rescales the feature space, label-free,
before the probe ever sees it, so RidgeCV's shrinkage no longer falls on
exactly the directions where the signal lives.

A supervised alternative, PLS, solves the same underlying problem by a
different route: it looks at features and labels together and constructs a
small number of new axes explicitly chosen to correlate with the label,
which are no longer collinear with each other. Ordinary linear regression on
those few well-behaved axes then recovers signal RidgeCV alone never could
— not because the regression got smarter, but because the hard geometric
problem (collinearity) was already solved by the projection before
regression saw the data. This effect is real and large at full sample size
(PLS-64 and whitening reach comparable raw-input ceilings at n≈4,716: 0.824
vs. 0.833 at 1-component); whitening is used as the primary baseline here
because it is unsupervised and therefore free of any risk of a supervised-
projection leak, and simpler to apply consistently across every sample size
in the ladder. PLS is reported alongside it as a cross-check, not a
replacement.

## Results

### Block scoreboard (1-component, full pool, mean-pooled, RidgeCV)

Every block was tested — the full ranking, including the final Transformer
layers, not just the winner:

| Rank | Block | Standardized | Whitened |
|---:|---|---:|---:|
| 1 | **Projector** | **0.858** | 0.736 |
| 2 | Transformer layer 2 | 0.854 | 0.735 |
| 3 | Transformer layer 1 | 0.850 | 0.736 |
| 4 | Transformer layer 3 | 0.828 | 0.723 |
| 5 | Transformer layer 5 | 0.811 | 0.692 |
| 6 | Transformer layer 4 | 0.798 | 0.690 |
| 7 | Transformer layer 6 | 0.796 | 0.650 |
| 8 | Transformer layer 8 | 0.769 | 0.606 |
| 9 | Transformer layer 7 | 0.769 | 0.629 |
| 10 | Transformer layer 9 | 0.757 | 0.595 |
| 11 | Transformer layer 11 | 0.737 | 0.600 |
| 12 | Transformer layer 10 | 0.730 | 0.599 |
| 13 | **Transformer layer 12 (final)** | **0.680** | 0.578 |
| 14 | FE (pre-LN) | 0.660 | 0.610 |
| 15 | FE (post-LN) | 0.638 | 0.602 |

Signal decays steadily with Transformer depth for this task — the
Projector (before any Transformer block runs) and the first few Transformer
layers are the strongest taps; the conventional final-layer tap is the
*weakest* Transformer-side option, beaten by every other Transformer layer
and even tied with the raw FE output.

### Squeezing the winning block further

Phase 2 (pooling, on the Projector only): four-segment pooling was clearly
the best scheme (0.918 vs. 0.858 for plain mean-pooling, standardized) —
splitting the sequence into four chunks before pooling captures structure a
single mean discards. Phase 3 (concatenating the top-2 blocks, Projector +
Transformer layer 2): no improvement (0.858, identical to the Projector
alone) — the two blocks carry redundant, not complementary, signal. Phase 4
(probe choice, on Projector/segment4/standardized): plain RidgeCV was the
best of 10 probes tried (0.918) — PLS and PCA+Ridge at every rank tested
underperformed it (0.32–0.84), likely because both discard capacity in
exactly the directions where this recipe's signal is concentrated, the
opposite of the situation on the raw-input side.

**Winning embedding recipe: Projector, four-segment pooling, standardized,
RidgeCV.**

### Full-pool comparison (n=4,716)

| n-comp | Raw (whitened) | Raw (z-scored) | Raw (z-scored, PLS-64) | Embedding (winning recipe) | Embedding (Transformer layer 12, conventional) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.833 | 0.763 | 0.824 | **0.919** | 0.682 |
| 2 | **0.937** | 0.902 | 0.925 | 0.900 | 0.790 |
| 3 | **0.964** | 0.929 | 0.935 | 0.905 | 0.817 |

### Label efficiency (median held-out R², repeated draws)

**1 component:**

| n_train | Raw (whitened) | Raw (z-scored) | Raw (z-scored, PLS-64) | Embedding (winning recipe) | Embedding (conventional) |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.04 | −0.23 | −3.00 | −0.39 | −0.21 |
| 50 | 0.54 | 0.13 | −1.83 | −0.01 | −0.07 |
| 200 | 0.76 | 0.61 | 0.12 | 0.31 | 0.17 |
| 500 | 0.80 | 0.67 | 0.67 | 0.61 | 0.34 |
| 1,000 | 0.81 | 0.70 | 0.77 | 0.78 | 0.51 |
| 2,000 | 0.81 | 0.72 | 0.80 | 0.89 | 0.56 |
| 4,716 (full pool) | 0.83 | 0.76 | 0.82 | **0.92** | 0.68 |

**2 components:**

| n_train | Raw (whitened) | Embedding (winning recipe) |
|---:|---:|---:|
| 10 | 0.00 | −0.26 |
| 50 | 0.46 | 0.21 |
| 200 | 0.87 | 0.41 |
| 1,000 | 0.92 | 0.79 |
| 4,716 (full pool) | **0.94** | 0.90 |

**3 components:**

| n_train | Raw (whitened) | Embedding (winning recipe) |
|---:|---:|---:|
| 10 | −0.02 | −0.35 |
| 50 | 0.35 | 0.20 |
| 200 | 0.90 | 0.43 |
| 1,000 | 0.96 | 0.79 |
| 4,716 (full pool) | **0.96** | 0.91 |

(Full numeric results, every rung and every raw-input variant: see the JSON
linked below.)

Full results:
[`code/eval_outputs/label_probe/run_2026-09-11/label_probe_results.json`](../code/eval_outputs/label_probe/run_2026-09-11/label_probe_results.json).
Plots:
[`label_efficiency.png`](../code/eval_outputs/label_probe/run_2026-09-11/label_efficiency.png)
(label-efficiency curves with IQR bands) and
[`true_vs_pred_grid.png`](../code/eval_outputs/label_probe/run_2026-09-11/true_vs_pred_grid.png)
(true-vs-predicted scatter, axes fixed to the same range across every panel
so cells are directly, visually comparable).

### Leak check

Every configuration above was paired with a shuffled-label canary: refit
the identical pipeline on randomly permuted labels. Real R² is large and
positive in every case; shuffled R² is consistently ≈0 (between −0.06 and
0.00), confirming none of the reported numbers are measurement artifacts.

## What this means

**Pushed as hard as this search pushed it, the embedding's best showing is
at 1-component: it overtakes raw input once there is enough labeled data
(roughly n≥1,500), and by a clear margin at the full pool (0.92 vs. 0.83).**
This is a real result, not a tie — it survived the leak canary and used a
recipe (Projector, four-segment pooling, standardized, plain RidgeCV) found
by a systematic search across 15 blocks, 5 pooling schemes, and 10 probes.

**At 2- and 3-components, raw input wins at every sample size tested,
including the full pool** — the deeper search closed some of the gap
relative to the earlier, mean-pooling-only recipe, but did not close it.

**In the few-shot regime that matters most for the stated goal (n≤500),
raw input is ahead at every component count, including 1-component** — the
embedding's 1-component win only appears once labeled data is plentiful
(thousands of samples), which is the opposite of "few-shot."

**For the stated goal — a few-shot regressor over the embedding as a
foundation-model probe — the practical conclusion is unchanged even after
substantially deepening the embedding-side search: raw input remains the
stronger substrate in the small-labeled-data regime, across every component
count tested.** The embedding's one genuine advantage (1-component, large
n) is real but not a few-shot result.

## Recommendations

1. **If deploying a probe on this backbone at 1-component with ≥1,500
   labeled examples available, use the embedding** — specifically the
   Projector's output, four-segment pooled, standardized, with plain
   RidgeCV. This is the one regime where the search found a genuine,
   leak-checked embedding advantage.
2. **For the few-shot regime (n≤500) at any component count, and for 2–3
   components at any sample size, use whitened raw input.** Whitening is
   free (unlabeled-fit) and consistently the strongest raw-input recipe.
3. **Treat the embedding's few-shot performance as still unproven for this
   task**, even after this deeper search. Either demonstrate an advantage
   directly — e.g. via stratified or active label selection (every draw
   here was i.i.d. random and untested otherwise) or a per-component-count
   recipe search (this study's search ran once, at 1-component, and was
   reused unchanged for 2–3-components) — or target a task better suited to
   showing representation-learning value. A smooth, low-dimensional,
   already-well-conditioned 245-point signal with a smooth scalar target is
   a difficult case for demonstrating an embedding's advantage over raw
   features; a task where raw features demonstrably fail (multi-component
   disentangling, anomaly detection, cross-instrument or
   cross-distribution transfer) is a stronger candidate.
4. **Re-run this same protocol on other backbone checkpoints** (in
   particular reconstruction-trained ones) before drawing any comparative
   conclusions about which checkpoint is "better" for this or a similar
   probing task — this study only evaluated the one SSL checkpoint named
   above.
5. **Distribution shift and calibration are untested** — this study is
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
pass over the full dataset (a few minutes); the 4-phase embedding recipe
search plus the label-efficiency ladders across all three component counts
take several hours, dominated by repeated RidgeCV fits at the smaller
sample sizes and by the phase-1 scoreboard (15 blocks × 2 normalizers, at
full-pool n).

Code: [`code/eval/label_probe/`](../code/eval/label_probe/) — fairseq-free,
tested (`pytest eval/label_probe/tests/ --import-mode=importlib` from
`code/`).
