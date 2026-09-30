# Step 6 — extended PLS sweep, and a label leak it exposed

## The leak

`step4_deployment_axes.py` and `study.py` (axis 3) both did:

```python
PLSRegression(n_components=k).fit_transform(X, y)   # <- fits on ALL of y
```

and *then* cross-validated a RidgeCV on the transformed features. PLS is a
**supervised** projection, so it was fitted using every label — including the
rows that later became test folds. The projection had already seen the test
labels; the cross-validation that followed was measuring a contaminated
feature space.

PCA in the same code is unsupervised (`fit_transform(X)` only), so it leaks no
labels. It is transductive over X, which is a milder and defensible
assumption. **The PCA numbers stand; the PLS numbers did not.**

Measured inflation:

| config | leaky (as published) | honest | inflation |
|---|---:|---:|---:|
| 1-comp pls_16 | +0.398 | +0.378 | +0.020 |
| 1-comp pls_64 | +0.711 | +0.620 | **+0.090** |
| 3-comp pls_16 | +0.552 | +0.525 | +0.027 |
| 3-comp pls_64 | +0.847 | +0.764 | **+0.083** |

Inflation grows with k, which is the expected signature: more components,
more label information smuggled through the projection.

## What it did to the recommendation

The published claim was **"use PLS-64, not concatenation (+0.080 at 1-comp,
+0.053 at 3-comp)"**. Fitting the projection inside the training fold only:

| | honest PLS peak | plain concat | verdict |
|---|---:|---:|---|
| 1-comp | **0.652** (k=96) | 0.630 | +0.022 — marginal |
| 3-comp | **0.789** (k=96–128) | 0.793 | −0.004 — **no gain** |

**The recommendation does not survive.** Supervised reduction gives a marginal
gain at one component and nothing at three. Plain concatenation is fine.

## The extended sweep (the original question)

k=64 was both the largest value swept in step 4 and the winner — the signature
of an un-found optimum. Extending it confirms the curve peaks and then decays
toward OLS-on-full-features, as PLS theory predicts:

| k | 1-comp | 3-comp |
|---:|---:|---:|
| 2 | 0.065 | 0.223 |
| 5 | 0.186 | 0.386 |
| 16 | 0.378 | 0.525 |
| 32 | 0.501 | 0.651 |
| 64 | 0.620 | 0.764 |
| **96** | **0.652** | **0.789** |
| 128 | 0.634 | **0.789** |
| 192 | 0.585 | 0.773 |
| 256 | 0.562 | — |
| 384 | 0.568 | — |

So the optimum is k≈96–128, not 64 — but at that optimum PLS still only ties
plain concatenation at 3-comp.

## Supervised-vs-unsupervised, restated honestly

Step 4 concluded supervised reduction beats unsupervised by 2–6×. With the
leak removed the gap is real but narrower, and it closes as k grows:

| k | 1-comp PLS | 1-comp PCA |
|---:|---:|---:|
| 16 | 0.378 | 0.204 |
| 64 | 0.620 | 0.368 |
| 128 | 0.634 | 0.437 |
| 256 | 0.562 | 0.576 |

At k=256 PCA has caught up and slightly passed PLS (PLS is decaying by then).
The claim that survives is narrower: **supervised reduction dominates at small
k; at large k the two converge, and neither beats plain concatenation by much.**

## Not run, by agreement

* **Autoencoder** — optimises reconstruction, the same objective family as PCA,
  which is measured failing here. This project already ran that experiment:
  TASKS.md T6 found reconstruction-trained backbones scoring ~0 on label
  regression.
* **UMAP** — deliberately distorts global structure; approximate `transform` on
  unseen data. Wrong trade for regression.
* **RRR** — does not apply. It minimises ||Y − XB||² s.t. rank(B) ≤ k, but
  `parameter_0` is scalar, so B is p×1 and rank(B) ≤ 1 unconditionally: the
  constraint is vacuous and RRR reduces to OLS/ridge. It *would* apply if the
  upstream pipeline's other parameters (the name `parameter_0` implies more)
  were recovered, making Y multivariate — worth asking the data owner, and
  plausibly a bigger win than any reducer choice here.

## The check that would have caught it

A shuffled-label canary: permute `y`, run the identical evaluation path, and
require R² ≈ 0. With the labels destroyed there is nothing to predict, so any
meaningfully positive score means label information is reaching the model
through a path other than the training labels.

```
LEAKY pipeline, shuffled labels : R² = +0.1794   <- LEAK DETECTED
FIXED pipeline, shuffled labels : R² = -0.2836   <- PASS
```

The old pipeline scores **+0.18 on randomly permuted labels**. That is
unambiguous, needs no baseline to interpret, and takes seconds to run — it
would have caught this before the recommendation was ever published.

Now available as `python -m eval.label_probe.reducers --canary --out_dir <dir>`.
**Run it whenever a supervised transform (PLS, supervised AE, feature
selection, target encoding) enters the pipeline.** Unsupervised transforms
(PCA, plain AE) cannot leak labels this way, but running the canary anyway is
cheap insurance against the transform accidentally becoming supervised.

## Update 2026-09-10 — the same extended sweep, run on raw input, reopens step 1

Everything above is about PLS on **transformer-stage** embeddings, where the
honest numbers changed the step-4 recommendation but not the step-1 headline.
Running the identical (leak-free, fold-internal, canary-verified) sweep on
**raw input** is a different result entirely, and it does change the step-1
headline.

A separate investigation that session (see the HTML report's §06a) found the
raw-input OLS baseline jumps from 0.404 to 0.824 at 1-comp under float64
arithmetic, and initially treated this as a numerical artifact confined to an
unstable, unregularized solve — not reachable by any practical estimator. That
conclusion was wrong. PLS reaches the identical number:

| comp | raw input, PLS (fold-internal, canary-clean) | best k |
|---:|---:|---:|
| 1 | **0.825** | 64–128 |
| 2 | **0.939** | 96 |
| 3 | **0.969** | 128 |
| 7 | **0.993** | 256 |
| 12 | **0.999** | 384 |

Stable to std=0.0008–0.0009 across 8 fold-seeds, identical under float32 and
float64 (unlike OLS), and shuffled-label canary at every k tested collapses to
≈0 (e.g. 3-comp k=128: shuffled R²=−0.080). This is not the OLS artifact —
it's real signal that RidgeCV (used for every headline number in this study)
cannot reach, because ridge's isotropic shrinkage penalizes low-singular-value
directions the hardest, and this signal lives exactly in the near-degenerate
directions created by highly collinear adjacent spectral bins. PLS selects
directions by covariance with the label instead, and finds it directly.

Against these numbers, the transformer-stage best-of-panel (RidgeCV, with PLS
folded in per the table above) is **0.645 / 0.763 / 0.793 / 0.866 / 0.945** at
1/2/3/7/12 comp — the backbone now recovers **78–95%** of the corrected
ceiling, monotonically rising with component count, with **no component count
at which the backbone beats raw input.** The step-1 "backbone wins at
1-comp" result was raw input's own ceiling being underestimated by RidgeCV,
not a real property of the backbone. See the HTML report §06a for the full
write-up; `input(z)`/FE/proj columns are spot-checked at 1 and 3 comp only
(both show comparably large corrections) and still need a full re-sweep.
