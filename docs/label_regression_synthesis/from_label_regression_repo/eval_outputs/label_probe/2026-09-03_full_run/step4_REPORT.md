> **CORRECTION (2026-09-10):** the `pls_64` entry in axis 3 is inflated by a
> label leak (supervised projection fitted on all labels, then
> cross-validated). Honest value at 12-comp is lower; at 1/3-comp PLS does
> not beat plain concat. See `step6_pls_correction.md`.

# Step 4 — Feature-Vector Expressivity

Staged sweep on cached `transformer`-stage reps: fix two axes at defaults, vary the third, keep the winner.


## Axis 1 — component count

| n-comp | R² |
|---|---|
| 1 | 0.6301 |
| 2 | 0.7528 |
| 3 | 0.7934 |
| 7 | 0.8640 |
| 12 | 0.9378 |

## Axis 2 — cross-component combination

| combination | R² |
|---|---|
| concat | 0.9378 |
| mean_over_comps (negative control) | 0.7378 |
| concat(mean,std) | 0.9376 |
| pairwise_features_only | 0.2743 |

`mean_over_comps` is included as a deliberate negative control (plan.md: exchangeability is false, this formulation is expected to collapse).


## Axis 3 — dimensionality

| reduction | R² |
|---|---|
| full | 0.9378 |
| pca_256 | 0.7161 |
| pls_64 | 0.9523 |

## Verdict

- Historical `(0,1)`-concat default (2-comp): R²=0.7528
- Best found this sweep (12-comp): R²=0.9523
- Raw delta: +0.1995 — but this **conflates two changes** and should not be quoted alone: it compares 12-comp against the 2-comp default, so most of it is the component count, not the feature-vector choice.
- **Matched-component delta (the feature-vector effect alone): +0.0145** (12-comp best vs 12-comp plain concat) — this is the number attributable to the combination/dimensionality choice.


**Dimensionality note:** supervised reduction (PLS) *beats* the full vector while unsupervised reduction (PCA) loses heavily — see axis 3. The label direction is not in the top principal components, so any method built on unsupervised PCA features inherits that loss.


**Component-count caveat:** these axes were swept at the *best* component count from axis 1, which is 12 — but a linear probe already reaches ~0.99 at 12-comp, so there is little headroom there. For the 1-3 component deployment regime see `step4_deployment_axes.md`.
