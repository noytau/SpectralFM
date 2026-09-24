# Step 3 — DS Investigation

## H2 — normalization cost (raw − z-scored best-linear R²)

| n-comp | raw | z-scored | cost |
|---|---|---|---|
| 1 | 0.4089 | 0.4684 | -0.0595 |
| 2 | 0.8311 | 0.7567 | +0.0744 |
| 3 | 0.8668 | 0.8135 | +0.0532 |
| 7 | 0.9348 | 0.9241 | +0.0106 |
| 12 | 0.9885 | 0.9799 | +0.0086 |

## H3 — component budget (best R² per stage per n-comp)

| n-comp | Raw input | Input (z-scored) | Post-FE (pre-LN) | Post-FE (post-LN) | Post-Projection | Post-Transformer |
|---|---|---|---|---|---|---|
| 1 | 0.409 | 0.468 | 0.538 | 0.567 | 0.538 | 0.628 |
| 2 | 0.831 | 0.757 | 0.753 | 0.762 | 0.754 | 0.757 |
| 3 | 0.867 | 0.814 | 0.791 | 0.789 | 0.789 | 0.793 |
| 7 | 0.935 | 0.924 | 0.896 | 0.896 | 0.893 | 0.866 |
| 12 | 0.988 | 0.980 | 0.954 | 0.952 | 0.949 | 0.939 |

## H8 — explicit cross-component features vs plain concat

| stage | concat R² (dim) | pairwise-feature R² (dim) |
|---|---|---|
| input_raw | 0.8633 (735) | 0.3975 (21) |
| transformer | 0.7934 (2304) | 0.0570 (21) |

## H9 — recovery ratio vs input ceiling (R²=0.9885, at 12-comp)

| stage | R² | recovery ratio |
|---|---|---|
| Raw input | 0.9885 | 100.0% |
| Input (z-scored) | 0.9799 | 99.1% |
| Post-FE (pre-LN) | 0.9543 | 96.5% |
| Post-FE (post-LN) | 0.9523 | 96.3% |
| Post-Projection | 0.9485 | 96.0% |
| Post-Transformer | 0.9393 | 95.0% |

## H1, H5, H6, H7 — answered by the pre-implementation diagnostics (see plan.md), not re-run here

- **H1** (sample-size x probe interaction): OLS degrades monotonically with n while RidgeCV improves (measured 0.735->0.665 vs 0.706->0.756, n=500->4716, 2-comp input). See plan.md's learning-curve table.
- **H5** (168 discrete levels): ruled out as a leakage source (`label_group` protocol in step 1 matches `primary` to 3 decimals).
- **H6** (comparability): split noise is 0.0005, three orders below the probe/n/normalization systematics -- the job is reporting the surface, which step 1's grid does.
- **H7** (non-linearity): see step 2's xgb-vs-ridge report.


## Not run this pass

- **H4** (pooling scheme) and the trained attention-pooling head require re-extracting at other poolings -- deferred to step 4 / a follow-up session; not silently skipped.
