# Step 4 supplement — deployment-relevant axes (1-comp and 3-comp)

Stage: `transformer`. RidgeCV, primary protocol, 3 repeats.

`run_step4` sweeps its axes at the *best* component count (always 12), but 12-comp is at ceiling (linear reaches 0.989) so those axes have no headroom there. The client expects 1-3 components, so this rerun puts the combination/dimensionality choices where they actually matter.


## 1-component

| variant | R² |
|---|---|
| pls_64 | +0.7096 |
| concat | +0.6301 |
| pca_256 | +0.5807 |
| pls_16 | +0.3982 |
| pca_50 | +0.3347 |
| pca_20 | +0.2413 |
| pls_5 | +0.1915 |
| pca_10 | +0.0720 |
| pls_2 | +0.0616 |
| pls_1 | +0.0431 |
| pca_5 | +0.0404 |
| pca_2 | +0.0140 |

## 3-component

| variant | R² |
|---|---|
| pls_64 | +0.8468 |
| concat(mean,std) | +0.7934 |
| concat | +0.7934 |
| mean_over_comps (negative control) | +0.6606 |
| pca_256 | +0.5655 |
| pls_16 | +0.5528 |
| pca_50 | +0.4710 |
| pls_5 | +0.3894 |
| pca_20 | +0.3108 |
| pls_2 | +0.2269 |
| pca_10 | +0.2089 |
| pls_1 | +0.1397 |
| pca_5 | +0.0661 |
| pca_2 | +0.0596 |
| pairwise_only | +0.0570 |