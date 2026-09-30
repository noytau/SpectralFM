# Step 5 follow-up — `tabpfn_pca50` (giving the embedding a fair reduction)

The main step-5 run used `tabpfn_pca10`, which the PCA-ceiling diagnostic showed retains only ~11% of the transformer's label signal at 1-comp (ceiling 0.072 vs full-dim 0.628) while raw input retains 77%. So "TabPFN works on input, not embeddings" was an artifact of the reduction. At k=50 the transformer ceiling rises to 0.333 vs raw input's 0.382, so this is the fair comparison.

20 draws, 1000-spectrum held-out eval set, transductive PCA basis.


## n_train = 20

| n-comp | Raw input | Input (z-scored) | Post-FE (pre-LN) | Post-FE (post-LN) | Post-Projection | Post-Transformer |
|---|---|---|---|---|---|---|
| 1 | +0.061 (0.65) | +0.070 (0.85) | -0.011 (0.40) | -0.012 (0.45) | -0.006 (0.45) | +0.010 (0.55) |
| 2 | +0.089 (0.95) | +0.090 (0.75) | +0.025 (0.60) | +0.025 (0.65) | +0.023 (0.70) | +0.004 (0.55) |
| 3 | +0.083 (0.80) | +0.078 (0.85) | +0.019 (0.70) | +0.010 (0.65) | +0.032 (0.60) | +0.010 (0.55) |

## n_train = 50

| n-comp | Raw input | Input (z-scored) | Post-FE (pre-LN) | Post-FE (post-LN) | Post-Projection | Post-Transformer |
|---|---|---|---|---|---|---|
| 1 | +0.241 (1.00) | +0.254 (1.00) | +0.064 (0.85) | +0.046 (0.90) | +0.053 (0.75) | +0.063 (0.85) |
| 2 | +0.400 (1.00) | +0.362 (1.00) | +0.133 (0.90) | +0.120 (0.95) | +0.140 (0.95) | +0.078 (0.90) |
| 3 | +0.458 (1.00) | +0.258 (1.00) | +0.062 (0.90) | +0.068 (0.85) | +0.114 (0.85) | +0.076 (0.90) |

Cells are `median R² (frac of draws with R²>0)`.
