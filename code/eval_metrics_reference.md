# Evaluation Metrics Reference for SpectralFM

This document catalogs all possible evaluation metrics for representation quality,
organized by task type. Metrics marked with `[v1]` are planned for initial implementation.

---

## 1. Labeled Regression (parameter_0 prediction)

**Task:** Given a representation (input or embedding), predict the continuous `parameter_0` value.

### Error metrics (lower = better)

| Metric | Formula | Description | sklearn |
|---|---|---|---|
| `[v1]` **MSE** | `mean((y - ŷ)²)` | Average squared error, penalizes large errors | `mean_squared_error` |
| **RMSE** | `sqrt(MSE)` | Same units as parameter_0 (interpretable) | `sqrt(mean_squared_error)` |
| **MAE** | `mean(|y - ŷ|)` | Average absolute error, robust to outliers | `mean_absolute_error` |
| **Max Error** | `max(|y - ŷ|)` | Worst-case prediction | `max_error` |
| **Median AE** | `median(|y - ŷ|)` | Robust central error | `median_absolute_error` |

### Goodness-of-fit (higher = better)

| Metric | Range | Description | sklearn |
|---|---|---|---|
| `[v1]` **R²** | (-inf, 1] | Fraction of variance explained. 1=perfect, 0=mean-baseline, <0=worse than mean | `r2_score` |
| **Adjusted R²** | (-inf, 1] | R² corrected for number of features (penalizes overfitting) | manual |
| **Explained Variance** | (-inf, 1] | Like R² but ignores constant bias in predictions | `explained_variance_score` |

### Correlation (higher magnitude = better)

| Metric | Range | Description | scipy |
|---|---|---|---|
| `[v1]` **Pearson r** | [-1, 1] | Linear correlation between predicted and true values | `pearsonr` |
| `[v1]` **Spearman rho** | [-1, 1] | Rank correlation — does the ordering match? Works for nonlinear monotonic relationships | `spearmanr` |
| **Kendall tau** | [-1, 1] | Fraction of concordant vs discordant pairs. More robust than Spearman | `kendalltau` |

### Probe models (what to fit on representations)

| Model | What it tests | sklearn |
|---|---|---|
| `[v1]` **Ridge** (L2 linear) | Is parameter_0 linearly encoded? Standard probe in repr. learning (DINO, MAE, data2vec) | `Ridge` |
| **Lasso** (L1 linear) | Which features matter? Sparse feature selection | `Lasso` |
| **K-NN Regressor** | Is parameter_0 locally smooth in repr. space? (non-parametric) | `KNeighborsRegressor` |
| **MLP (1 hidden layer)** | Is parameter_0 present but nonlinearly encoded? | `MLPRegressor` |
| **SVR** (Support Vector) | Nonlinear probe with kernel trick | `SVR` |

---

## 2. Unlabeled Evaluation (no parameter_0 needed)

These metrics work on ANY dataset, using structure that's "free" from the data.

### 2a. Component-based clustering

**Idea:** Every WAV file has a component ID in its filename (e.g., `comp0`, `comp3`).
Samples from the same component type should be more similar than samples from different components.
This is a "free label" available in all multi-component datasets.

| Metric | Range | Description | What it tests |
|---|---|---|---|
| **Adjusted Rand Index (ARI)** | [-1, 1] | Agreement between K-means clusters and component IDs | Do embeddings group by component? |
| **Normalized Mutual Information (NMI)** | [0, 1] | Shared information between clusters and component IDs | Same, information-theoretic |
| **Silhouette Score** | [-1, 1] | How tight are same-component clusters vs cross-component? | Cluster quality |
| **V-Measure** | [0, 1] | Harmonic mean of homogeneity and completeness | Balanced clustering quality |

### 2b. Neighbor-based (no labels at all)

| Metric | Description | What it tests |
|---|---|---|
| **Same-component K-NN precision** | Among K nearest embedding neighbors, what fraction share the same component ID? | Local structure preservation |
| **Component retrieval mAP** | Mean average precision treating component ID as relevance | Ranking quality |
| **Embedding-input alignment** | Spearman correlation between pairwise distances in input vs embedding space | Does the model preserve relative structure? |
| **Embedding variance ratio** | `var(inter-component distances) / var(intra-component distances)` | Are component clusters well-separated? |

### 2c. Information-theoretic

| Metric | Description |
|---|---|
| **Mutual Information (embeddings, component)** | Bits of component information captured |
| **Vendi Score** | Diversity of the embedding distribution (eigenvalue-based) |
| **Effective Rank** | Intrinsic dimensionality of embedding space |

### 2d. Centered Kernel Alignment (CKA)

| Metric | Description |
|---|---|
| **Linear CKA(input, embedding)** | Structural similarity between input and embedding spaces |
| **RBF CKA(input, embedding)** | Same but with nonlinear kernel |

---

## 3. Labeled Data Analysis: What makes spectrograms share the same parameter_0?

Analysis of `dataset0022` (4,716 rows, 16 components, 168 unique parameter_0 values).
All plots are in `code/plots/labeled_data_analysis/`.

### Finding 1: Same-parameter spectrograms are almost identical in direction

Within a group sharing the same `parameter_0`:
- **Intra-group cosine similarity: 0.996** (very high — nearly parallel vectors)
- Row mean varies slightly (e.g., [0.14, 0.25]) — same shape, different amplitude

BUT **cross-group cosine similarity is also 0.9998** — all groups point in nearly the same direction.
Cosine similarity cannot distinguish groups.

**Concrete example (component 0):**

```
Row  98 (p0 = -1.77):  [0.013, 0.015, 0.016, 0.019, 0.022, ...]
Row  15 (p0 ≈  0.00):  [0.025, 0.026, 0.027, 0.029, 0.032, ...]
Row 405 (p0 = +1.77):  [0.029, 0.026, 0.024, 0.024, 0.024, ...]

DIFFERENT p0 (-1.77 vs +1.77):  cos=0.9864, euclid=1.54
SAME p0     (-1.77 vs -1.77):   cos=0.9951, euclid=0.58
```

Two rows from the SAME group can be more different than rows from OPPOSITE groups.
Spatial variability within a group swamps the parameter_0 signal.

![Same vs Different parameter_0](plots/labeled_data_analysis/01_same_vs_different_parameter0_comp0.png)

### Finding 2: The signal is in magnitude, not direction

| Space | Intra-group euclid dist | Inter-group euclid dist | Ratio |
|---|---|---|---|
| Component 0 | 0.672 | 0.652 | **0.97** (barely different) |

With a single component (245 features), intra-group and inter-group distances are nearly identical.
The parameter_0 signal is buried in spatial variability within each group.

![Intra vs inter-group distances](plots/labeled_data_analysis/06_intra_vs_inter_group_distances.png)

### Finding 3: parameter_0 is encoded across multiple components

Pearson correlation between `parameter_0` and each component's mean value:

| Component | Pearson r | Strength |
|---|---|---|
| comp 30 | **-0.417** | Moderate (strongest!) |
| comp 27 | -0.167 | Weak |
| comp 14 | -0.108 | Very weak |
| comp 20 | -0.108 | Very weak |
| comp 2 | -0.086 | Very weak |
| comp 29 | +0.079 | Very weak |
| comp 0 | +0.010 | None |
| comp 26 | -0.027 | None (constant comp) |

**Only component 30 has meaningful correlation with parameter_0.** The rest are weak or zero.

![Component correlation with parameter_0](plots/labeled_data_analysis/04_component_correlation_with_parameter0.png)

Component 30 shows clear per-feature correlation structure. Component 0 shows almost none:

![Comp 30 vs Comp 0 signal](plots/labeled_data_analysis/02_comp30_vs_comp0_parameter0_signal.png)

### Finding 4: Spectral distance tracks parameter distance — but only across components

Using group centroids, Pearson correlation between |delta parameter_0| and euclidean distance:

| Feature set | Pearson r | Meaning |
|---|---|---|
| Component 0 alone (245-d) | **0.14** | Nearly useless — a single component can't distinguish groups |
| All 16 components (3920-d) | **0.41** | Moderate — combining components helps |
| Component 30 alone (245-d) | **0.64** | Strong — comp 30 is the key discriminative channel |

![Spectral vs parameter distance](plots/labeled_data_analysis/07_spectral_vs_parameter_distance.png)

### Visualizing the full picture

Heatmap of all 16 components for 3 different parameter_0 values:

![All components heatmap](plots/labeled_data_analysis/03_all_components_heatmap_by_parameter0.png)

Difference heatmap (subtracting the p0 approx 0 baseline) reveals which components change with parameter_0:

![Multi-component difference heatmap](plots/labeled_data_analysis/08_multicomp_heatmap_difference.png)

### parameter_0 distribution

168 unique values spanning [-1.77, +1.77], centered at 0. Group sizes range from 1 to 91 rows.

![parameter_0 distribution](plots/labeled_data_analysis/05_parameter0_distribution.png)

### Why this matters for evaluation

1. **A linear probe on single-component WAVs will struggle** — each WAV is only 245 features from ONE component, and most components have near-zero correlation with parameter_0
2. **Component 30 is special** — if the model can learn to emphasize comp 30 features, it could predict parameter_0 well
3. **Multi-component evaluation would be stronger** — concatenating all 16 components for a row gives 3920 features and Pearson=0.41 baseline
4. **The model's job** is to learn a representation where parameter_0 becomes linearly accessible, even from a single component — that would be impressive and would mean the model learned cross-component structure from pre-training

### Implications for unlabeled data

Since we know component IDs from filenames in ALL datasets:
- We can run **component-based clustering** on any dataset (multi_channel, sampled_data, etc.)
- This tests: "did the model learn that different components represent different physical quantities?"
- No labels needed — the component ID IS the label

For `labeled_data` specifically, we can go further:
- Test parameter_0 regression (supervised)
- Test whether samples with similar parameter_0 cluster together in embedding space
- Compare: can the model do what raw features cannot (single comp -> predict parameter_0)?

---

## 4. Planned Metrics Dictionary (key names in EvalResult)

### Labeled regression (prefix: `label_reg_`)

```
label_reg_input_r2          R² of Ridge on raw inputs (baseline)
label_reg_input_mse         MSE on raw inputs
label_reg_input_pearson     Pearson r on raw inputs
label_reg_input_spearman    Spearman ρ on raw inputs
label_reg_emb_r2            R² of Ridge on embeddings
label_reg_emb_mse           MSE on embeddings
label_reg_emb_pearson       Pearson r on embeddings
label_reg_emb_spearman      Spearman ρ on embeddings
label_reg_improvement_r2    emb_r2 - input_r2 (positive = model helps)
label_reg_n_samples         Number of labeled samples used
```

### Unlabeled component clustering (prefix: `comp_cluster_`)

```
comp_cluster_ari            Adjusted Rand Index
comp_cluster_nmi            Normalized Mutual Information
comp_cluster_silhouette     Silhouette Score
comp_cluster_knn_precision  Same-component K-NN precision
comp_cluster_n_components   Number of component types found
comp_cluster_n_samples      Number of samples used
```
