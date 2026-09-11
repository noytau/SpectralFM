# Step 5 — Few-shot / label-efficiency regime

Held-out eval set: 1000 spectra (fixed, disjoint from every training draw). 100 random draws per (n_train, probe, stage).

**Note on `draws`:** TabPFN costs ~2.4s/draw on GPU vs microseconds for the linear/kNN probes, so it runs on a reduced budget (20 draws, transductive only, 1/2/3-comp, n_train ∈ {10,20,50}); the full grid would be ~29 GPU-hours. Its medians therefore carry wider uncertainty than the rest of the panel — compare with that in mind.

**Why this is a separate study:** steps 1-4 measure how much label information a probe recovers given ~4,716 labels. This measures what a deployment with ~20 labels can actually do. They are different questions and need not agree — see plan.md.


## 1-component — median held-out R² at n_train=20 (best probe per stage, transductive PCA)

| stage | best probe | median R² | IQR | frac draws R²>0 | draws | asymptotic R² (step 1) |
|---|---|---|---|---|---|---|
| Raw input | `tabpfn_pca10` | +0.080 | [+0.031, +0.119] | 0.75 | 20 | 0.409 |
| Input (z-scored) | `tabpfn_pca10` | +0.067 | [+0.041, +0.107] | 0.85 | 20 | 0.468 |
| Post-FE (pre-LN) | `dummy` | -0.017 | [-0.048, -0.004] | 0.00 | 100 | 0.538 |
| Post-FE (post-LN) | `dummy` | -0.017 | [-0.048, -0.004] | 0.00 | 100 | 0.567 |
| Post-Projection | `dummy` | -0.017 | [-0.048, -0.004] | 0.00 | 100 | 0.538 |
| Post-Transformer | `dummy` | -0.017 | [-0.048, -0.004] | 0.00 | 100 | 0.628 |

### 1-component — label-efficiency curve (median R², transductive, best probe per cell)

| stage | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| Raw input | -0.050 | -0.023 | +0.080 | -0.014 | +0.154 | +0.128 |
| Input (z-scored) | -0.050 | -0.023 | +0.067 | -0.005 | +0.187 | +0.167 |
| Post-FE (pre-LN) | -0.050 | -0.023 | -0.017 | -0.014 | +0.006 | -0.001 |
| Post-FE (post-LN) | -0.050 | -0.023 | -0.017 | -0.014 | +0.005 | +0.003 |
| Post-Projection | -0.050 | -0.023 | -0.017 | -0.014 | +0.006 | -0.000 |
| Post-Transformer | -0.050 | -0.023 | -0.017 | -0.014 | +0.009 | +0.051 |

## 2-component — median held-out R² at n_train=20 (best probe per stage, transductive PCA)

| stage | best probe | median R² | IQR | frac draws R²>0 | draws | asymptotic R² (step 1) |
|---|---|---|---|---|---|---|
| Raw input | `tabpfn_pca10` | +0.104 | [+0.034, +0.186] | 0.85 | 20 | 0.831 |
| Input (z-scored) | `tabpfn_pca10` | +0.133 | [+0.090, +0.222] | 0.90 | 20 | 0.757 |
| Post-FE (pre-LN) | `tabpfn_pca10` | +0.053 | [+0.032, +0.129] | 0.85 | 20 | 0.753 |
| Post-FE (post-LN) | `tabpfn_pca10` | +0.048 | [+0.033, +0.125] | 0.85 | 20 | 0.762 |
| Post-Projection | `tabpfn_pca10` | +0.060 | [+0.029, +0.111] | 0.85 | 20 | 0.754 |
| Post-Transformer | `tabpfn_pca10` | +0.017 | [-0.031, +0.048] | 0.60 | 20 | 0.757 |

### 2-component — label-efficiency curve (median R², transductive, best probe per cell)

| stage | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| Raw input | -0.034 | -0.023 | +0.104 | +0.043 | +0.232 | +0.184 |
| Input (z-scored) | -0.044 | -0.014 | +0.133 | +0.173 | +0.296 | +0.402 |
| Post-FE (pre-LN) | -0.001 | -0.023 | +0.053 | +0.110 | +0.199 | +0.285 |
| Post-FE (post-LN) | -0.003 | -0.023 | +0.048 | +0.095 | +0.206 | +0.306 |
| Post-Projection | -0.003 | -0.023 | +0.060 | +0.094 | +0.197 | +0.300 |
| Post-Transformer | -0.041 | -0.022 | +0.017 | +0.023 | +0.151 | +0.291 |

## 3-component — median held-out R² at n_train=20 (best probe per stage, transductive PCA)

| stage | best probe | median R² | IQR | frac draws R²>0 | draws | asymptotic R² (step 1) |
|---|---|---|---|---|---|---|
| Raw input | `tabpfn_pca10` | +0.098 | [+0.037, +0.139] | 0.90 | 20 | 0.867 |
| Input (z-scored) | `tabpfn_pca10` | +0.072 | [+0.021, +0.118] | 0.75 | 20 | 0.814 |
| Post-FE (pre-LN) | `tabpfn_pca10` | +0.036 | [+0.004, +0.067] | 0.75 | 20 | 0.791 |
| Post-FE (post-LN) | `tabpfn_pca10` | +0.035 | [-0.007, +0.059] | 0.70 | 20 | 0.789 |
| Post-Projection | `tabpfn_pca10` | +0.027 | [+0.000, +0.059] | 0.75 | 20 | 0.789 |
| Post-Transformer | `tabpfn_pca10` | +0.017 | [-0.038, +0.046] | 0.60 | 20 | 0.793 |

### 3-component — label-efficiency curve (median R², transductive, best probe per cell)

| stage | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| Raw input | +0.001 | -0.023 | +0.098 | +0.082 | +0.222 | +0.244 |
| Input (z-scored) | -0.036 | -0.023 | +0.072 | +0.151 | +0.284 | +0.404 |
| Post-FE (pre-LN) | -0.050 | -0.023 | +0.036 | +0.029 | +0.188 | +0.302 |
| Post-FE (post-LN) | -0.050 | -0.023 | +0.035 | +0.024 | +0.189 | +0.309 |
| Post-Projection | -0.050 | -0.023 | +0.027 | +0.020 | +0.182 | +0.304 |
| Post-Transformer | -0.039 | -0.023 | +0.017 | -0.005 | +0.130 | +0.279 |

## 12-component — median held-out R² at n_train=20 (best probe per stage, transductive PCA)

| stage | best probe | median R² | IQR | frac draws R²>0 | draws | asymptotic R² (step 1) |
|---|---|---|---|---|---|---|
| Raw input | `ridge_strong` | +0.161 | [+0.044, +0.229] | 0.80 | 100 | 0.988 |
| Input (z-scored) | `pca5_ridge` | +0.007 | [-0.044, +0.110] | 0.56 | 100 | 0.980 |
| Post-FE (pre-LN) | `dummy` | -0.017 | [-0.048, -0.004] | 0.00 | 100 | 0.954 |
| Post-FE (post-LN) | `dummy` | -0.017 | [-0.048, -0.004] | 0.00 | 100 | 0.952 |
| Post-Projection | `dummy` | -0.017 | [-0.048, -0.004] | 0.00 | 100 | 0.949 |
| Post-Transformer | `ridge_strong` | -0.011 | [-0.131, +0.033] | 0.34 | 100 | 0.939 |

### 12-component — label-efficiency curve (median R², transductive, best probe per cell)

| stage | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|
| Raw input | -0.042 | +0.008 | +0.161 | +0.286 | +0.404 | +0.509 |
| Input (z-scored) | -0.050 | -0.017 | +0.007 | +0.194 | +0.367 | +0.619 |
| Post-FE (pre-LN) | -0.050 | -0.023 | -0.017 | +0.006 | +0.187 | +0.325 |
| Post-FE (post-LN) | -0.050 | -0.023 | -0.017 | -0.002 | +0.164 | +0.316 |
| Post-Projection | -0.050 | -0.023 | -0.017 | -0.003 | +0.162 | +0.314 |
| Post-Transformer | -0.050 | -0.023 | -0.011 | +0.026 | +0.151 | +0.265 |

## Transductive vs inductive PCA at n_train=20

Transductive = PCA basis fitted on all 4,716 spectra's representations (no labels used; deployment does have unlabeled spectra). Inductive = PCA fitted inside the 20-sample draw only. Both shown because they are different deployment assumptions.

| n-comp | stage | transductive | inductive |
|---|---|---|---|
| 1 | Raw input | +0.080 (`tabpfn_pca10`) | -0.029 (`pca1_ridge`) |
| 1 | Input (z-scored) | +0.067 (`tabpfn_pca10`) | -0.025 (`pca1_ridge`) |
| 1 | Post-FE (pre-LN) | -0.017 (`dummy`) | -0.022 (`pca1_ridge`) |
| 1 | Post-FE (post-LN) | -0.017 (`dummy`) | -0.021 (`pca1_ridge`) |
| 1 | Post-Projection | -0.017 (`dummy`) | -0.021 (`pca1_ridge`) |
| 1 | Post-Transformer | -0.017 (`dummy`) | -0.026 (`pca1_ridge`) |
| 2 | Raw input | +0.104 (`tabpfn_pca10`) | -0.028 (`pca1_ridge`) |
| 2 | Input (z-scored) | +0.133 (`tabpfn_pca10`) | +0.054 (`pca5_ridge`) |
| 2 | Post-FE (pre-LN) | +0.053 (`tabpfn_pca10`) | -0.011 (`pca5_ridge`) |
| 2 | Post-FE (post-LN) | +0.048 (`tabpfn_pca10`) | -0.009 (`pca5_ridge`) |
| 2 | Post-Projection | +0.060 (`tabpfn_pca10`) | -0.007 (`pca5_ridge`) |
| 2 | Post-Transformer | +0.017 (`tabpfn_pca10`) | -0.012 (`pca2_ridge`) |
| 3 | Raw input | +0.098 (`tabpfn_pca10`) | -0.016 (`pca5_ridge`) |
| 3 | Input (z-scored) | +0.072 (`tabpfn_pca10`) | +0.016 (`pca5_ridge`) |
| 3 | Post-FE (pre-LN) | +0.036 (`tabpfn_pca10`) | -0.025 (`pca5_ridge`) |
| 3 | Post-FE (post-LN) | +0.035 (`tabpfn_pca10`) | -0.026 (`pca5_ridge`) |
| 3 | Post-Projection | +0.027 (`tabpfn_pca10`) | -0.022 (`pca5_ridge`) |
| 3 | Post-Transformer | +0.017 (`tabpfn_pca10`) | -0.016 (`pca5_ridge`) |
| 12 | Raw input | +0.161 (`ridge_strong`) | +0.015 (`pca5_ridge`) |
| 12 | Input (z-scored) | +0.007 (`pca5_ridge`) | +0.030 (`pca5_ridge`) |
| 12 | Post-FE (pre-LN) | -0.017 (`dummy`) | -0.028 (`pca1_ridge`) |
| 12 | Post-FE (post-LN) | -0.017 (`dummy`) | -0.027 (`pca1_ridge`) |
| 12 | Post-Projection | -0.017 (`dummy`) | -0.028 (`pca1_ridge`) |
| 12 | Post-Transformer | -0.011 (`ridge_strong`) | -0.022 (`pca5_ridge`) |

## PCA-ceiling diagnostic

Large-n (n≈4,716) RidgeCV R² on the **same PCA-k features** the few-shot probes see. This bounds what *any* probe could extract from those features, so a weak `tabpfn_pca{k}` result can be attributed correctly: if the ceiling here is high but the few-shot number is low, the probe/sample size is the limit; if the ceiling itself is low, PCA discarded the label direction (it maximizes variance, not label-relevance) and no probe on those features can recover it.

| n-comp | stage | k=10 | k=20 | k=50 | full-dim (step 1) |
|---|---|---|---|---|---|
| 1 | Raw input | +0.314 | +0.376 | +0.382 | 0.409 |
| 1 | Input (z-scored) | +0.276 | +0.323 | +0.406 | 0.468 |
| 1 | Post-FE (pre-LN) | +0.055 | +0.220 | +0.350 | 0.538 |
| 1 | Post-FE (post-LN) | +0.053 | +0.222 | +0.351 | 0.567 |
| 1 | Post-Projection | +0.048 | +0.237 | +0.346 | 0.538 |
| 1 | Post-Transformer | +0.072 | +0.241 | +0.333 | 0.628 |
| 2 | Raw input | +0.415 | +0.664 | +0.817 | 0.831 |
| 2 | Input (z-scored) | +0.427 | +0.547 | +0.752 | 0.757 |
| 2 | Post-FE (pre-LN) | +0.295 | +0.402 | +0.465 | 0.753 |
| 2 | Post-FE (post-LN) | +0.301 | +0.409 | +0.469 | 0.762 |
| 2 | Post-Projection | +0.309 | +0.397 | +0.474 | 0.754 |
| 2 | Post-Transformer | +0.238 | +0.384 | +0.472 | 0.757 |
| 3 | Raw input | +0.392 | +0.629 | +0.863 | 0.867 |
| 3 | Input (z-scored) | +0.338 | +0.503 | +0.739 | 0.814 |
| 3 | Post-FE (pre-LN) | +0.253 | +0.412 | +0.471 | 0.791 |
| 3 | Post-FE (post-LN) | +0.254 | +0.416 | +0.470 | 0.789 |
| 3 | Post-Projection | +0.256 | +0.410 | +0.475 | 0.789 |
| 3 | Post-Transformer | +0.210 | +0.312 | +0.474 | 0.793 |
| 12 | Raw input | +0.431 | +0.781 | +0.955 | 0.988 |
| 12 | Input (z-scored) | +0.364 | +0.514 | +0.833 | 0.980 |
| 12 | Post-FE (pre-LN) | +0.237 | +0.380 | +0.470 | 0.954 |
| 12 | Post-FE (post-LN) | +0.241 | +0.381 | +0.471 | 0.952 |
| 12 | Post-Projection | +0.239 | +0.378 | +0.469 | 0.949 |
| 12 | Post-Transformer | +0.274 | +0.344 | +0.424 | 0.939 |

## Scope

This settles which representation is more label-efficient under one honest small-n protocol. It does **not** validate a deployment pipeline: real deployment also has to choose *which* 20 spectra get labeled (stratified/active selection would beat random draws and is the obvious follow-up), plus distribution shift and calibration. Out of scope here.
