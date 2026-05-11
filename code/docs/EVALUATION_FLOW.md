# SpectralFM Evaluation Framework — Complete Reference

> Generated from codebase review, April 2026.
> Covers all evaluation scripts, methods, checkpoint handling, output structure, and past experiments.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Entry Points and Runners](#2-entry-points-and-runners)
3. [Evaluation Methods (In-Depth)](#3-evaluation-methods-in-depth)
4. [FE Decoder Reconstruction Evaluation](#4-fe-decoder-reconstruction-evaluation)
5. [Label Regression Evaluation](#5-label-regression-evaluation)
6. [Checkpoint Discovery and Selection](#6-checkpoint-discovery-and-selection)
7. [Model Loading Pipeline](#7-model-loading-pipeline)
8. [Output Structure and Reports](#8-output-structure-and-reports)
9. [Plot Catalog](#9-plot-catalog)
10. [Standalone Visualization Scripts](#10-standalone-visualization-scripts)
11. [Shell Orchestrator](#11-shell-orchestrator)
12. [WandB, Logging, and External Integrations](#12-wandb-logging-and-external-integrations)
13. [Metric Naming Convention](#13-metric-naming-convention)
14. [Past Experiments in `eval_results/`](#14-past-experiments-in-eval_results)
15. [Utility Modules Reference](#15-utility-modules-reference)
16. [Recipes and Quick-Start](#16-recipes-and-quick-start)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                    │
│                                                                         │
│  evaluation_runner.py     Main workhorse: multi-checkpoint comparison   │
│  run_full_evaluation.py   Orchestrator: full/scaling/transfer/dynamics  │
│  eval_fe_decoder.py       FE decoder reconstruction evaluation          │
│  eval_label_regression.py Standalone label regression probe             │
│  compare_checkpoints.py   Quick multi-checkpoint comparisons            │
│  run_comprehensive_eval.sh  Bash: runs FE-decoder + runner + label-reg │
│  label_reg_evaluation.py  Label regression plot suite from caches       │
└─────────────┬───────────────────────────────────┬───────────────────────┘
              │                                   │
              ▼                                   ▼
┌─────────────────────────┐         ┌─────────────────────────────┐
│     CORE MODULES        │         │    VISUALIZATION / METRICS  │
│                         │         │                             │
│  model_loader.py        │         │  eval_plots.py              │
│  eval_utils.py          │         │  eval_metrics.py            │
│                         │         │  plot_structured_sim...py   │
│                         │         │  plot_model_comparison.py   │
│                         │         │  plot_input_vs_features..py │
│                         │         │  plot_masking_examples.py   │
└─────────────────────────┘         └─────────────────────────────┘
```

### Module Dependency Map

| Module | Imports from project |
|--------|---------------------|
| `evaluation_runner.py` | `model_loader`, `eval_utils`, `eval_plots`, `eval_metrics`, `compare_checkpoints` |
| `eval_fe_decoder.py` | `model_loader`, `eval_utils`, `eval_plots`, `eval_metrics` |
| `eval_label_regression.py` | `eval_fe_decoder` (checkpoint discovery, embedding extraction), `eval_metrics` |
| `run_full_evaluation.py` | `evaluation_runner`, `eval_plots`, `label_reg_evaluation` |
| `compare_checkpoints.py` | `evaluation_runner` |
| `eval_utils.py` | `model_loader` (lazy, for FE extraction) |
| `eval_plots.py` | None (standalone matplotlib/seaborn) |
| `eval_metrics.py` | None (standalone numpy/scipy/sklearn) |
| `label_reg_evaluation.py` | None (standalone, reads cached `.npz` files) |

---

## 2. Entry Points and Runners

### 2.1 `evaluation_runner.py` — Main Workhorse

The primary evaluation script. Handles single or multi-checkpoint evaluation, generates per-run and cross-run comparison plots, outlier analysis, and comprehensive reports.

**Key classes:**

| Class | Purpose |
|-------|---------|
| `CheckpointDiscovery` | Scans directories for `.pt` checkpoint files |
| `CheckpointInfo` | Dataclass: path, run_dir, date, time, checkpoint_type, epoch, config |
| `EvalResult` | Dataclass: checkpoint_path, run_name, timestamp, metrics dict, config_summary |
| `EvaluationRunner` | Core runner: loads models, runs `_eval_*` methods, generates plots and reports |
| `ReportGenerator` | Produces CSV, JSON, TXT summary, and Markdown comparison reports |

**Main flow:**
1. Parse CLI arguments
2. Discover checkpoints via `CheckpointDiscovery` or explicit `--checkpoint` path
3. Create timestamped output directory: `<output_dir>/<YYYYMMDD_HHMMSS>/`
4. Instantiate `EvaluationRunner(output_dir, data_dir)`
5. Call `runner.evaluate_all(checkpoints, eval_methods, ...)` which:
   - Pre-loads samples once (100 samples, seed 42) for fair cross-checkpoint comparison
   - Iterates through each checkpoint calling `evaluate_checkpoint()`
   - Each `evaluate_checkpoint()` loads the model, extracts embeddings, runs requested `_eval_*` methods
   - Generates per-run plots in `plots/<run_name>/`
   - Caches numpy arrays in `data/`
6. Generate cross-run comparison plots (if `len(results) > 1`)
7. Run outlier analysis (if `--analyze_outliers`)
8. Generate reports via `ReportGenerator` + `generate_evaluation_summary()` + `generate_comparison_report_with_images()`

**Can be run together:** All `--eval_methods` run in a single invocation. Cross-run plots only appear with 2+ checkpoints.

**CLI reference:**

```
python evaluation_runner.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint_dir` | `/mnt5/noy/fairseq/outputs` | Root directory to scan for checkpoints |
| `--checkpoint` | — | Path to a single `.pt` file (skips directory scan) |
| `--extra_checkpoints` | — | Additional `.pt` files to include from any directory |
| `--output_dir` | `code/eval_results` | Base output (timestamped subdir created) |
| `--eval_data_dir` | — | Evaluation dataset override (highest priority) |
| `--data_dir` | `/mnt5/noy/fairseq/data/single_channel_1m/` | Default data dir (lower priority than `--eval_data_dir`) |
| `--eval_methods` | `embedding_similarity` | Space-separated method names |
| `--all_methods` | off | Run all: embedding_similarity, noise_robustness, stack_similarity, signal_completion, validation_loss |
| `--best_only` | off | Only `checkpoint_best.pt` files |
| `--latest_only` | off | Most recent checkpoint only |
| `--run_names` | — | Whitelist of run names |
| `--plot_matrices` | off | N×N similarity matrix heatmaps |
| `--custom_dataset_path` | — | Second dataset for 4-way comparison |
| `--include_random_weights` | off | Add random-init baseline (5-way) |
| `--debug` | off | Save spectrograms to `debug_plots/` |
| `--analyze_outliers` | off | Find lowest-similarity samples |
| `--no_analyze_inliers` | off | Skip highest-similarity analysis |
| `--outlier_run_name` | all | Restrict outlier analysis to one run |
| `--outlier_similarity_type` | `both` | `embedding`, `input`, or `both` |
| `--nova_data_dir` | env or `/mnt5/noy/fairseq/data` | Parent of nova datasets for structured_similarity |
| `--structured_similarity_seed` | 42 | RNG seed for structured panel |
| `--structured_similarity_entries_json` | auto-detected | Path to 100-entry JSON file |
| `--structured_similarity_ignore_entries_json` | off | Force RNG-built panel instead of JSON |
| `--mask_memory_path` | — | Fixed mask file for evaluation |
| `--report_name` | auto | Custom report filename prefix |

---

### 2.2 `run_full_evaluation.py` — Multi-Mode Orchestrator

Higher-level orchestrator with six modes. Wraps `EvaluationRunner` for different evaluation scenarios.

```
python run_full_evaluation.py --mode MODE [OPTIONS]
```

| Mode | Description | Key inputs |
|------|-------------|------------|
| `full` | All methods on a single checkpoint | `--checkpoint` |
| `scaling_laws` | Sweep across checkpoints/dataset sizes | `--checkpoint_dir` |
| `transfer` | Cross-dataset transfer evaluation | `--checkpoint` |
| `light` | No fairseq dependency (HuggingFace model) | `--model_path` |
| `training_dynamics` | Metric trajectories over intermediate checkpoints | `--checkpoint_dir` |
| `ablation` | Generate ablation study configs (JSON only) | — |

**Outputs per mode:**

| Mode | Output files |
|------|-------------|
| `full` | `full_evaluation_results.json`, label_reg plots (if applicable) |
| `scaling_laws` | `scaling_laws.png`, `scaling_law_results.json` |
| `transfer` | `transfer_results.json` |
| `light` | `light_component_clustering.png`, `light_repr_geometry.png`, `light_evaluation_results.json` |
| `training_dynamics` | `training_dynamics_results.json` |
| `ablation` | `ablation_configs.json` |

**Individual runner** — does not require `evaluation_runner.py` for `ablation` mode. All other modes instantiate `EvaluationRunner` internally.

---

### 2.3 `eval_fe_decoder.py` — FE Decoder Reconstruction

**Individual runner.** Standalone script that trains small decoders on frozen CNN feature extractor outputs to reconstruct input spectrograms. See [Section 4](#4-fe-decoder-reconstruction-evaluation) for full details.

---

### 2.4 `eval_label_regression.py` — Standalone Label Regression Probe

**Individual runner.** Trains ridge regression on transformer embeddings to predict `parameter_0`. See [Section 5](#5-label-regression-evaluation) for full details.

---

### 2.5 `compare_checkpoints.py` — Quick Multi-Checkpoint Comparison

Compares multiple explicit checkpoint paths. Instantiates `EvaluationRunner` and calls `evaluate_all()`, then adds extra comparison plots (validation loss bars, embedding metrics CSV).

```
python compare_checkpoints.py --checkpoints path1.pt path2.pt [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoints` | required | One or more `.pt` paths |
| `--output_dir` | `eval_results/comparison` | Output directory |
| `--data_dir` | `/mnt5/noy/fairseq/data/single_channel_1m/` | Default data |
| `--custom_dataset_path` | — | Second dataset |
| `--eval_methods` | `validation_loss embedding_similarity` | Methods to run |
| `--debug` | off | Debug mode |

**Extra outputs beyond runner:**
- `validation_loss_comparison.png`, `.csv`, `.txt`
- `embedding_metrics_comparison.csv`
- `comparison_summary.txt`

---

### 2.6 `label_reg_evaluation.py` — Label Regression Plot Suite (From Caches)

**Individual runner.** Generates five diagnostic scatter-plot figures from **pre-computed `.npz` embedding caches**. Does not re-extract embeddings from checkpoints.

```
python label_reg_evaluation.py [--output_dir DIR]
```

Uses hardcoded checkpoint references and cache paths (see [Section 5.2](#52-label_reg_evaluationpy-cache-based-plot-suite)).

---

## 3. Evaluation Methods (In-Depth)

Pass one or more to `--eval_methods` in `evaluation_runner.py`. Use `--all_methods` for all five core methods.

### 3.1 `embedding_similarity`

**Method:** `_eval_embedding_similarity`
**What it measures:** Pairwise cosine similarity of transformer embeddings; Pearson/Spearman correlation with input-space similarity; variance ratio; mean similarity.
**Always runs first** when included — also extracts and caches CNN FE outputs for cross-run plots.

**Metrics produced:**

| Key | Description |
|-----|-------------|
| `valid_pearson_corr` | Pearson r between input and embedding pairwise cosine similarities |
| `valid_spearman_corr` | Spearman ρ (rank correlation) |
| `valid_sim_variance_ratio` | σ(embedding sims) / σ(input sims) — collapse detector |
| `valid_emb_mean_sim` | Mean pairwise embedding cosine similarity (lower = less collapsed) |
| `valid_emb_std_sim` | Std of pairwise embedding similarities |
| `input_mean_sim` | Mean pairwise input cosine similarity |
| `best_loss` | Best training loss from checkpoint |

**Per-checkpoint plots:** `embedding_similarity_comparison_valid.png` (3- or 4-way panel: input / embedding / frozen / random)
**Cross-run plots (2+ checkpoints):** embedding + FE grid, histogram comparison, matrix comparison, metrics bar chart

---

### 3.2 `structured_similarity`

**Method:** `_eval_structured_similarity`
**What it measures:** Block-structured cosine similarity across a fixed 100-sample panel spanning four nova datasets. The panel is designed so that samples within each 10-sample group come from the same physical component/stack.

**Panel construction (100 samples):**
```
nova_data_dir/
├── single_channel_all/   → 30 samples (3 stacks × 10)
├── multi_channel/         → 30 samples (3 components × 10)
├── sampled_data/          → 20 samples (2 components × 10)
└── labeled_data/          → 20 samples (2 components × 10)
```

**Collapse diagnosis:** A healthy model shows 10 bright 10×10 blocks on the diagonal; a collapsed model shows a uniformly bright matrix.

**Per-checkpoint plots:** `structured_similarity.png` (3-row: Input / FE / Embedding)
**Cross-run plots:** `all_models_structured_similarity_with_fe.png`, per-group `divein_*` plots, per-dataset plots

**Requires:** `--nova_data_dir` (defaults to env or `/mnt5/noy/fairseq/data`)

---

### 3.3 `label_regression`

**Method:** `_eval_label_regression` (inside `evaluation_runner.py`)
**What it measures:** Linear probe (L-BFGS / Ridge) predicting `parameter_0` from transformer embeddings vs raw input features. Train sizes: 100, 500, 1000, 2000.

**Metrics produced (per train size `n`):**

| Key | Description |
|-----|-------------|
| `label_reg_emb_r2_{n}` | R² of embedding probe |
| `label_reg_input_r2_{n}` | R² of raw-input baseline |
| `label_reg_improvement_{n}` | R² gain: embedding − baseline |
| `label_reg_emb_pearson_{n}` | Pearson r of embedding probe |
| `label_reg_best_r2` | Best R² across all train sizes |

**Cross-run plot:** `label_regression_comparison_train_size.png` (grid: rows = train sizes, columns = models)
**Requires:** `--labeled_data_dir` (defaults to `nova_data/labeled_data`)

---

### 3.4 `noise_robustness`

**Method:** `_eval_noise_robustness`
**What it measures:** Stability of embeddings under perturbations: Gaussian noise (std/mean variants), gain changes (low/high).

**Metrics (per noise type `nt`):**

| Key | Description |
|-----|-------------|
| `noise_{nt}_emb_sim_mean` | Mean cosine sim between clean and noisy embeddings |
| `noise_{nt}_emb_sim_std` | Std of cosine similarities |

**Cross-run plot:** `noise_robustness_comparison.png` (bar chart)

---

### 3.5 `signal_completion`

**Method:** `_eval_signal_completion`
**What it measures:** Can the model predict masked-out portions of the signal? Tests causal (50%, 25%) and random (30%, 50%) masking strategies.

**Metrics (per strategy `s`):**

| Key | Description |
|-----|-------------|
| `completion_{s}_cos_sim_mean` | Mean cosine similarity at masked positions |
| `completion_{s}_mse_mean` | Mean MSE at masked positions |

**Per-checkpoint plots:** Histogram of cos-sim, MSE distribution, scatter
**Cross-run plot:** `signal_completion_comparison.png`

---

### 3.6 `stack_similarity`

**Method:** `_eval_stack_similarity`
**What it measures:** Whether top-K neighbors in embedding space match stack membership (from filenames).

**Metrics:**

| Key | Description |
|-----|-------------|
| `stack_match_score_mean` | Mean stack-match score |
| `stack_match_improvement_pct` | Improvement over input-space baseline |

**Per-checkpoint plots:** Match score histogram, embedding vs input similarity comparison
**Cross-run plot:** `stack_similarity_comparison.png`

---

### 3.7 `validation_loss`

**Method:** `_eval_validation_loss`
**What it measures:** Re-computes the training loss on the evaluation split. Sanity check metric.

**Metrics:**

| Key | Description |
|-----|-------------|
| `eval_loss` | Validation loss value |

---

### 3.8 Additional Methods (in `eval_metrics.py`, available via `run_full_evaluation.py`)

These are implemented in `eval_metrics.py` and accessible through `run_full_evaluation.py` or programmatic use:

| Method flag | Metric prefix | What it computes |
|-------------|--------------|------------------|
| `component_clustering` | `comp_cluster_*` | KMeans clustering by component ID (ARI, NMI, Silhouette, V-Measure, KNN Precision) |
| `knn_retrieval` | `knn_*` / `comp_knn_*` | KNN retrieval and neighbor overlap |
| `repr_geometry` | `repr_*` | SVD spectrum, CKA (linear/RBF), effective rank, uniformity, alignment, Vendi score |
| `downstream_probing` | `downstream_*` | Ridge/MLP probes (R², Pearson r) |
| `fewshot` | `downstream_fewshot_*` | Few-shot learning curves at various label fractions |
| `parameter_verification` | `spectral_param_verify_*` | ROC + cosine score histograms for same/different parameter pairs |
| `component_detection` | `spectral_comp_detect_*` | Component detection metrics |
| `baselines` | — | Signal processing features (FFT, wavelet, moments, peaks, centroid) |
| `efficiency` | `efficiency_*` | Computational efficiency metrics |
| `attention` | — | Attention map visualization (per-layer heatmaps) |
| `failure_analysis` | — | Worst-percentile error analysis, per-component MAE |
| `augmentation_invariance` | — | Augmentation invariance testing |

---

## 4. FE Decoder Reconstruction Evaluation

### 4.1 `eval_fe_decoder.py`

**Individual runner** (not part of `evaluation_runner.py`'s `--eval_methods`). Trains small decoders on frozen CNN feature extractor (FE) outputs to reconstruct 245-dim input spectrograms.

**Architecture:** FE → mean-pool → decoder → reconstructed spectrogram (245-dim)

**Decoder variants** (configurable via `--decoder_variants`):

| Spec | Architecture | Parameters |
|------|-------------|------------|
| `0` | Linear: 512 → 245 | ~126K |
| `512` | MLP: 512 → 512 → 245 | ~388K |
| `512:256` | MLP: 512 → 512 → 256 → 245 | ~457K |

**CLI:**

```
python eval_fe_decoder.py --checkpoint_dir DIR [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` / `--checkpoint_dir` | required (mutually exclusive) | Checkpoint source |
| `--best_only` / `--latest_only` / `--run_names` | — | Checkpoint filtering |
| `--eval_data_dir` | — | WAV directory |
| `--inputs_npy` | — | Pre-extracted `[N,D]` spectrograms |
| `--nova_data_dir` | — | If neither data arg set, loads structured 100-sample panel |
| `--epochs` | 200 | Decoder training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--decoder_variants` | `0 512 512:256` | Decoder architectures |
| `--include_embedding_decoder` | off | Also train on transformer embeddings |
| `--output_dir` | `eval_results/fe_decoder` | Output directory |

**Metrics produced (per variant):**

| Key | Description |
|-----|-------------|
| `fe_dec_cosine_mean` / `fe_dec_cosine_std` | Cosine similarity between original and reconstructed |
| `fe_dec_mse` / `fe_dec_mae` | Mean squared / absolute error |
| `fe_dec_r2` | R² score of reconstruction |
| `fe_dec_train_loss` | Final training loss |

**Output structure:**

```
<output_dir>/
├── structured_subset_meta.json          # (if using structured panel)
├── comparison_bar_chart.png             # Multi-checkpoint metrics bar chart
├── decoder_variants_cross_checkpoint.png # Variant × checkpoint grouped bars
├── fe_vs_transformer_bar_chart.png      # (if --include_embedding_decoder)
├── fe_vs_transformer_architecture_multi_model.png
├── diagnostics/
│   ├── D1_per_bin_error_heatmap.png     # Per-frequency-bin RMSE across models
│   ├── D2_pca_component_r2.png          # R² per PCA component
│   ├── D3_residual_scatter.png          # Residual vs original scatter
│   └── D4_fe_vs_transformer_r2.png      # Per-sample FE vs transformer R²
├── <run_label>/
│   ├── fe_outputs_train.npy / fe_outputs_eval.npy
│   ├── inputs_train.npy / inputs_eval.npy
│   ├── Linear/
│   │   ├── metrics.json
│   │   ├── reconstructed_eval.npy
│   │   ├── per_cosine.npy / per_mse.npy
│   │   └── metrics_transformer.json     # (if --include_embedding_decoder)
│   ├── MLP-512/ ...
│   ├── MLP-512-256/ ...
│   └── plots/
│       ├── reconstruction_samples_Linear.png
│       ├── reconstruction_samples_MLP-512.png
│       ├── score_distributions_Linear.png
│       ├── decoder_variants_comparison.png
│       ├── reconstruction_triple_Linear.png  # (if embedding decoder)
│       ├── reconstruction_all_variants.png
│       └── fe_vs_transformer_by_architecture.png
```

---

### 4.2 Embedding Extraction Functions

Both `eval_fe_decoder.py` and `eval_label_regression.py` use:

| Function | Source | Output shape |
|----------|--------|-------------|
| `extract_fe_outputs_from_inputs(inputs, ckpt_path)` | `eval_fe_decoder.py` | `[N, 512]` — mean-pooled CNN FE outputs |
| `extract_embeddings_from_inputs(inputs, ckpt_path)` | `eval_fe_decoder.py` | `[N, 768]` — mean-pooled transformer embeddings |
| `extract_fe_outputs_from_fairseq_checkpoint(ckpt, samples)` | `eval_utils.py` | `[N, 512]` — FE → LN → proj → mean pool |

---

## 5. Label Regression Evaluation

### 5.1 `eval_label_regression.py` — Standalone Probe

**Individual runner.** Trains ridge regression probes on embeddings and raw inputs to predict `parameter_0`.

```
python eval_label_regression.py --checkpoint_dir DIR [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` / `--checkpoint_dir` | required | Checkpoint source |
| `--labeled_data_dir` | `nova_data/labeled_data` | Directory with `labels.tsv` and WAV files |
| `--max_samples` | 2000 | Max labeled samples |
| `--output_dir` | cwd | Output root |

**Metrics:**

| Key | Description |
|-----|-------------|
| `label_reg_input_*` | Ridge probe on raw spectrograms |
| `label_reg_emb_*` | Ridge probe on transformer embeddings |
| `label_reg_improvement_r2` | R² gain (embedding − input) |

**Outputs:**
- `label_reg_results.json` — aggregated results for all checkpoints
- `<run_label>/label_regression.json` — per-checkpoint metrics
- `plots/label_regression_comparison.png` — two-panel: R² bars + ΔR² bars

---

### 5.2 `label_reg_evaluation.py` — Cache-Based Plot Suite

Generates five diagnostic figures from **pre-computed `.npz` embedding caches**. Does not load checkpoints at runtime.

```
python label_reg_evaluation.py [--output_dir DIR]
```

**Hardcoded configuration:**

```python
CHECKPOINTS = [
    ("2026-01-07_21-50-07",       "Jan-07\nsingle-ch 9k",    1),
    ("2026-02-25_13-46-46",       "Feb-25\nsingle-ch 18.5k", 1),
    ("2026-03-03_17-45-36-multi", "Mar-03\nmulti-ch 17.4k",  0),
]
```

**Cache locations:**
- Input cache: `/mnt5/noy/fairseq/data/single_channel_1m/label_reg_emb_cache_2026-01-07_21-50-07.npz`
- Layer caches: `/mnt5/noy/fairseq/data/single_channel_1m/tasks/<run>/task1_layer_sweep/layer_cache/layer_<L>.npz`
- Labels: `fairseq/data/nova_data/labeled_data/labels.tsv`

**Output plots (under `plots/label_reg_plots/`):**

| File | Description |
|------|-------------|
| `comparison_train_size.png` | Grid: rows = train sizes (100/1k/2k), columns = raw input + 3 models |
| `comparison_linear_vs_mlp.png` | Linear vs MLP probe comparison |
| `comparison_n_components.png` | 1 vs 2 vs 3 spectral components |
| `param0_distributions.png` | `parameter_0` distribution + per-model KDE overlays |
| `merged_comp_comparison_1000.png` | 490-dim vs 1536-dim embeddings scatter |

**Integration:** Called by `run_full_evaluation.py --mode full` when `label_regression` is in methods, and by `run_comprehensive_eval.sh`.

---

## 6. Checkpoint Discovery and Selection

### 6.1 `CheckpointDiscovery` (in `evaluation_runner.py`)

Scans `--checkpoint_dir` recursively for checkpoint files.

**Three directory layouts supported:**

| Layout | Pattern | Example |
|--------|---------|---------|
| **Standard fairseq** | `<date>/<time>/checkpoints/*.pt` | `2026-01-07/21-50-07/checkpoints/checkpoint_best.pt` |
| **Flat named (RunAI copy)** | `<date_time>/checkpoint_best.pt` | `2026-01-07_21-50-07/checkpoint_best.pt` |
| **Flat .pt files** | `*.pt` directly under base_dir | `2026-04-14_07-42-33_recon-fe1.0_recon-tr0.0.pt` |

**Discovery methods:**

| Method | Returns |
|--------|---------|
| `find_all_checkpoints()` | All `checkpoint_best.pt`, `checkpoint_last.pt`, and numbered checkpoints |
| `find_best_checkpoints()` | Only `checkpoint_best.pt` files |
| `find_latest_checkpoint()` | Most recent best checkpoint (sorted by date/time) |

**Run name inference:**
- Standard layout: `{date}_{time}` from directory names
- Flat named: date prefix extracted from `.pt` filename stem
- Unknown: `"unknown"` date, stem as time/name

**Config loading:** Attempts `.hydra/config.yaml` first, then `hydra_train.log` fallback.

---

### 6.2 `discover_checkpoints()` (in `eval_fe_decoder.py`)

Alternative discovery function used by `eval_fe_decoder.py` and `eval_label_regression.py`.

**Two layouts:**

| Layout | Pattern |
|--------|---------|
| **Layout A (flat)** | `*.pt` files directly under dir |
| **Layout B (subdirs)** | `<run_name>/checkpoint_best.pt` or `checkpoint_last.pt` or first `*.pt` |

**Filtering options:** `--best_only`, `--latest_only`, `--run_names`

**WandB run name resolution:** `_wandb_run_name_from_checkpoint()` reads `cfg.common.wandb_run_name` from checkpoint metadata if present, using it as the output directory label.

---

### 6.3 Checkpoint Selection Flags

| Flag | Works in | Behavior |
|------|----------|----------|
| `--checkpoint PATH` | `evaluation_runner`, `eval_fe_decoder`, `eval_label_regression` | Single explicit `.pt` path |
| `--checkpoint_dir DIR` | All scripts | Auto-discover from directory |
| `--extra_checkpoints` | `evaluation_runner` | Additional `.pt` files from any location |
| `--best_only` | All scripts | Only `checkpoint_best.pt` |
| `--latest_only` | All scripts | Most recent checkpoint |
| `--run_names NAME [...]` | All scripts | Whitelist by run name |

---

### 6.4 Checkpoint Paths Reference

| Location | Path |
|----------|------|
| Collapse ablation outputs | `/mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse/` |
| RunAI checkpoints (best) | `/mnt5/noy/SpectralFM/checkpoints/runai/<run-name>/checkpoint_best.pt` |
| Recon loss experiments | `/mnt5/noy/SpectralFM/checkpoints/runai/recon_loss_experiment_*/` |
| RunAI grouped runs | `/mnt5/noy/SpectralFM/checkpoints/runai/2026-03-10-compare-single-to-multi/` |
| Base LibriSpeech pretrained | `/mnt5/noy/SpectralFM/fairseq/base_libri.pt` |

---

## 7. Model Loading Pipeline

### 7.1 `load_fairseq_checkpoint()` (in `model_loader.py`)

The primary model loading function used by all evaluation scripts.

**Flow:**
1. Resolve device (CUDA if available)
2. Pre-read checkpoint with `checkpoint_utils.load_checkpoint_to_cpu()` to extract `state["cfg"]`
3. **Path remapping:** If config paths start with `/storage/noy` (RunAI), remap to `/mnt5/noy` when local file exists
4. If base `model_path` doesn't exist locally, set `skip_pretrained_weights = True`
5. Call `checkpoint_utils.load_model_ensemble_and_task([checkpoint_path], arg_overrides, strict=False)`
6. Patch `model_cfg` with defaults for missing keys
7. Build `checkpoint_info` dict: `num_updates`, `epoch`, `best_loss`, full `cfg`
8. Move model to device

**Returns:** `(model, model_cfg, checkpoint_info)`

### 7.2 Feature Extraction Paths

| Path | Source module | Input → Output |
|------|--------------|----------------|
| Transformer embeddings | `evaluation_runner.py` | `model(source, features_only=True)` → `out["x"]` → mean pool → `[N, 768]` |
| CNN FE outputs | `eval_utils.py` | `model.feature_extractor(source)` → LN → `post_extract_proj` → mean pool → `[N, 512]` |
| FE for decoder | `eval_fe_decoder.py` | Same as above, batched with `--fe_batch_size` |
| Full transformer for decoder | `eval_fe_decoder.py` | `model(source, features_only=True)` → mean pool → `[N, 768]` |

---

## 8. Output Structure and Reports

### 8.1 Standard `evaluation_runner.py` Output

```
<output_dir>/<YYYYMMDD_HHMMSS>/
├── plots/
│   ├── embedding_similarity_comparison_valid.png
│   ├── embedding_similarity_histogram_comparison_valid.png
│   ├── embedding_similarity_matrix_comparison_valid.png
│   ├── embedding_metrics_bar_comparison_valid.png
│   ├── all_models_structured_similarity_with_fe.png
│   ├── per_group_<group>_structured_similarity_with_fe.png      (×10 groups)
│   ├── per_dataset_<dataset>_structured_similarity_with_fe.png  (×4 datasets)
│   ├── divein_<group>_structured_similarity.png                 (×10 groups)
│   ├── label_regression_comparison_train_size.png
│   ├── noise_robustness_comparison.png
│   ├── stack_similarity_comparison.png
│   ├── signal_completion_comparison.png
│   ├── <run_name>/
│   │   ├── embedding_similarity_comparison_valid.png
│   │   ├── similarity_matrices.png
│   │   ├── structured_similarity.png
│   │   ├── label_regression_scatter.png
│   │   ├── outlier_analysis_embedding.png   (if --analyze_outliers)
│   │   └── outlier_analysis_input.png
│   └── ...
├── data/
│   ├── embeddings_<run>_valid.npy
│   ├── fe_outputs_<run>_valid.npy
│   ├── inputs_<run>_valid.npy
│   ├── embeddings_<run>_structured_similarity.npy
│   ├── fe_outputs_<run>_structured_similarity.npy
│   ├── inputs_<run>_structured_similarity.npy
│   ├── embeddings_<run>_label_reg.npy
│   ├── inputs_<run>_label_reg.npy
│   └── embedding_similarity_scores_<run>_valid.npy
├── debug_plots/              (if --debug)
├── spectrograms/
├── eval_report_<timestamp>.json
├── eval_report_<timestamp>.csv
├── eval_report_<timestamp>_summary.txt
└── eval_report_<timestamp>_comparison.md
```

### 8.2 Report Formats

| File | Format | Contents |
|------|--------|----------|
| `eval_report_*.json` | JSON array of `EvalResult` dicts | Full metrics per checkpoint, config summaries |
| `eval_report_*.csv` | CSV | Columns: `run_name`, `checkpoint_path`, `eval_timestamp`, `metric_*`, `config_*` |
| `eval_report_*_summary.txt` | Plain text | "SPECTRAL FM EVALUATION SUMMARY" — per-run metric summaries, methods run, best loss |
| `eval_report_*_comparison.md` | Markdown | Comparison table with embedded image references |

### 8.3 Data Caching

Numpy arrays are cached in `data/` for reuse:
- `embeddings_<run>_valid.npy` — transformer embeddings `[N, 768]`
- `fe_outputs_<run>_valid.npy` — CNN FE outputs `[N, 512]`
- `inputs_<run>_valid.npy` — raw spectrograms `[N, 245]`
- Structured similarity and label regression variants follow the same pattern

These caches enable post-hoc re-plotting without re-running inference (used by `plot_structured_similarity_with_fe.py`).

---

## 9. Plot Catalog

### 9.1 Cross-Run Comparison Plots (require 2+ checkpoints)

| Plot | Generated by | Trigger |
|------|-------------|---------|
| Embedding + FE 2-row grid | `_plot_embedding_similarity_comparison` | `embedding_similarity` |
| Histogram comparison | `plot_embedding_similarity_histogram_comparison` | `embedding_similarity` |
| Matrix heatmap comparison | `plot_embedding_similarity_matrix_comparison` | `embedding_similarity` + `--plot_matrices` |
| Metrics bar chart (Pearson/Spearman/VR/mean) | In comparison flow | `embedding_similarity` |
| Structured similarity all-models grid | `plot_structured_similarity_all_models` | `structured_similarity` |
| Per-group dive-in | `_write_structured_similarity_multi_plot` | `structured_similarity` |
| Label regression grid | Comparison flow | `label_regression` |
| Noise robustness bars | `plot_noise_robustness_comparison` | `noise_robustness` |
| Stack similarity bars | `plot_stack_similarity_comparison` | `stack_similarity` |
| Signal completion bars | `plot_signal_completion_comparison` | `signal_completion` |

### 9.2 Per-Checkpoint Plots

| Plot | Method | Description |
|------|--------|-------------|
| 3/4/5-way similarity panel | `embedding_similarity` | Input / Embedding / Frozen / Random heatmaps |
| Similarity matrices | `embedding_similarity` | N×N input vs embedding |
| Structured similarity 3-row | `structured_similarity` | Input / FE / Embedding |
| Label regression scatter | `label_regression` | True vs predicted per train size |
| Match score histogram | `stack_similarity` | Stack membership matching |
| Signal completion histogram | `signal_completion` | Cos-sim distribution at masked positions |
| Noise robustness spectrogram | `noise_robustness` | Clean vs noisy visualization |

### 9.3 FE Decoder Plots

| Plot | Scope | Description |
|------|-------|-------------|
| `reconstruction_samples_*.png` | Per variant per run | Grid: original vs reconstructed lines |
| `score_distributions_*.png` | Per variant per run | Cosine and MSE histograms |
| `decoder_variants_comparison.png` | Per run | 3-panel bars across variants |
| `comparison_bar_chart.png` | Cross-checkpoint | Metrics per checkpoint |
| `decoder_variants_cross_checkpoint.png` | Cross-checkpoint | Variant × checkpoint grouped bars |
| `reconstruction_triple_*.png` | Per variant (if embedding decoder) | Original / FE recon / Transformer recon |
| `D1_per_bin_error_heatmap.png` | Diagnostic | Frequency-bin RMSE per model |
| `D2_pca_component_r2.png` | Diagnostic | R² per PCA component |
| `D3_residual_scatter.png` | Diagnostic | Residual vs original |
| `D4_fe_vs_transformer_r2.png` | Diagnostic | Per-sample FE vs transformer R² |

### 9.4 `eval_plots.py` — Full Function Catalog

| Function | Plot type |
|----------|-----------|
| `plot_component_clustering` | PCA 2D scatter + clustering metric bars |
| `plot_repr_geometry` | SVD spectrum + CKA + uniformity/alignment bars |
| `plot_probing_comparison` | R² and Pearson r bars per probe type |
| `plot_fewshot_curve` | Few-shot learning curve |
| `plot_parameter_verification` | ROC + cosine score histograms |
| `plot_attention_maps` | Layer attention heatmaps |
| `plot_failure_analysis` | Error histogram, Q-Q, per-component bars |
| `plot_scaling_laws` | Log-x metric vs dataset size lines |
| `plot_metric_intercorrelation` | Spearman correlation heatmap |
| `plot_training_dynamics` | Multi-panel time series with trends |
| `plot_mask_sweep` | Cos-sim vs mask probability |
| `plot_transfer_matrix` | Transfer pair bar chart |
| `plot_ablation_study` | Multi-panel ablation variable lines/bars |
| `plot_per_bin_completion_error` | Twin bar charts (cos-sim / MSE per bin) |
| `plot_fe_decoder_reconstruction_samples` | Original vs reconstructed grid |
| `plot_fe_decoder_score_distribution` | Per-sample cosine/MSE histograms |
| `plot_fe_vs_transformer_comparison_bar_chart` | FE vs transformer grouped bars |
| `plot_fe_vs_transformer_by_architecture` | By decoder variant bars |
| `plot_reconstruction_triple` | 3-row: original / FE recon / TR recon |
| `plot_per_bin_error_heatmap` | D1: models × 245 bins RMSE |
| `plot_pca_component_r2` | D2: R² per PCA component |
| `plot_residual_scatter` | D3: residual vs original scatter |
| `plot_fe_vs_transformer_r2` | D4: per-sample scatter + diagonal |
| `plot_all_decoder_variants` | Large grid: original + FE/TR per variant |
| `plot_structured_similarity_all_models` | 2×(1+K) grid: input + models |

---

## 10. Standalone Visualization Scripts

These scripts are **independent runners** — they do not integrate with `evaluation_runner.py`.

### 10.1 `plot_structured_similarity_with_fe.py`

**Post-hoc regeneration** of structured similarity plots from a finished `evaluation_runner` output directory. Reads saved `.npy` files and `eval_report_*.json`, optionally computes FE outputs.

```
python plot_structured_similarity_with_fe.py --eval_dir <timestamped_dir> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--eval_dir` | required | Timestamped eval results directory |
| `--nova_data_dir` | `/mnt5/noy/fairseq/data` | Nova datasets parent |
| `--recompute` | off | Force FE recomputation |

---

### 10.2 `plot_model_comparison.py`

Compares three fixed models: base_libri pretrained, FE-only trained, full-train. Generates waveform/spectrogram visualizations and 4-way similarity analysis.

**No CLI** — uses hardcoded paths. Run: `python plot_model_comparison.py`

**Outputs:** `eval_results/model_comparison/`

---

### 10.3 `plot_input_vs_features_comparison.py`

One-off analysis of input normalization: raw vs `F.layer_norm` through `ConvFeatureExtractionModel`.

**No CLI** — hardcoded paths. **Outputs:** `eval_results/normalization_stages/`

---

### 10.4 `plot_masking_examples.py`

Illustrative figures of causal vs random masking strategies on waveforms. Didactic, no model inference.

**No CLI** — hardcoded paths. **Outputs:** `eval_results/runai_comparison/plots/masking_examples/`

---

### 10.5 `eval_collapse_similarity.py`

Standalone collapse diagnosis — N×N cosine similarity heatmaps for transformer embeddings and CNN FE outputs, one row per checkpoint.

```
python eval_collapse_similarity.py --checkpoint_dir DIR [OPTIONS]
```

**Output:** `<output_dir>/<timestamp>/collapse_cosine_similarity.png`

---

### 10.6 `eval_normalize_with_runner.py`

One-off comparison of `task.normalize=True` vs `False` on validation loss. Uses `EvaluationRunner._eval_validation_loss`.

**No CLI** — hardcoded checkpoint path. Run: `python eval_normalize_with_runner.py`

---

## 11. Shell Orchestrator

### `run_comprehensive_eval.sh`

Bash script that runs four evaluation tracks in sequence for a checkpoint directory:

```bash
bash code/run_comprehensive_eval.sh [CHECKPOINT_DIR]
```

**Default config:**
- `CKPT_DIR`: `/mnt5/noy/SpectralFM/checkpoints/runai/fe_vs_transformer_collapse_with_var_loss`
- `OUTPUT_DIR`: `eval_results/var_loss_eval`
- `DECODER_VARIANTS`: `0 512 512:256`

**Execution order:**

1. **FE Decoder reconstruction** via `eval_fe_decoder.py`
   - `--decoder_variants 0 512 512:256 --include_embedding_decoder --best_only`
   - Output: `<root>/reconstruction/`

2. **Structured similarity + noise robustness** via `evaluation_runner.py`
   - `--eval_methods structured_similarity noise_robustness --best_only`
   - Output: `<root>/runner/<timestamp>/`

3. **Label regression plot suite** via `label_reg_evaluation.py`
   - `--output_dir $ROOT`
   - Output: `<root>/plots/label_reg_plots/`

4. Summary + log
   - `<root>/summary.txt`, `<root>/run.log`

---

## 12. WandB, Logging, and External Integrations

### 12.1 WandB Integration

WandB is integrated at **training** time, not during evaluation:

**In `model_loader.py` — `train_feature_extractor_only()`:**
- `wandb.init(project="SpectralFM", name=model_string)` — initializes a WandB run
- Logs per-epoch: `wandb.log({"epoch": epoch, "loss": loss})`
- Logs summary: `wandb.log({"avg_loss": avg_loss, "run_id": run_id})`
- Run name format: `experiment_3-mask_type={mask_type}-mask={mask_ratio}-arch={arch_type}-...`

**In `eval_fe_decoder.py` — checkpoint labeling:**
- `_wandb_run_name_from_checkpoint(checkpoint_path)` reads `cfg.common.wandb_run_name` from fairseq checkpoint metadata
- Used to label per-run output directories with the WandB run name when available

**In `wandb_logger.py`:**
- This is a **standalone demo** (Yelp reviews + DistilBERT + WandB) — **not** integrated with SpectralFM evaluation
- Sets `WANDB_PROJECT`, `WANDB_LOG_MODEL`, `WANDB_WATCH` environment variables
- Logs `accuracy` metric via HuggingFace `Trainer`

### 12.2 MLflow Integration

In `model_loader.py` — `train_feature_extractor_only()`:
- `mlflow.start_run(run_name="train_feature_extractor_only")`
- Logs parameters and metrics during FE-only training

### 12.3 Fairseq Training Integration

The fairseq `data2vec_audio.py` model handles its own training logging via Hydra and fairseq's built-in logging. The evaluation framework reads checkpoint configs produced by this training pipeline.

### 12.4 Report Generation

All evaluation scripts produce structured reports:
- **JSON** — machine-readable, full metric dictionaries
- **CSV** — tabular, one row per checkpoint
- **TXT** — human-readable summary
- **Markdown** — comparison table with image references

---

## 13. Metric Naming Convention

All metric keys in `EvalResult.metrics` follow prefix conventions:

| Prefix | Category | Example |
|--------|----------|---------|
| `label_reg_` | Supervised regression on parameter_0 | `label_reg_emb_r2`, `label_reg_improvement_r2` |
| `comp_cluster_` | Component clustering | `comp_cluster_ari`, `comp_cluster_nmi` |
| `comp_knn_` | Component KNN/neighbor | `comp_knn_precision_k10` |
| `repr_` | Representation geometry | `repr_cka_linear`, `repr_uniformity` |
| `spectral_` | Spectral-domain | `spectral_param_verify_auc` |
| `downstream_` | Downstream task results | `downstream_linear_probe_r2` |
| `noise_` | Noise robustness | `noise_gaussian_robustness_ratio` |
| `stack_` | Stack/similarity | `stack_match_improvement_pct` |
| `completion_` | Signal completion | `completion_cos_sim_mean` |
| `eval_` | Validation loss | `eval_loss` |
| `fe_dec_` | FE decoder reconstruction | `fe_dec_cosine_mean`, `fe_dec_r2` |
| `valid_` | Embedding similarity (valid set) | `valid_pearson_corr`, `valid_sim_variance_ratio` |
| `efficiency_` | Computational efficiency | `efficiency_flops` |
| `dynamics_` | Training dynamics | `dynamics_loss_trend` |
| `scaling_` | Scaling laws | `scaling_slope` |
| `transfer_` | Transfer evaluation | `transfer_r2_gain` |
| `mask_sweep_` | Mask sweep | `mask_sweep_cos_sim` |
| `ablation_` | Ablation analysis | `ablation_delta` |

**Rules:**
1. Snake_case only
2. Space names in multi-space metrics: `label_reg_emb_r2` vs `label_reg_input_r2`
3. Improvement metrics: `*_improvement_*` = embedding-space minus input-space

---

## 14. Past Experiments in `eval_results/`

The `code/eval_results/` directory contains results from all past evaluation runs. Key experiment groups:

### 14.1 RunAI Long-Train Evaluation

**Directory:** `runai_long_train_2026-02-25_13-46-46/`

The most extensive evaluation hub. Contains 10+ timestamped sub-runs comparing three main checkpoints:
- **2026-01-07_21-50-07** — Jan-07, single-channel 9k training
- **2026-02-25_13-46-46** — Feb-25, single-channel 18.5k training
- **2026-03-03_17-45-36-multi** — Mar-03, multi-channel 17.4k training

Includes: full `eval_report_*` suites, enriched JSONs, debug runs, noise robustness CSVs, clustering analysis (`clustering_single_channel_one*`), per-run spectrograms, and debug plots.

Training loss progression: **0.985 → 0.781 → 0.347** (shows training improvement over time).

---

### 14.2 Collapse Experiments

**Directories:** `compare_collapse/`, `compare_collapse_050426/`, `compare_collapse2/`, `collapse_similarity/`

Evaluation of `fe_vs_transformer_collapse` ablation runs comparing different FE/transformer training configurations:
- `fe-identity_trans-train_base` — identity FE, trained transformer
- `fe-train_trans-frozen_base` — trained FE, frozen transformer
- `fe-train_trans-train_base` — both trained

Results include embedding metrics CSVs, noise robustness CSVs, match DataFrames, signal completion CSVs, and comparison summaries.

---

### 14.3 FE Decoder Reconstruction

**Directories:** `fe_decoder/`, `fe_decoder_recon_only/`, `fe_decoder_20260415_011500/`, `fe_decoder_struct100_smoke_*`, `fe_decoder_v8_quick/`, `fe_decoder_wandbname_test/`

FE decoder reconstruction evaluations across different checkpoints and configurations. `fe_decoder_recon_only/` contains the clean single-checkpoint evaluation of `recon_only_l1_lr1e-4` with Linear, MLP-512, and MLP-512-256 decoder variants.

---

### 14.4 Reconstruction Loss Experiments

**Directories:** `recon_loss_experiment_1_*` (×4 timestamps), `recon_loss_rerun_20260413_235850/`, `structured_sim_fe_recon_loss_20260414/`

Evaluation of reconstruction-loss training variants (`recon-fe*_recon-tr*_frozen-enc*`). Includes FE decoder metrics, label regression comparison, and structured similarity with FE. The rerun directory contains both `fe_decoder/` and `label_regression/` subdirectories.

---

### 14.5 Linear Regression Layer Sweeps

**Directories:** `linear_regr_2026-01-07_21-50-07/`, `linear_regr_2026-02-25_13-46-46/`, `linear_regr_2026-03-03_17-45-36-multi/`

Per-checkpoint linear regression analysis with task directories:
- `task1_layer_sweep/` — layer-wise feature quality
- `task2_mlp_probe/` — MLP probe comparison
- `task3_pca/` — PCA analysis
- `task4_all_params/` — all-parameter evaluation

---

### 14.6 Label Regression Runs

**Directories:** `label_reg_20260415_010741/`, `label_reg_20260415_085856/`, `label_reg_20260415_090434/`

Timestamped outputs from `label_reg_evaluation.py` with scatter plots and distribution analyses.

---

### 14.7 Variance Loss Evaluation

**Directory:** `var_loss_eval/`

Comprehensive evaluation of `fe_vs_transformer_collapse_with_var_loss` checkpoints. Contains nested reconstruction, runner, and summary outputs across multiple dates.

---

### 14.8 Other Experiments

| Directory | Purpose |
|-----------|---------|
| `base_libri_test_clean/`, `base_libri_official_test_clean/` | LibriSpeech test-clean baseline |
| `comparison_4checkpoints/` | Four-checkpoint comparison |
| `full_20260307_*` | Full-mode evaluation runs |
| `sanity_validation_*` | Validation sanity checks |
| `validation_loss_2026-01-14_21-52-11/` | Early validation loss tracking |
| `spectrogram_plots/`, `spectrograms/` | Standalone spectrogram visualizations |
| `runai_comparison/` | RunAI model comparison + masking examples |

---

### 14.9 Timestamped Flat Runs

Directories like `20260415_085644/`, `20260414_235825/`, `20260317_213209/` etc. are individual `evaluation_runner.py` sessions. Most contain:
- `eval_report_*.{json,csv,txt,md}` — full reports
- `plots/` — per-run and comparison plots
- `data/` — cached numpy arrays
- `debug_plots/`, `spectrograms/` — optional debug outputs

---

## 15. Utility Modules Reference

### 15.1 `eval_utils.py`

| Function | Purpose |
|----------|---------|
| `extract_component_id(filename)` | Parse component ID from WAV basename |
| `extract_component_ids_from_dataset(dataset)` | IDs + filenames from fairseq dataset |
| `extract_component_ids_from_directory(data_dir)` | Scan `*.wav` in directory |
| `load_labels_tsv(dataset_dir)` | Read `labels.tsv` → `{filename: float}` |
| `extract_attention_maps(model, sample_input, device)` | Forward hooks for layer attention |
| `load_wav_files_torchaudio(data_dir, max_samples, target_length)` | Light WAV loading (pad/truncate to 245) |
| `build_structured_similarity_subset(nova_data_dir, seed)` | Build 100-sample structured panel |
| `load_structured_similarity_spectrograms(nova_data_dir, seed)` | Build + load + normalize |
| `extract_fe_outputs_from_fairseq_checkpoint(ckpt, samples, device)` | CNN FE → LN → proj → mean pool |
| `analyze_failures(y_true, y_pred)` | Worst-percentile error analysis |
| `DatasetSplitter` | Train/test splits by component, overlap analysis |

### 15.2 `eval_metrics.py`

All functions return `Dict[str, float]` metric dictionaries:

| Function | Metric prefix |
|----------|--------------|
| `compute_fe_decoder_metrics` | `fe_dec_*` |
| `compute_knn_metrics` | `knn_*` |
| `compute_component_clustering_metrics` | `comp_cluster_*` |
| `compute_repr_geometry_metrics` | `repr_*` |
| `compute_linear_probing_metrics` | `downstream_*` |
| `compute_fewshot_metrics` | `downstream_fewshot_*` |
| `compute_parameter_verification` | `spectral_param_verify_*` |
| `compute_efficiency_metrics` | `efficiency_*` |
| `compute_scaling_law_metrics` | `scaling_*` |
| `compute_transfer_metrics` | `transfer_*` |
| `compute_extended_signal_completion` | `ext_completion_*` |
| `compute_mask_sweep_metrics` | `mask_sweep_*` |
| `compute_training_dynamics_metrics` | `dynamics_*` |
| `compute_ablation_analysis` | `ablation_*` |
| `compute_metric_intercorrelation` | `metric_corr_*` |
| `bootstrap_confidence_interval` | Statistical helper |
| `paired_bootstrap_test` | Statistical helper |

### 15.3 `eval_plots.py`

See [Section 9.4](#94-eval_plotspy--full-function-catalog) for the complete function catalog. All functions accept data + output path, return the saved file path. Uses `matplotlib.use("Agg")` for headless rendering.

---

## 16. Recipes and Quick-Start

### Smoke test on a single checkpoint

```bash
python evaluation_runner.py \
  --checkpoint /mnt5/noy/SpectralFM/checkpoints/runai/2026-01-07_21-50-07/checkpoint_best.pt \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_100 \
  --output_dir eval_results/smoke_test
```

### Compare all collapse experiments

```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --eval_methods embedding_similarity structured_similarity \
  --output_dir eval_results/compare_collapse
```

### Full FE decoder + structured sim + label reg (comprehensive)

```bash
bash code/run_comprehensive_eval.sh /path/to/checkpoint_dir
```

### FE decoder reconstruction only

```bash
python eval_fe_decoder.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/checkpoints/runai/recon_loss_experiment_1 \
  --best_only \
  --decoder_variants 0 512 512:256 \
  --include_embedding_decoder \
  --output_dir eval_results/fe_decoder_recon
```

### Label regression with custom checkpoints

```bash
python eval_label_regression.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/checkpoints/runai/recon_loss_experiment_3 \
  --max_samples 2000 \
  --output_dir eval_results/label_reg_test
```

### Regenerate structured similarity plots from existing run

```bash
python plot_structured_similarity_with_fe.py \
  --eval_dir eval_results/20260317_213209 \
  --recompute
```

### Training dynamics (metric trajectory across checkpoints)

```bash
python run_full_evaluation.py \
  --mode training_dynamics \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse/fe-train_trans-train_base \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --output_dir eval_results/dynamics
```

### Compare specific checkpoints directly

```bash
python compare_checkpoints.py \
  --checkpoints \
    /mnt5/noy/SpectralFM/checkpoints/runai/2026-01-07_21-50-07/checkpoint_best.pt \
    /mnt5/noy/SpectralFM/checkpoints/runai/2026-02-25_13-46-46/checkpoint_best.pt \
  --eval_methods validation_loss embedding_similarity \
  --output_dir eval_results/direct_compare
```

### What can be run together vs individually

| Script | Runs as part of other scripts? | Individual runner? |
|--------|-------------------------------|-------------------|
| `evaluation_runner.py` | Used by `compare_checkpoints.py`, `run_full_evaluation.py`, `run_comprehensive_eval.sh` | Yes |
| `eval_fe_decoder.py` | Used by `run_comprehensive_eval.sh` | Yes |
| `eval_label_regression.py` | Used by `run_comprehensive_eval.sh` | Yes |
| `label_reg_evaluation.py` | Used by `run_full_evaluation.py`, `run_comprehensive_eval.sh` | Yes |
| `compare_checkpoints.py` | Standalone | Yes |
| `run_full_evaluation.py` | Standalone orchestrator | Yes |
| `plot_structured_similarity_with_fe.py` | Standalone post-hoc | Yes |
| `eval_collapse_similarity.py` | Standalone | Yes |
| All `plot_*.py` | Standalone | Yes |

**Within `evaluation_runner.py`**, all `--eval_methods` run in a single invocation and share pre-loaded samples for fair comparison. The shell script `run_comprehensive_eval.sh` is the only mechanism that orchestrates `eval_fe_decoder.py` + `evaluation_runner.py` + `label_reg_evaluation.py` together.
