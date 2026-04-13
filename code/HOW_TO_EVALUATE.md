# SpectralFM Evaluation Manual

All evaluation is run from the `code/` directory.

```bash
cd /mnt5/noy/SpectralFM/code
```

---

## Table of Contents

1. [Quick-start examples](#1-quick-start-examples)
2. [Entry points](#2-entry-points)
3. [Evaluation methods](#3-evaluation-methods)
4. [Output files](#4-output-files)
5. [Checkpoint selection](#5-checkpoint-selection)
6. [Data selection](#6-data-selection)
7. [Multi-checkpoint side-by-side comparison](#7-multi-checkpoint-side-by-side-comparison)
8. [Outlier / inlier analysis](#8-outlier--inlier-analysis)
9. [Full CLI reference — evaluation_runner.py](#9-full-cli-reference--evaluation_runnerpy)
10. [Full CLI reference — run_full_evaluation.py](#10-full-cli-reference--run_full_evaluationpy)
11. [Common recipes](#11-common-recipes)

---

## 1. Quick-start examples

### Smoke test on a single checkpoint
```bash
python evaluation_runner.py \
  --checkpoint /mnt5/noy/SpectralFM/checkpoints/runai/2026-01-07_21-50-07/checkpoint_best.pt \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_100 \
  --output_dir eval_results/smoke_test
```

### Compare all collapse experiments side-by-side
```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --eval_methods embedding_similarity \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --output_dir eval_results/compare_collapse
```

### Structured similarity — collapse diagnosis with block heatmaps and dive-in plots
```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --eval_methods structured_similarity \
  --output_dir eval_results/compare_collapse
  # --nova_data_dir defaults to /mnt5/noy/SpectralFM/fairseq/data/nova_data
```

### Label regression — measure if embeddings encode parameter_0
```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --eval_methods label_regression \
  --output_dir eval_results/compare_collapse
  # --labeled_data_dir defaults to /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data
```

### Compare specific stacks only
```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --run_names fe-identity_trans-train_base fe-train_trans-train_base fe-train_trans-frozen_base \
  --eval_methods embedding_similarity stack_similarity \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --output_dir eval_results/collapse_subset
```

### Full suite on one checkpoint
```bash
python run_full_evaluation.py \
  --mode full \
  --checkpoint /mnt5/noy/SpectralFM/checkpoints/runai/2026-01-07_21-50-07/checkpoint_best.pt \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --output_dir eval_results/full_eval
```

---

## 2. Entry points

| Script | Best for |
|---|---|
| `evaluation_runner.py` | **Main workhorse.** Multi-checkpoint comparison, full plot suite, outlier analysis. |
| `run_full_evaluation.py` | Single-checkpoint deep dives, scaling-law sweeps, training-dynamics trajectories. |

Both scripts create a **timestamped subdirectory** inside `--output_dir` automatically, so re-running the same command never overwrites previous results.

---

## 3. Evaluation methods

Pass one or more to `--eval_methods`. Use `--all_methods` to run all six.

| Method | Flag | What it measures | Notes |
|---|---|---|---|
| Embedding similarity | `embedding_similarity` | Pairwise cosine similarity of learned embeddings; Pearson/Spearman with input space; variance ratio | **Default.** Always runs first. Also extracts CNN FE outputs for cross-run plots. |
| Stack similarity | `stack_similarity` | How well top-K neighbors in embedding space match stack membership from filenames | Requires multi-channel filenames with stack IDs |
| Structured similarity | `structured_similarity` | Block-structured cosine similarity across four datasets; Input / FE / Embedding heatmaps with annotated dive-in plots | Needs `--nova_data_dir` (default already set). Best collapse diagnostic. |
| Label regression | `label_regression` | Linear probe on `parameter_0` from `labeled_data`; R², Pearson r, MAE at multiple train sizes vs raw-input baseline | Needs `--labeled_data_dir` (default already set). Best measure of learned representation quality. |
| Noise robustness | `noise_robustness` | How stable embeddings are under Gaussian / time-stretch / pitch-shift noise | Slow; skip for quick sweeps |
| Signal completion | `signal_completion` | Can the model predict masked-out portions of the signal? | Uses the model's mask predictor |
| Validation loss | `validation_loss` | Re-computes the training loss on the eval split | Useful as a sanity check |

### Collapse diagnosis methods (most important for current experiments)

`embedding_similarity` + `structured_similarity` together give the clearest collapse picture:

```bash
--eval_methods embedding_similarity structured_similarity
```

### `structured_similarity` — how it works

Unlike every other method (which operates on a single `--eval_data_dir`), `structured_similarity` builds a **fixed 100-sample subset** that spans four datasets at once:

```
nova_data_dir/                       ← default: /mnt5/noy/SpectralFM/fairseq/data/nova_data
├── single_channel_all/   ← 30 samples  (3 stacks × 10, stack = dataset_index // 10)
├── multi_channel/        ← 30 samples  (3 components × 10)
├── sampled_data/         ← 20 samples  (2 components × 10)
└── labeled_data/         ← 20 samples  (2 components × 10)
```

Each 10-sample group comes from the **same physical component or stack**, so the resulting 100×100 similarity matrix has 10 clearly visible blocks on the diagonal. Collapse is immediately visible: a healthy model shows bright blocks on a dark background; a collapsed model shows a uniformly bright matrix everywhere.

`--nova_data_dir` has a sensible default (`/mnt5/noy/SpectralFM/fairseq/data/nova_data`) so you usually don't need to pass it. If the datasets aren't there the runner skips the method with a warning.

```bash
# Run structured_similarity with the default nova_data_dir
python code/evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --eval_methods structured_similarity

# Override nova_data_dir if your datasets live elsewhere
python code/evaluation_runner.py \
  --checkpoint_dir ... \
  --eval_methods structured_similarity \
  --nova_data_dir /my/custom/nova_data
```

### `label_regression` — how it works

Trains a linear probe (L-BFGS) on the model's transformer embeddings to predict the continuous `parameter_0` label from `labeled_data/labels.tsv`. The same probe is also trained on raw input features as a **baseline**. Both are evaluated on a held-out set.

**Why it matters:** Unlike the similarity metrics (which are unsupervised), label regression directly measures whether the model has learned to encode a semantically meaningful property. A collapsed model scores near or below the raw-input baseline regardless of train size.

**What gets evaluated:** train sizes `[100, 500, 1000, 2000]` — shows how quickly useful structure emerges with more labeled data. The "Training size" scatter grid (rows = train size, columns = models) is the main cross-run output.

```bash
# Run label_regression (labeled_data_dir defaults to nova_data/labeled_data)
python code/evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --eval_methods label_regression

# Override if your labeled_data lives elsewhere
python code/evaluation_runner.py \
  --checkpoint_dir ... \
  --eval_methods label_regression \
  --labeled_data_dir /my/custom/labeled_data
```

**Metrics** (prefix `label_reg_`):

| Key | Description |
|---|---|
| `label_reg_emb_r2_{n}` | R² of embedding probe trained on `n` samples |
| `label_reg_input_r2_{n}` | R² of raw-input baseline (same probe, raw features) |
| `label_reg_improvement_{n}` | R² gain: embedding − baseline |
| `label_reg_emb_pearson_{n}` | Pearson r of embedding probe |
| `label_reg_best_r2` | Best R² across all train sizes |

`--labeled_data_dir` defaults to `/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data`. The directory must contain `valid.tsv` (audio manifest) and `labels.tsv` (`filename\tparameter_0` per line). If missing, the method is skipped with a warning.

---

## 4. Output files

Every evaluation run writes to a **timestamped directory** under `--output_dir`:

```
<output_dir>/
├── plots/                                  # All cross-run comparison plots
│   ├── embedding_similarity_comparison_valid.png        # 2-row grid: embeddings (top) + CNN FE (bottom) per checkpoint
│   ├── embedding_similarity_histogram_comparison_valid.png   # Histogram of pairwise cosine sim distributions
│   ├── embedding_similarity_matrix_comparison_valid.png      # N×N heatmaps side-by-side
│   ├── embedding_metrics_bar_comparison_valid.png       # Bar charts: Pearson / Spearman / variance ratio / mean sim
│   ├── all_models_structured_similarity_with_fe.png     # Full 100-sample block-structured grid (requires structured_similarity)
│   ├── per_group_stack_29815_structured_similarity_with_fe.png   # 10×10 block per group (×10 groups)
│   ├── per_dataset_single_channel_all_structured_similarity_with_fe.png  # All samples per dataset (×4 datasets)
│   ├── divein_stack_29815_structured_similarity.png     # Waveform thumbnails + annotated 10×10 heatmaps (×10 groups)
│   ├── label_regression_comparison_train_size.png  # Cross-run scatter grid: rows=train size, cols=models (requires label_regression, 2+ checkpoints)
│   ├── <run_name>/
│   │   ├── embedding_similarity_comparison_valid.png    # Per-checkpoint 3/4-way panel (input / embedding / frozen / random)
│   │   ├── similarity_matrices.png                     # 2-panel input vs embedding matrix
│   │   ├── structured_similarity.png                   # Per-checkpoint 3-row: Input / FE / Embedding
│   │   └── label_regression_scatter.png                # Per-checkpoint 2-col scatter: Input baseline | Embedding (one row per train size)
│   └── ...
├── data/                                   # Cached numpy arrays (reusable without re-running the model)
│   ├── embeddings_<run>_valid.npy
│   ├── fe_outputs_<run>_valid.npy          # CNN feature extractor outputs
│   ├── inputs_<run>_valid.npy
│   ├── embeddings_<run>_structured_similarity.npy   # Structured subset arrays (per run)
│   ├── fe_outputs_<run>_structured_similarity.npy
│   ├── inputs_<run>_structured_similarity.npy
│   ├── embeddings_<run>_label_reg.npy               # Label regression arrays (per run)
│   ├── inputs_<run>_label_reg.npy
│   └── embedding_similarity_scores_<run>_valid.npy
├── eval_report_<timestamp>.json            # Full metrics for all checkpoints
├── eval_report_<timestamp>.csv             # Same, tabular format
├── eval_report_<timestamp>_summary.txt     # Human-readable summary
└── eval_report_<timestamp>_comparison.md  # Markdown comparison table
```

### Key plots explained

**`embedding_similarity_comparison_valid.png`** (the main diagnostic — **requires 2+ checkpoints**)
Two rows per checkpoint column:
- Row 0 — transformer embedding similarity matrix. A healthy model shows clear off-diagonal variation; a collapsed model shows a nearly uniform bright matrix.
- Row 1 — CNN feature extractor (FE) output similarity matrix. Comparing this to row 0 tells you *where* collapse originated: if row 1 is already bright/uniform, collapse started in the FE; if only row 0 is bright, the transformer is the culprit.

> This plot (and all other cross-run comparison plots) are only generated when evaluating **2 or more checkpoints** in a single run. Use `--checkpoint_dir` rather than `--checkpoint` to trigger them. The per-checkpoint FE vectors are always cached to `data/fe_outputs_<run>_valid.npy` as long as `embedding_similarity` is in `--eval_methods`.

**`embedding_similarity_histogram_comparison_valid.png`**
Distribution of all pairwise cosine similarities per checkpoint. Healthy: broad distribution centered around 0.5–0.8. Collapsed: spike at ~1.0.

**`embedding_metrics_bar_comparison_valid.png`** (new)
Four bar charts across all checkpoints: Pearson correlation with input space, Spearman correlation, variance ratio (higher = less collapsed), and mean embedding similarity (lower = less collapsed). Use this for a quick numerical ranking.

**`all_models_structured_similarity_with_fe.png`** (requires `structured_similarity`, 2+ checkpoints)
Full 100×100 block-structured heatmap grid. Two rows:
- Row 0 — transformer embeddings per model (+ Input Space as column 0)
- Row 1 — CNN FE outputs per model (+ Input Space)

A healthy model shows 10 bright 10×10 blocks on a dark background. A collapsed model shows a uniformly bright matrix. Comparing rows 0 and 1 tells you where collapse starts.

**`divein_<group>_structured_similarity.png`** (requires `structured_similarity`, 2+ checkpoints)
One file per 10-sample group (10 files total). Each figure has:
- Left thin column: waveform thumbnails for each of the 10 samples (aligned to heatmap rows)
- Remaining columns: fully annotated 10×10 cosine-similarity heatmaps — one for Input Space, then embeddings and FE output per model

Cell values are printed inside each cell so you can read exact per-pair similarities. Title shows group name, source dataset, and sample index range.

**`label_regression_comparison_train_size.png`** (requires `label_regression`, 2+ checkpoints)
Cross-run scatter grid: rows = train sizes (100 / 500 / 1000 / 2000), column 0 = raw-input baseline, remaining columns = one per model. Each cell is a "true vs predicted `parameter_0`" scatter plot annotated with R², Pearson r, and MAE. The best-performing cell per row is highlighted. Use this to judge whether a model's representations capture `parameter_0` better than naive input features and how efficiently they do so (i.e. good R² with only 100 samples = strong representation).

**`<run_name>/label_regression_scatter.png`** (per checkpoint)
Same scatter grid but for a single checkpoint only (Input baseline + Embedding columns). Generated alongside `structured_similarity.png` during per-checkpoint evaluation.

---

## 5. Checkpoint selection

### From a directory (auto-discovery)

`--checkpoint_dir` scans recursively for `checkpoint_best.pt`, `checkpoint_last.pt`, and numbered checkpoints.

```bash
# All checkpoints in a directory tree
--checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse

# Only checkpoint_best.pt files (recommended for comparisons)
--checkpoint_dir ... --best_only

# Only the most recently modified checkpoint
--checkpoint_dir ... --latest_only

# Specific run names (subdirectory names)
--checkpoint_dir ... --run_names fe-identity_trans-train_base fe-train_trans-train_base
```

Run names are the subdirectory names under `--checkpoint_dir`. For the collapse experiments they look like `fe-identity_trans-train_base`, `fe-train_trans-frozen_base`, etc. (see `HOW_TO_RUN.md`).

### Single checkpoint file

```bash
--checkpoints /mnt5/noy/SpectralFM/checkpoints/runai/2026-01-07_21-50-07/checkpoint_best.pt
```

### Multiple explicit checkpoint files

```bash
--checkpoints \
  /mnt5/noy/SpectralFM/checkpoints/runai/2026-01-07_21-50-07/checkpoint_best.pt \
  /mnt5/noy/SpectralFM/checkpoints/runai/2026-02-25_13-46-46/checkpoint_best.pt
```

`--checkpoints` accepts any number of `.pt` paths. The run name is inferred from the parent directory (the folder above `checkpoints/`), so the two checkpoints above get run names `2026-01-07_21-50-07` and `2026-02-25_13-46-46`.

### Checkpoint paths reference

| Location | Path |
|---|---|
| Collapse ablation outputs | `/mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse/` |
| RunAI checkpoints (best) | `/mnt5/noy/SpectralFM/checkpoints/runai/<run-name>/checkpoint_best.pt` |
| RunAI grouped runs | `/mnt5/noy/SpectralFM/checkpoints/runai/2026-03-10-compare-single-to-multi/` |

---

## 6. Data selection

Use `--eval_data_dir` to override the dataset. Priority order:
1. `--eval_data_dir` (explicit, recommended)
2. `--data_dir`
3. `cfg.task.data` from the checkpoint config (fallback)

### Dataset reference

| Dataset | Samples | Use case | Path |
|---|---|---|---|
| `single_channel_100` | 100 | Smoke test / debug | `fairseq/data/nova_data/single_channel_100` |
| `single_channel_1k` | 1,000 | Fast iteration | `fairseq/data/nova_data/single_channel_1k` |
| `single_channel_10k` | 22,222 | **Default for most evaluations** | `fairseq/data/nova_data/single_channel_10k` |
| `single_channel_one` | 1,000,000 | Large-scale embedding eval | `fairseq/data/nova_data/single_channel_one` |
| `multi_channel` | 3,412,476 | Multi-component clustering | `fairseq/data/nova_data/multi_channel` |
| `labeled_data` | 66,024 | Supervised label regression | `fairseq/data/nova_data/labeled_data` |
| `sampled_data` | 22,428 | Diverse component coverage | `fairseq/data/nova_data/sampled_data` |

All paths are relative to `/mnt5/noy/SpectralFM/`.

### Evaluating on two datasets at once

`--custom_dataset_path` runs `embedding_similarity` on a second dataset in addition to the "valid" dataset, producing a 4-way comparison panel and separate FE+embedding comparison plots for each dataset.

```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --eval_methods embedding_similarity \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --custom_dataset_path /mnt5/noy/SpectralFM/fairseq/data/nova_data/multi_channel \
  --output_dir eval_results/dual_dataset
```

---

## 7. Multi-checkpoint side-by-side comparison

> **Cross-run plots only appear when `len(results) > 1`.** Always use `--checkpoint_dir` (not `--checkpoint`) to get them. The FE output vectors (`data/fe_outputs_*.npy`) are cached per-checkpoint regardless.

When `--checkpoint_dir` returns more than one checkpoint, the runner automatically generates these cross-run plots after finishing all individual evaluations:

| Plot | Triggered by |
|---|---|
| Embedding + FE 2-row grid | `embedding_similarity` |
| Histogram comparison | `embedding_similarity` |
| Matrix heatmap comparison | `embedding_similarity` + `--plot_matrices` |
| Metrics bar chart | `embedding_similarity` |
| Noise robustness bars | `noise_robustness` |
| Stack similarity bars | `stack_similarity` |
| Signal completion bars | `signal_completion` |

The same 100 samples (fixed seed 42) are used for every checkpoint to ensure all comparisons are fair.

---

## 8. Outlier / inlier analysis

Find samples that are hardest (or easiest) for the model to distinguish:

```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --run_names fe-train_trans-train_base \
  --eval_methods embedding_similarity \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --output_dir eval_results/outliers \
  --analyze_outliers \
  --outlier_run_name fe-train_trans-train_base \
  --outlier_similarity_type both
```

- `--analyze_outliers` — finds samples with the **lowest** average pairwise similarity (the ones the model considers most unique)
- `--analyze_inliers` (default: on) — also finds samples with the **highest** average similarity (the collapsed / hardest-to-distinguish ones)
- `--outlier_similarity_type embedding|input|both` — which similarity space to use for ranking
- `--outlier_run_name` — restrict analysis to one specific run
- `--outlier_dataset` — restrict to a specific dataset (defaults to `custom_dataset_path` name or `"valid"`)

Plots land in `<output_dir>/plots/<run_name>/`.

---

## 9. Full CLI reference — `evaluation_runner.py`

```
python evaluation_runner.py [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint_dir PATH` | `/mnt5/noy/fairseq/outputs` | Root directory to scan for checkpoints |
| `--checkpoint PATH` | — | Path to a single `.pt` file (skips directory scan) |
| `--output_dir PATH` | `code/eval_results` | Where to write results (timestamped subdir created automatically) |
| `--eval_data_dir PATH` | — | Evaluation dataset override (recommended over `--data_dir`) |
| `--eval_methods M [M …]` | `embedding_similarity` | One or more of: `embedding_similarity`, `stack_similarity`, `noise_robustness`, `signal_completion`, `validation_loss` |
| `--all_methods` | off | Run all five evaluation methods |
| `--best_only` | off | Only evaluate `checkpoint_best.pt` files |
| `--latest_only` | off | Only evaluate the most recently modified checkpoint |
| `--run_names N [N …]` | — | Whitelist of run names to evaluate (must match subdirectory names exactly) |
| `--plot_matrices` | off | Generate N×N similarity matrix heatmap comparison |
| `--custom_dataset_path PATH` | — | Second dataset for 4-way comparison alongside the default valid split |
| `--include_random_weights` | off | Add a random-init baseline to the per-checkpoint 4-way comparison (makes it 5-way) |
| `--debug` | off | Save sample spectrograms to `debug_plots/` for inspection |
| `--report_name NAME` | auto | Custom prefix for the report files |
| `--analyze_outliers` | off | Find and visualize samples with lowest average embedding similarity |
| `--no_analyze_inliers` | off | Skip the inlier half of outlier analysis |
| `--outlier_run_name NAME` | all runs | Restrict outlier analysis to this run |
| `--outlier_dataset NAME` | `valid` | Dataset name to use for outlier analysis |
| `--outlier_similarity_type` | `both` | `embedding`, `input`, or `both` |

---

## 10. Full CLI reference — `run_full_evaluation.py`

```
python run_full_evaluation.py --mode MODE [OPTIONS]
```

| Mode | Description |
|---|---|
| `full` | All applicable methods on a single checkpoint |
| `scaling_laws` | Sweep across checkpoints in `--checkpoint_dir` (dataset-size scaling) |
| `transfer` | Cross-dataset transfer evaluation |
| `light` | No fairseq — uses torchaudio + HuggingFace model |
| `training_dynamics` | Evaluates intermediate checkpoints to plot metric trajectories |
| `ablation` | Generate ablation study configs |

| Flag | Default | Description |
|---|---|---|
| `--mode MODE` | `full` | Evaluation mode (see table above) |
| `--checkpoint PATH` | — | `.pt` file (required for `full` mode) |
| `--checkpoint_dir PATH` | — | Directory of checkpoints (required for `scaling_laws`, `training_dynamics`) |
| `--output_dir PATH` | `code/eval_results` | Output root (`<mode>_<timestamp>/` subdir created) |
| `--eval_data_dir PATH` | — | Dataset override |
| `--max_samples N` | `500` | Max samples per evaluation |
| `--include_random_weights` | off | Include random-init baseline |

---

## 11. Common recipes

### Track a single checkpoint thoroughly

```bash
python evaluation_runner.py \
  --checkpoint /mnt5/noy/SpectralFM/checkpoints/runai/2026-01-07_21-50-07/checkpoint_best.pt \
  --all_methods \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --custom_dataset_path /mnt5/noy/SpectralFM/fairseq/data/nova_data/multi_channel \
  --plot_matrices \
  --debug \
  --output_dir eval_results/deep_dive_$(date +%Y%m%d)
```

### Compare all collapse experiments (full matrix)

```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --eval_methods embedding_similarity stack_similarity \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --plot_matrices \
  --output_dir eval_results/compare_collapse_$(date +%Y%m%d)
```

### Collapse ablation: only the three key variants

```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --run_names fe-identity_trans-train_base fe-train_trans-frozen_base fe-train_trans-train_base \
  --eval_methods embedding_similarity \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --output_dir eval_results/collapse_key3
```

### Training dynamics (metric trajectory across checkpoints)

```bash
python run_full_evaluation.py \
  --mode training_dynamics \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse/fe-train_trans-train_base \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --output_dir eval_results/dynamics_fe_train
```

### Diagnose collapse on a live run (latest checkpoint only)

```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --latest_only \
  --eval_methods embedding_similarity \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_100 \
  --output_dir eval_results/live_check
```

### Find outliers in the best collapse run

```bash
python evaluation_runner.py \
  --checkpoint_dir /mnt5/noy/SpectralFM/fairseq/outputs/fe_vs_transformer_collapse \
  --best_only \
  --run_names fe-train_trans-train_base \
  --eval_methods embedding_similarity \
  --eval_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
  --analyze_outliers \
  --outlier_run_name fe-train_trans-train_base \
  --outlier_similarity_type both \
  --output_dir eval_results/outliers_$(date +%Y%m%d)
```
