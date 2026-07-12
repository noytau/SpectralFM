# SpectralFM Evaluation Suite

Lightweight evaluation package — **no fairseq required**. Runs in a plain venv.

Entry point: **`eval/runner.py`** (`python -m eval.runner`). Every run writes a
timestamped directory under `--output_dir` containing `eval_report.html`,
`eval_report.md`, `run_info.md`, all figures (PNG), and CSV exports.

## Install

```bash
pip install -r requirements.txt
```

## Usage modes

### 1. Single-checkpoint evals

```bash
python -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode file --checkpoint_path /path/to/checkpoint.pt \
  --evals embedding_similarity noise_robustness clustering label_regression \
  --labeled_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data \
  --output_dir /mnt5/noy/SpectralFM/code/eval_outputs
```

### 2. Multi-checkpoint comparison

One summary row + one report section (with all per-checkpoint figures) per checkpoint.

```bash
python -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode multiple --checkpoint_path /dir/of/checkpoints \
  --evals checkpoint_comparison \
  --nova_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data \
  --labeled_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data \
  --output_dir /mnt5/noy/SpectralFM/code/eval_outputs
```

To compare an explicit list of files (instead of a directory glob), use the Python API
with `EvalConfig(checkpoint_paths=[...])`.

### 3. Signal reconstruction (multi-component checkpoints)

Unlike the other evals, reconstruction loads **separate checkpoints for each model
component** — an FE autoencoder checkpoint and/or a transformer-recon checkpoint —
because those parts are trained and saved independently
(branch: `reconstruction-loss-experiments`).

```bash
python -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --evals signal_reconstruction \
  --recon_fe_ckpt /mnt5/noy/SpectralFM/autoencoder_experiments/fe_signal_recon_100k_.../ckpt_....pt \
  --recon_tr_ckpt /mnt5/noy/SpectralFM/autoencoder_experiments/transformer_recon_.../ckpt_tr_....pt \
  --output_dir /mnt5/noy/SpectralFM/code/eval_outputs
```

| Arg | Checkpoint keys | Reconstruction pathway |
|-----|-----------------|------------------------|
| `--recon_fe_ckpt` | `encoder` / `layer_norm` / `decoder` | input → FE → LayerNorm → MirrorDecoder → signal |
| `--recon_tr_ckpt` | `transformer_mirror` / `backbone_ckpt` | input → FE → LN → proj → Transformer → TransformerMirrorDecoder → signal |

The transformer backbone is loaded from the fairseq checkpoint recorded in the TR
checkpoint's `backbone_ckpt` field (remapped into the HF `Data2VecAudioModel`, no
fairseq import). Per-sample `F.layer_norm` normalization is applied when the
checkpoint records it; override with `--recon_normalize true|false`.

Modes combine freely, e.g. `--evals checkpoint_comparison signal_reconstruction`.

## Evaluations

| Name | Description |
|------|-------------|
| `embedding_similarity` | Same-stack query: top-k neighbors in input vs embedding space, match rates + full per-query CSV (`match_df_*.csv`) |
| `signal_reconstruction` | True reconstruction through the full pipeline from per-component checkpoints (see above) |
| `noise_robustness` | Cosine similarity of clean vs noisy embeddings, 6 noise types + noise example grid |
| `clustering` | KMeans on embeddings vs stack labels: ARI / NMI / silhouette + t-SNE/UMAP scatter |
| `label_regression` | Ridge probe predicting `parameter_0` from inputs vs embeddings (needs `--labeled_data_dir`) |
| `structured_similarity` | Cosine maps of the canonical 100-sample panel at 4 pipeline stages (needs `--nova_data_dir`) |
| `checkpoint_comparison` | Runs the above (except reconstruction) across multiple checkpoints |

Every method runs **both** standalone (`--evals <name> ...`) and inside
`checkpoint_comparison` — same figures and CSVs either way.

## Checkpoint modes (`--checkpoint_mode`)

| Mode | `--checkpoint_path` | Description |
|------|---------------------|-------------|
| `hf` | — | `facebook/data2vec-audio-base` from HuggingFace |
| `file` | `.pt` file | Single checkpoint (fairseq / fe-recon / tr-recon / state_dict formats auto-detected) |
| `dir` | directory | Picks `checkpoint_best.pt` → `checkpoint_last.pt` → latest numbered |
| `multiple` | directory | All files matching `--checkpoint_pattern`, for `checkpoint_comparison` |

## Output directory contents

| File | Content |
|------|---------|
| `eval_report.html` | Self-contained report (all figures embedded) |
| `eval_report.md` / `run_info.md` | Markdown report / what was analyzed |
| `checkpoint_comparison.png` | Scalar metric grid across checkpoints |
| `noise_example_grid_*.png`, `noisy_vs_clean_*.png` | Noise robustness examples |
| `recon_overlay.png`, `recon_mse_bars.png` | Signal reconstruction figures |
| `similarity_comparison_{best,worst}_query*.png` | Same-stack query neighbor grids |
| `kmeans_clustered_then_{tsne,umap}_*.png` | Clustering scatters |
| `label_reg_true_vs_pred_*.png`, `label_regression_comparison.png` | Label regression |
| `struct_sim_*.png` | Structured similarity 4-panel cosine maps |
| `match_df_*.csv`, `noise_df_*.csv`, `recon_df.csv`, `*_comparison_df.csv` | Full per-sample / per-query results |
