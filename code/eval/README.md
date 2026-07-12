# SpectralFM Evaluation Suite

Lightweight, **fairseq-free** evaluation package for SpectralFM models
(`facebook/data2vec-audio-base` backbone with a custom 1D-CNN feature extractor,
trained on 245-point NOVA radio spectrograms).

Entry point: **`eval/runner.py`** (`python -m eval.runner`). Every run writes a
timestamped directory under `--output_dir` containing a self-contained
`eval_report.html` (all figures embedded), `eval_report.md`, `run_info.md`,
every figure as PNG, and full per-sample CSV exports.

---

## Installation

Requires Python ≥ 3.10. No fairseq, no conda dependency — a plain venv works:

```bash
git clone https://github.com/noytau/SpectralFM.git && cd SpectralFM
git checkout eval-methods

python3 -m venv .venv && source .venv/bin/activate
pip install -r code/eval/requirements.txt
```

`requirements.txt` contents and why each package is needed:

| Package | Used for |
|---|---|
| `torch`, `torchaudio` | model inference, wav loading |
| `transformers` | HF `Data2VecAudioModel` backbone |
| `numpy`, `scipy`, `pandas` | metrics, stats, dataframes |
| `scikit-learn` | KMeans, ridge probes, KNN, cosine similarity |
| `matplotlib`, `seaborn` | all figures |
| `soundfile` | fast wav reading |
| `tabulate` | markdown tables in reports |
| `omegaconf` | unpickling fairseq-format checkpoints **without** fairseq |
| `umap-learn` *(optional)* | UMAP clustering scatter (falls back to t-SNE only) |

First run downloads `facebook/data2vec-audio-base` (~360 MB) from HuggingFace;
set `HF_HOME` if you need a custom cache location.

---

## Running on Geoffrey

Geoffrey already has everything installed in the `spectralfm_env` conda env
(torch 2.8, transformers 4.57). The repo lives at `/mnt5/noy/SpectralFM`.

```bash
ssh Geoffry

# Pick a free GPU first (GPU 3 is dead — currently use 4):
nvidia-smi

cd /mnt5/noy/SpectralFM/code
CUDA_VISIBLE_DEVICES=4 /mnt5/noy/miniconda3/envs/spectralfm_env/bin/python3 -u -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode file \
  --checkpoint_path /mnt5/noy/SpectralFM/checkpoints/recon_runs_copied/tv_fe_short_3.pt \
  --evals embedding_similarity noise_robustness clustering label_regression structured_similarity \
  --recon_ckpt /mnt5/noy/SpectralFM/checkpoints/recon_runs_copied/tv_fe_short_3.pt \
  --n_holdout_stacks 20 --k 5 --batch_size 64 \
  --nova_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data \
  --labeled_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data \
  --output_dir /mnt5/noy/SpectralFM/code/eval_outputs
```

Outputs land in `/mnt5/noy/SpectralFM/code/eval_outputs/<YYYY-MM-DD_HH-MM-SS>/`.
Open `eval_report.html` in a browser — it is fully self-contained.

For long runs, wrap with `nohup ... > /tmp/eval_run.log 2>&1 &` and `tail -f` the log.

### Data paths on the cluster

The NFS volume is mounted at `/mnt5/noy/` on Geoffrey and `/storage/noy/` on RunAI
(same data). Standard eval datasets:

| Purpose | Path |
|---|---|
| Eval data (`--data_source`) | `/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav` |
| Canonical panel root (`--nova_data_dir`) | `/mnt5/noy/SpectralFM/fairseq/data/nova_data` |
| Labeled data (`--labeled_data_dir`) | `/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data` |

---

## The evaluation methods

Every method runs **standalone** (`--evals <name>`) and (except
`signal_reconstruction`) **per-checkpoint inside `checkpoint_comparison`** —
identical figures and CSVs either way.

### 1. `embedding_similarity` — same-stack query (k-NN retrieval)

For each held-out sample: find its top-k nearest neighbors by cosine similarity
in (a) raw input space and (b) embedding space (mean-pooled `last_hidden_state`),
and count how many neighbors come from the same stack (observation).

- **Metrics:** `embedding_stack_match_rate`, `input_stack_match_rate`,
  `match_score_avg` (0–100; >50 ⇒ embeddings preserve stack structure better than raw signals).
- **Figures:** cosine similarity maps (raw + softmax, input & embedding space),
  intra/inter-stack similarity distributions, match-count histograms,
  `similarity_comparison_{best,worst}_query*.png` — 3×(k+1) grids: query signal,
  its embedding-space neighbors, its input-space neighbors with similarity values.
- **CSV:** `match_df[_<ckpt>].csv` — per-query neighbors, similarities, stack
  matches, `match_diff`, `match_score`.

### 2. `signal_reconstruction` — true reconstruction through the pipeline

Reconstructs the raw signal from three pipeline stages and scores per-sample MSE
against the (optionally layer-normed) input. Takes its **own checkpoint args**,
independent of `--checkpoint_mode`, because the decoders are trained separately:

| Pathway | Tap point | Decoder |
|---|---|---|
| **FE** | post-LayerNorm FE output `[B, 47, 512]` | `MirrorDecoder` (ConvTranspose mirror of the FE) |
| **Projection** | `post_extract_proj` output `[B, 47, 768]` | `TransformerMirrorDecoder` |
| **Transformer** | encoder output `[B, 47, 768]` | `TransformerMirrorDecoder` |

Checkpoint options:

- `--recon_ckpt <3ae.pt>` — single 3AE checkpoint (keys `data2vec_audio` +
  `fe_mirror` / `proj_mirror` / `transformer_mirror`); runs **all** contained
  pathways. The fine-tuned backbone embedded in the file is used, including
  v1 single weight-normed positional conv handling.
- `--recon_fe_ckpt <fe.pt>` — standalone FE autoencoder
  (keys `encoder`/`layer_norm`/`decoder`); FE pathway only.
- `--recon_tr_ckpt <tr.pt>` — transformer-recon head
  (keys `transformer_mirror`/`backbone_ckpt`); backbone auto-loaded from the
  recorded `backbone_ckpt` path.

Normalization (`F.layer_norm` per sample) is read from the checkpoint's
`normalize` flag; override with `--recon_normalize true|false`.

- **Metrics:** `recon_{fe,proj,tr}_mse_mean` / `_median`.
- **Figures:** `recon_overlay.png` (target vs each pathway, per-row y-limits),
  `recon_mse_bars.png` (per-sample, log scale).
- **CSV:** `recon_df.csv` — per-sample `fe_mse` / `proj_mse` / `tr_mse`.

### 3. `noise_robustness` — embedding stability under input noise

Adds 6 noise types to each input (`gaussian_std`, `gaussian_mean`, `shot_low`,
`shot_high`, `gain_low`, `gain_high`) and measures cosine similarity between the
clean and noisy embeddings (higher = more robust).

- **Metrics:** mean cosine similarity per noise type.
- **Figures:** `noise_example_grid[_<ckpt>].png` — 2×3 grid over all noise types,
  clean vs noisy signal + embedding similarity per cell;
  `noisy_vs_clean_<type>*.png` — best/worst 3 samples per noise type;
  summary bar chart per noise type (across checkpoints in comparison mode).
- **CSV:** `noise_df[_<ckpt>].csv` — per-sample per-noise-type similarity.

### 4. `clustering` — stack recovery with KMeans

KMeans on embeddings with k = number of stacks, scored against true `stack_idx`.

- **Metrics:** ARI, NMI, V-measure, silhouette, same-stack KNN precision,
  retrieval mAP, inter/intra variance ratio, embedding–input distance alignment.
- **Figures:** `kmeans_clustered_then_tsne[_<ckpt>].png` and `..._umap...png` —
  PCA(50) → t-SNE/UMAP scatter colored by KMeans cluster.

### 5. `label_regression` — parameter_0 linear probe (multi-channel)

RidgeCV (5-fold) predicting `parameter_0` from raw inputs vs from embeddings,
on spectra from `--labeled_data_dir` (`labels.tsv` + wavs named
`dataset<D>_comp<C>_spec_<S>.wav`; components of a spectrum share one label).
Runs three configurations on the same spectra: 1-comp (raw 245 / emb 768),
2-comp (490 / 1536), 3-comp (735 / 2304) — raw channels concatenated,
per-component embeddings concatenated.

- **Metrics:** `label_reg_input_r2`, `label_reg_emb_r2`,
  `label_reg_improvement_r2` (= emb − input; positive ⇒ embeddings add
  label-relevant information) per config (suffix `''`/`_2c`/`_3c`),
  plus MSE/RMSE/MAE/pearson/spearman per probe.
- **Figures:** `label_reg_true_vs_pred[_<ckpt>].png` — true-vs-predicted scatter
  for both probes; `label_regression_comparison.png` — R² bars + ΔR² across
  checkpoints (comparison mode).

### 6. `structured_similarity` — canonical 100-sample panel

Deterministic (seed=42) panel spanning 4 datasets — samples 0–29
`single_channel_all` (3 stacks×10), 30–59 `multi_channel` (3 comps×10),
60–79 `sampled_data` (2×10), 80–99 `labeled_data` (2×10) — pushed through the
model with all 4 representation stages extracted in one pass.

- **Figures:** `struct_sim[_<ckpt>].png` — 2×2 cosine-similarity heatmaps:
  Input (245) / FE output (512) / Projection (768) / Embedding (768), with
  white block boundaries and dataset tick labels.
- Requires `--nova_data_dir`.

### 7. `checkpoint_comparison` — everything across multiple checkpoints

Runs methods 1, 3, 4, 5, 6 per checkpoint (with `--checkpoint_mode multiple`),
producing one summary row per checkpoint (`comparison_df` + scalar-metric grid
plot) and a per-checkpoint report section with all figures. Sub-evals toggle via
`EvalConfig` flags (`run_noise_in_comparison` etc.). Combine with
`signal_reconstruction` in one command for the complete suite.

---

## Checkpoint formats (auto-detected)

| Format | Detection keys | Loaded as |
|---|---|---|
| `hf` | — (`--checkpoint_mode hf`) | HF pretrained `facebook/data2vec-audio-base` |
| `fairseq` | `cfg` + `model` | keys remapped into the HF model (`_remap_fairseq_keys`) |
| `3ae` | `data2vec_audio` | embedded fine-tuned backbone (+ mirror heads for reconstruction) |
| `fe_recon` | `encoder`+`layer_norm`+`decoder` | FE + LN into the HF model (recon: + MirrorDecoder) |
| `tr_recon` | `transformer_mirror`+`backbone_ckpt` | backbone from recorded path (recon: + mirror head) |
| `state_dict` | anything else | direct `load_state_dict` |

## Checkpoint modes (`--checkpoint_mode`)

| Mode | `--checkpoint_path` | Description |
|---|---|---|
| `hf` | — | HuggingFace pretrained, no file needed |
| `file` | `.pt` file | single checkpoint, any format above |
| `dir` | directory | picks `checkpoint_best.pt` → `checkpoint_last.pt` → latest numbered |
| `multiple` | directory (or `--checkpoint_paths f1.pt f2.pt ...`) | all for `checkpoint_comparison` |

---

## Full-suite example (comparison + reconstruction, as run on Geoffrey)

```bash
cd /mnt5/noy/SpectralFM/code
CUDA_VISIBLE_DEVICES=4 /mnt5/noy/miniconda3/envs/spectralfm_env/bin/python3 -u -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode multiple \
  --checkpoint_paths \
    /mnt5/noy/SpectralFM/checkpoints/recon_runs_copied/tv_fe_short_3.pt \
    /mnt5/noy/SpectralFM/checkpoints/recon_runs_copied/3ae_norm_exp2_long.pt \
  --evals checkpoint_comparison signal_reconstruction \
  --recon_ckpt /mnt5/noy/SpectralFM/checkpoints/recon_runs_copied/tv_fe_short_3.pt \
  --n_holdout_stacks 20 --k 5 --batch_size 64 \
  --nova_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data \
  --labeled_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data \
  --output_dir /mnt5/noy/SpectralFM/code/eval_outputs
```

Runtime: ~10 min per checkpoint on one RTX 2080 Ti (label regression on 2,000
samples is the slowest stage) + ~1 min for reconstruction.

## Output directory contents

| File | Content |
|---|---|
| `eval_report.html` / `eval_report.md` | self-contained report / markdown version |
| `run_info.md` | what was analyzed (data, checkpoints, evals, device) |
| `checkpoint_comparison.png` + `*_comparison_df.csv` | scalar metric grid + table |
| `recon_overlay.png`, `recon_mse_bars.png`, `recon_df.csv` | signal reconstruction |
| `noise_example_grid*.png`, `noisy_vs_clean_*.png`, `noise_df*.csv` | noise robustness |
| `similarity_comparison_*.png`, `match_df*.csv`, `cosine_*.png`, `emb_*.png` | same-stack query |
| `kmeans_clustered_then_*.png` | clustering scatters |
| `label_reg_true_vs_pred*.png`, `label_regression_comparison.png` | label regression |
| `struct_sim*.png` | structured similarity panels |

## Key constraints

- All eval code must stay importable **without fairseq**.
- SpectralFM wavs are 245 frames; embedding evals layer-norm inputs to match
  data2vec training, but reconstruction receives **raw** signals (normalization
  applied only if the recon checkpoint was trained with it).
- The eval `data_loader` assigns `stack_idx = file_index // 10` (files sorted by
  name) — stacks are consecutive groups of 10.
