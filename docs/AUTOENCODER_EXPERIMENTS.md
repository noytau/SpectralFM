# FE Autoencoder Experiments

Summary of the convolutional autoencoder experiments for evaluating and improving the Feature Extractor (FE) reconstruction capability.

---

## 1. Motivation

The data2vec audio pipeline uses a CNN-based Feature Extractor (FE) to compress 245-bin spectrograms into latent representations. We want to understand how well the FE can reconstruct the original signal, and whether reconstruction quality generalizes across different datasets.

---

## 2. Architecture

### Encoder (FE — `ConvFeatureExtractionModel`)

| Layer | Operation | Output |
|-------|-----------|--------|
| 1 | Conv1d(1→512, k=5, s=5) + LayerNorm + GELU | (B, 512, 49) |
| 2 | Conv1d(512→512, k=3, s=1) + LayerNorm + GELU | (B, 512, 47) |
| 3 | Conv1d(512→512, k=3, s=1) + LayerNorm + GELU | (B, 512, 45) |
| 4 | Conv1d(512→512, k=3, s=1) + LayerNorm + GELU | (B, 512, 43) |
| 5 | Conv1d(512→512, k=3, s=1) + LayerNorm + GELU | (B, 512, 41)* |
| LN | LayerNorm(512) | (B, 512, 47) |

*Actual FE config is `[(512,3,1)]×4 + [(512,5,5)]` producing output dim 47.

### Decoder (MirrorDecoder — `ConvTranspose1d`)

| Layer | Operation | Output |
|-------|-----------|--------|
| 1 | ConvTranspose1d(512→512, k=5, s=5) + LayerNorm + GELU | (B, 512, 237) |
| 2 | ConvTranspose1d(512→512, k=3, s=1) + LayerNorm + GELU | (B, 512, 239) |
| 3 | ConvTranspose1d(512→512, k=3, s=1) + LayerNorm + GELU | (B, 512, 241) |
| 4 | ConvTranspose1d(512→512, k=3, s=1) + LayerNorm + GELU | (B, 512, 243) |
| 5 | ConvTranspose1d(512→1, k=3, s=1) | (B, 1, 245) |

**Parameters:** Encoder 3.68M, Decoder 3.68M, Total 7.35M

---

## 3. Training Setup

- **Loss:** MSE (L2)
- **Optimizer:** Adam
- **Batch size:** 64
- **LR schedule:** Cosine with linear warmup (15 steps → peak 1e-4 → decay to 0)
- **Initialization:** Random (no pretrained FE weights)
- **Logging:** Weights & Biases (`spectralfm-autoencoder` project)
- **Platform:** RunAI (A5000/A6000 GPUs)

### Data Loading

Two data loading strategies were implemented:
1. **Preload (`load_data`):** Loads all WAV files into memory upfront. Works for ≤10K samples, bottleneck for larger datasets.
2. **Lazy (`LazyWavDataset` + `DataLoader`):** Reads WAV files on-the-fly from a fairseq-style manifest TSV with `num_workers=4`. Required for 100K+ samples.

---

## 4. Experiments

### Phase 1: Hyperparameter sweep (lr=1e-4, RunAI)

| n_samples | steps | Tag | Final L2 |
|-----------|-------|-----|----------|
| 100 | 1,000 | `lr0.0001_n100_s1000` | ~0.03 |
| 100 | 10,000 | `lr0.0001_n100_s10000` | ~0.01 |
| 1,000 | 1,000 | `lr0.0001_n1000_s1000` | ~0.05 |
| 1,000 | 10,000 | `lr0.0001_n1000_s10000` | ~0.005 |
| 10,000 | 1,000 | `lr0.0001_n10000_s1000` | ~0.1 |
| 10,000 | 10,000 | `lr0.0001_n10000_s10000` | ~0.003 |

### Phase 2: Cosine LR + longer training

| n_samples | steps | warmup | Tag | Notes |
|-----------|-------|--------|-----|-------|
| 1,000 | 10,000 | 15 | `lr0.0001_n1000_s10000_w15` | Cosine schedule |
| 1,000 | 20,000 | 15 | `lr0.0001_n1000_s20000_w15` | Extended |
| 10,000 | 10,000 | 15 | `lr0.0001_n10000_s10000_w15` | Cosine schedule |
| 10,000 | 50,000 | 15 | `lr0.0001_n10000_s50000_w15` | **Best single_channel result** |

### Phase 3: Scaling to more data

| n_samples | steps | Source dataset | Tag |
|-----------|-------|----------------|-----|
| 100,000 | 50,000 | single_channel_all (lazy loader) | `lr0.0001_n100000_s50000_w15` |
| 950,000 | 50,000 | single_channel_one (lazy loader) | `lr0.0001_n950000_s50000_w15` |

---

## 5. Cross-Dataset Reconstruction Results

Evaluated on 100 samples per dataset. All checkpoints trained on single-channel data.

### Trained on 10K samples (single_channel_all), 50K steps

| Dataset | Mean L2 | Std L2 | Median | Min | Max |
|---------|---------|--------|--------|-----|-----|
| single_channel_10k (in-distribution) | 6.6E-04 | 5.6E-04 | 4.5E-04 | 8.0E-05 | 2.9E-03 |
| multi_channel | 4.0E-02 | 3.5E-02 | 2.3E-02 | 5.1E-03 | 1.5E-01 |
| labeled_data | 3.0E-02 | 1.1E-02 | 3.0E-02 | 1.0E-02 | 7.6E-02 |

### Trained on 100K samples (single_channel_all), 50K steps

| Dataset | Mean L2 | Std L2 | Median | Min | Max |
|---------|---------|--------|--------|-----|-----|
| single_channel_10k (in-distribution) | 4.6E-04 | 4.6E-04 | 3.2E-04 | 5.9E-05 | 3.2E-03 |
| multi_channel | 5.3E-02 | 5.0E-02 | 3.0E-02 | 4.7E-03 | 1.8E-01 |
| labeled_data | 3.0E-02 | 1.3E-02 | 2.9E-02 | 1.1E-02 | 8.9E-02 |

### Trained on 950K samples (single_channel_one), 50K steps

| Dataset | Mean L2 | Std L2 | Median | Min | Max |
|---------|---------|--------|--------|-----|-----|
| single_channel_10k (in-distribution) | 4.4E-04 | 4.6E-04 | 3.0E-04 | 4.7E-05 | 3.1E-03 |
| multi_channel | 4.9E-02 | 4.3E-02 | 3.0E-02 | 5.2E-03 | 1.8E-01 |
| labeled_data | 3.1E-02 | 1.3E-02 | 2.9E-02 | 9.4E-03 | 8.1E-02 |

### Key Observations

- **In-distribution (single_channel)** improves with more data: 6.6E-04 → 4.6E-04 → 4.4E-04 (33% reduction 10K→950K)
- **multi_channel** and **labeled_data** remain 50-60x worse — more single-channel data does not help cross-domain generalization
- **multi_channel** has highest variance (std ~0.04), some samples reconstruct reasonably (5E-03) while others fail (0.15+)
- **labeled_data** is more consistent (std ~0.01) but uniformly shifted, suggesting a systematic distribution difference

---

## 6. Generated Artifacts

All outputs are in `autoencoder_experiments/runai/`:

### Checkpoints (`.pt`)
Each checkpoint contains: encoder state, layer_norm state, decoder state, training loss curve, LR schedule, and training config metadata.

### Plots
- **`analyze_*.png`** — Training loss curve + encoder heatmap + per-sample GT vs pred overlays + residuals
- **`bestworst_<dataset>_*.png`** — 5 best and 5 worst reconstruction samples per dataset (GT vs pred + residual bars)
- **`stats_*.png`** — Per-dataset L2 histograms + cross-dataset mean L2 bar chart comparison
- **`diagram_training_setup.png`** — Training pipeline architecture diagram
- **`diagram_signal_flow.png`** — Signal dimension flow through encoder and decoder layers

---

## 7. Code

### Main script: `code/train_reconstruction.py`

Modes:
- `--mode train` — Train FE + MirrorDecoder autoencoder (future: FE + Transformer)
- `--mode analyze` — Load checkpoint, evaluate on multiple datasets, generate stats + best/worst plots
- `--mode interp` — Inference with pretrained interpolation decoder (requires full fairseq model)

Key arguments:
```
--ckpt              Path to data2vec checkpoint for FE init (or "none" for random)
--ckpt_ae           Path to autoencoder checkpoint (for analyze mode)
--data_dir          Directory with .wav files
--manifest          Fairseq-style manifest TSV (for lazy loading)
--datasets          Comma-separated dataset names for analyze (e.g. "single_channel_10k,multi_channel,labeled_data")
--n_stat            Number of samples for statistical evaluation (default: 100)
--lr                Learning rate (default: 1e-3)
--warmup            Warmup steps for cosine schedule (0 = constant LR)
--n_samples         Number of training samples
--steps             Training steps
--k                 Number of samples to visualize
--wandb_project     W&B project name for logging
```

### Dataset paths
Resolved automatically via `_resolve_nova_root()`:
- Local (Geoffrey): `/mnt5/noy/SpectralFM/fairseq/data/nova_data/<dataset>/wavs`
- RunAI: `/storage/noy/SpectralFM/fairseq/data/nova_data/<dataset>/wavs`

Built-in dataset aliases: `single_channel_10k`, `single_channel_1k`, `single_channel_100`, `multi_channel`, `labeled_data`

---

## 8. W&B Tracking

All RunAI training runs are logged to Weights & Biases:
- **Project:** `spectralfm-autoencoder`
- **Metrics:** `train/loss`, `train/lr`, `train/epoch`, `train/grad_norm` (every 100 steps)
- **Run names** match checkpoint tags (e.g. `lr0.0001_n10000_s50000_w15`)

---

## 9. Prior Work: Fairseq recon_only Decoder Experiments

Before the standalone autoencoder, reconstruction decoders were tested within the fairseq pipeline (`recon_only=True` in `data2vec_audio.py`). These used a frozen pretrained FE and trained only the decoder head:

| Decoder type | Description | 20K step L1 |
|---|---|---|
| `linear` (default) | Mean-pool → Linear(512→245) | baseline |
| `mlp` | Mean-pool → MLP(512→1024→245) | similar to linear |
| `interp` | Skip mean-pool, interpolate 47→245, Linear(512→1) | best fairseq variant |
| `conv1d` | Skip mean-pool, Conv1d stack | failed (instability) |
| `flat` | Flatten 512×47 → Linear(→245) | poor |

The mean-pool bottleneck in `linear`/`mlp` decoders motivated the standalone autoencoder approach (this document), which avoids mean-pooling entirely by using a symmetric `ConvTranspose1d` mirror decoder.

---

*Last updated: 2026-05-11*
