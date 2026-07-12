# SpectralFM Eval Package — Overview

Lightweight evaluation package under `code/eval/`. **No fairseq required.**
See `README.md` for installation and the Geoffrey run guide; this file documents
the architecture and every evaluation in code-level detail.

---

## Architecture

```
code/eval/
├── checkpoint_loader.py          # Block 1: load any checkpoint (HF / fairseq / 3AE / fe-recon / tr-recon)
├── model.py                      # Block 2: model components — no fairseq
├── data_loader.py                # Block 3: load wav / CSV / labeled data, masking, dataloaders
├── metrics.py                    # Block 4: pure metric functions (numpy/scipy/sklearn)
├── evaluations/
│   ├── embedding_similarity.py   # Eval 1: same-stack query — do embeddings group stacks?
│   ├── signal_reconstruction.py  # Eval 2: true reconstruction — FE / projection / transformer
│   ├── noise_robustness.py       # Eval 3: embedding stability under 6 noise types
│   ├── clustering.py             # Eval 4: KMeans vs stack labels (ARI/NMI/silhouette/…)
│   ├── label_regression.py       # Eval 5: parameter_0 ridge probe (input vs embedding)
│   ├── structured_similarity.py  # Eval 6: canonical 100-sample panel, 4 pipeline stages
│   └── checkpoint_comparison.py  # Eval 7: run Evals 1,3,4,5,6 across multiple checkpoints
├── runner.py                     # EvalRunner + EvalConfig + CLI entry point
├── report.py                     # HTML + markdown + PNG + CSV report generator
├── requirements.txt              # minimal install (see table below)
└── README.md                     # installation + how to run on Geoffrey
```

---

## Quick Start

```bash
pip install -r code/eval/requirements.txt

# Single checkpoint — the per-model evals
python -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode file --checkpoint_path /path/to/ckpt.pt \
  --evals embedding_similarity noise_robustness clustering label_regression structured_similarity \
  --nova_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data \
  --labeled_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data \
  --output_dir /mnt5/noy/SpectralFM/code/eval_outputs

# Compare multiple checkpoints + signal reconstruction, one command
python -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode multiple --checkpoint_paths ckptA.pt ckptB.pt \
  --evals checkpoint_comparison signal_reconstruction \
  --recon_ckpt /path/to/3ae_ckpt.pt \
  --nova_data_dir ... --labeled_data_dir ... \
  --output_dir /mnt5/noy/SpectralFM/code/eval_outputs
```

Python API:

```python
from eval.runner import EvalRunner, EvalConfig

runner = EvalRunner(EvalConfig(
    data_source="/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav",
    checkpoint_mode="multiple",
    checkpoint_paths=["/path/a.pt", "/path/b.pt"],
    evals=["checkpoint_comparison", "signal_reconstruction"],
    recon_ckpt="/path/to/3ae_ckpt.pt",
    nova_data_dir="/mnt5/noy/SpectralFM/fairseq/data/nova_data",
    output_dir="/mnt5/noy/SpectralFM/code/eval_outputs",
))
results = runner.run()
runner.report(results)
```

---

## Block 1 — Checkpoint Loader (`checkpoint_loader.py`)

Four loading modes, all using raw `torch.load` or HuggingFace — zero fairseq dependency.

| Mode | When to use |
|------|-------------|
| `hf` | Pull `facebook/data2vec-audio-base` from HuggingFace (no local file) |
| `file` | Single `.pt` file — format auto-detected by key inspection (below) |
| `dir` | Directory scan: prefers `checkpoint_best.pt` → `checkpoint_last.pt` → highest numbered |
| `multiple` | Directory glob or explicit list → `[(label, model, path), ...]` for comparison eval |

Format auto-detection (`_detect_checkpoint_type`):

| Format | Detection keys | Loaded as |
|--------|----------------|-----------|
| `fairseq` | `cfg` + `model` | keys remapped to HF names (`_remap_fairseq_keys`: `self_attn→attention`, `fc1→feed_forward.intermediate_dense`, `post_extract_proj→feature_projection.projection`, …) |
| `3ae` | `data2vec_audio` | **embedded fine-tuned backbone** via `load_3ae_backbone` — handles the v1 single weight-normed positional conv (`pos_conv.0.weight_g/v` → `SingleLayerPosConv`) and drops aux monitoring decoders |
| `fe_recon` | `encoder`+`layer_norm`+`decoder` | FE conv stack + LayerNorm into the HF model |
| `tr_recon` | `transformer_mirror`+`backbone_ckpt` | backbone loaded from the recorded `backbone_ckpt` path |
| `state_dict` | anything else | direct `load_state_dict` |

---

## Block 2 — Model (`model.py`)

```python
class FairseqConvFeatureExtractor(nn.Module):
    # conv+LayerNorm stack matching fairseq checkpoints: [(512,3,1)]×4 + [(512,5,5)], 245→47

class CustomFeatureExtractor(nn.Module):
    # archs: conv1d | 2_conv1d_relu | 2_conv1d | 3_conv1d_relu | 3_conv1d

class MirrorDecoder(nn.Module):
    # ConvTranspose1d mirror of the FE: [B, 512, 47] → [B, 1, 245]
    # (FE reconstruction head; ported from train_reconstruction.py)

class TransformerMirrorDecoder(nn.Module):
    # [B, T, C] → stem 1×1 conv C→512 → optional LayerNorm → MirrorDecoder → [B, 245]
    # (projection + transformer reconstruction heads)

class SingleLayerPosConv(nn.Module):
    # fairseq wav2vec2-style positional conv (single weight-normed Conv1d + SamePad + GELU)
    # drop-in replacement for HF's 5-layer pos_conv_embed when a checkpoint uses
    # conv_pos=128 / conv_pos_groups=16 (3AE backbones)

def build_model(hf_model, arch="conv1d"):
    hf_model.feature_extractor = CustomFeatureExtractor(arch)   # or Fairseq… when loading ckpts
    return hf_model
```

---

## Block 3 — Data Loader (`data_loader.py`)

```python
load_data(source)                    # auto: .wav dir → DataFrame, or CSV → DataFrame
load_labeled_data(dir, max_samples)  # labels.tsv + wavs → (inputs [N,245], labels [N])
split_stack_holdout(df, n=5)         # hold out first N stacks for eval
split_partial_stack(df, ratio=0.3)   # hold out 30% of each stack
build_dataloader(df, mask_ratio, masking_type, batch_size)
    # applies per-sample F.layer_norm (data2vec preprocessing) IN PLACE on df['data']
    # → the runner keeps a raw copy (df_raw) for signal_reconstruction
```

## Block 4 — Metrics (`metrics.py`)

Pure functions (numpy/scipy/sklearn), ported from the legacy `eval_metrics.py`:
`knn_retrieval`, `compute_component_clustering_metrics` (ARI/NMI/V-measure/silhouette/
KNN-precision/mAP/variance-ratio), `compute_linear_probing_metrics` (RidgeCV probe),
`compute_extended_signal_completion`.

---

## Eval 1 — Embedding Similarity / same-stack query (`evaluations/embedding_similarity.py`)

**Question:** Does the embedding space group same-stack samples better than raw input space?

For each sample: find top-k cosine neighbors in input space and embedding space; count
same-stack neighbors. Match score > 50 means the model beats the raw-input baseline.

```python
match_score = ((len(emb_matches) - len(inp_matches) + k) / (2 * k)) * 100  # 0–100
```

Returns: `embedding_stack_match_rate`, `input_stack_match_rate`, `match_score_avg`,
`results_df` (per-query neighbors + similarities → `match_df*.csv`), `embeddings`,
`inputs`, `similarity_distributions` (intra/inter-stack, full sim matrices).
Figures: cosine maps, distribution histograms, best/worst query neighbor grids.

---

## Eval 2 — Signal Reconstruction (`evaluations/signal_reconstruction.py`)

**Question:** Can each pipeline stage be decoded back into the original signal?

True reconstruction through the full pipeline — three pathways, each with its own
trained decoder. Ported from the `reconstruction-loss-experiments` branch
(`train_reconstruction.py` / `compare_fe_vs_trans_recon.py`).

| Pathway | Tap point (training hook) | Decoder | Ckpt key |
|---------|---------------------------|---------|----------|
| **FE** | post-LN FE output `[B,47,512]` (`backbone.layer_norm` hook) | `MirrorDecoder` | `fe_mirror` |
| **Projection** | `post_extract_proj` output `[B,47,768]` | `TransformerMirrorDecoder` | `proj_mirror` |
| **Transformer** | encoder output `[B,47,768]` | `TransformerMirrorDecoder` | `transformer_mirror` |

Checkpoint options (independent of `--checkpoint_mode` — reconstruction decoders are
trained and saved separately from the eval backbone):

```bash
--recon_ckpt    3ae.pt       # single 3AE file: embedded backbone + all heads → ALL pathways
--recon_fe_ckpt fe.pt        # standalone FE AE (encoder/layer_norm/decoder) → FE pathway
--recon_tr_ckpt tr.pt        # transformer head (transformer_mirror/backbone_ckpt) → TR pathway
--recon_normalize true|false # override the ckpt's recorded per-sample layer_norm flag
```

The 3AE path runs one backbone forward per batch with hooks at the three tap points —
exactly mirroring training. Score: per-sample `MSE(recon, target)` where target is the
(optionally normalized) input.

Returns: `recon_{fe,proj,tr}_mse_mean/median`, `results_df` (→ `recon_df.csv`),
example `panel`. Figures: `recon_overlay.png` (target vs each pathway), `recon_mse_bars.png`.

---

## Eval 3 — Noise Robustness (`evaluations/noise_robustness.py`)

**Question:** How stable are embeddings when noise is added to the input?

Score = cosine similarity between clean and noisy embeddings. Higher = more robust.

| Noise type | Formula |
|-----------|---------|
| `gaussian_std` | `x + N(0, 0.01)` |
| `gaussian_mean` | `x + N(2, 0.001)` |
| `shot_low` | `Poisson(x * 0.1) / 0.1` |
| `shot_high` | `Poisson(x * 0.05) / 0.05` |
| `gain_low` | `x * N(1, 0.05)` |
| `gain_high` | `x * N(1, 0.1)` |

Returns: `summary = {noise_type: mean_cos_sim}`, `results_df` (→ `noise_df*.csv`),
`clean_data`, `noisy_data`. Figures: **Noise Example Grid** (2×3 over all types,
clean vs noisy + emb sim per cell), best/worst clean-vs-noisy overlays, summary bars.

---

## Eval 4 — Clustering (`evaluations/clustering.py`)

**Question:** Does KMeans on embeddings recover the stack structure?

KMeans with k = n_stacks, scored against true `stack_idx` via
`metrics.compute_component_clustering_metrics`.

Returns: `comp_cluster_ari`, `comp_cluster_nmi`, `comp_cluster_vmeasure`,
`comp_cluster_silhouette`, `comp_cluster_knn_precision`, `comp_cluster_retrieval_map`,
`comp_cluster_variance_ratio`, `comp_cluster_emb_input_align`, plus `pred_labels`.
Figures: PCA(50) → t-SNE and UMAP scatters colored by KMeans cluster.

---

## Eval 5 — Label Regression (`evaluations/label_regression.py`)

**Question:** Do embeddings encode `parameter_0` better than the raw signal?

RidgeCV (5-fold cross-val) probe on inputs vs embeddings, using
`--labeled_data_dir` (`labels.tsv` + wavs, default 2,000 spectra).

**Multi-channel:** wavs follow `dataset<D>_comp<C>_spec_<S>.wav`; components of the
same spectrum share one label. The eval runs three configurations on the SAME spectra
(only those having all 3 components), matching `label_reg_evaluation.py`:

| Config | Raw input | Embedding |
|--------|-----------|-----------|
| 1-comp (C0) | 245 | 768 |
| 2-comp (C0,C1) | 490 | 1536 |
| 3-comp (C0,C1,C2) | 735 | 2304 |

Raw channels are concatenated; embeddings are extracted per component then concatenated.

Returns: `label_reg_input_*` / `label_reg_emb_*` (r2/mse/rmse/mae/pearson/spearman)
per config (suffix `''` / `_2c` / `_3c`), `label_reg_improvement_r2*` = emb R² − input R²,
cross-val predictions (1-comp).
Figures: true-vs-predicted scatter per probe; R² + ΔR² bars across checkpoints.

---

## Eval 6 — Structured Similarity (`evaluations/structured_similarity.py`)

**Question:** How does similarity structure evolve through the pipeline stages?

Deterministic 100-sample panel (seed=42): 0–29 `single_channel_all` (3 stacks×10),
30–59 `multi_channel` (3 comps×10), 60–79 `sampled_data` (2×10), 80–99 `labeled_data`
(2×10). One forward pass extracts 4 representations: input (245) → FE output (512) →
projection (768) → embedding (768).

Returns: the 4 representations + 4 cosine similarity matrices + group labels.
Figure: 2×2 heatmap panel with white block boundaries and dataset tick labels.
Requires `--nova_data_dir`.

---

## Eval 7 — Checkpoint Comparison (`evaluations/checkpoint_comparison.py`)

Wraps Evals 1, 3, 4, 5, 6 and runs them for every checkpoint from
`CheckpointLoader.load_multiple()` (directory glob or `--checkpoint_paths` list).
Produces `comparison_df` — one row per checkpoint with all scalar metrics — plus a
per-checkpoint report section containing every figure. Sub-evals toggle via
`EvalConfig.run_{noise,clustering,label_regression}_in_comparison`.
`signal_reconstruction` runs alongside (once per run), not per-checkpoint.

---

## Block 5 — Unified Runner (`runner.py`)

```python
@dataclass
class EvalConfig:
    data_source:        str            # .wav dir or CSV path
    checkpoint_mode:    str            # hf | file | dir | multiple
    checkpoint_path:    str | None
    checkpoint_paths:   List[str]      # explicit files for 'multiple'
    evals:              List[str]      # embedding_similarity, signal_reconstruction,
                                       # noise_robustness, clustering, label_regression,
                                       # structured_similarity, checkpoint_comparison
    split_mode:         str            # stack_holdout | partial_stack | none
    n_holdout_stacks:   int   = 5
    k:                  int   = 5
    batch_size:         int   = 16
    device:             str   = "auto"
    nova_data_dir:      str | None     # structured similarity panel root
    labeled_data_dir:   str | None     # label regression (defaults to <nova_data_dir>/labeled_data)
    recon_ckpt:         str | None     # 3AE checkpoint → all reconstruction pathways
    recon_fe_ckpt:      str | None     # standalone FE AE checkpoint
    recon_tr_ckpt:      str | None     # transformer-recon checkpoint
    recon_normalize:    bool | None    # None → use ckpt's recorded flag
    output_dir:         str   = "eval_outputs"
```

Every method runs both standalone and inside `checkpoint_comparison` — same figures
and CSVs. `python -m eval.runner --help` shows all args + usage-mode examples.

---

## Block 6 — Report Generator (`report.py`)

Each run writes a timestamped directory `output_dir/<YYYY-MM-DD_HH-MM-SS>/`:

```
eval_report.html                     ← self-contained (all figures base64-embedded)
eval_report.md / run_info.md         ← markdown report / what was analyzed
checkpoint_comparison.png            ← scalar metric grid across checkpoints
recon_overlay.png / recon_mse_bars.png / recon_df.csv          ← Eval 2
noise_example_grid*.png / noisy_vs_clean_*.png / noise_df*.csv ← Eval 3
kmeans_clustered_then_{tsne,umap}*.png                         ← Eval 4
label_reg_true_vs_pred*.png / label_regression_comparison.png  ← Eval 5
struct_sim*.png                                                ← Eval 6
similarity_comparison_{best,worst}_query*.png / match_df*.csv  ← Eval 1
cosine_map_*.png / cosine_sim_*.png / emb_*.png                ← Eval 1
*_comparison_df.csv                                            ← Eval 7
```

---

## Requirements (`requirements.txt`)

```
torch>=2.0.0            torchaudio>=2.0.0       transformers>=4.30.0
numpy>=1.24.0           scipy>=1.10.0           pandas>=2.0.0
scikit-learn>=1.2.0     matplotlib>=3.7.0       seaborn>=0.12.0
soundfile>=0.12.0       tabulate>=0.9.0         omegaconf>=2.0.0
umap-learn>=0.5.0   # optional — UMAP scatter (falls back to t-SNE)
```

| Package | Used by |
|---------|---------|
| `torch` / `torchaudio` | model inference, wav loading, masking, MSE |
| `transformers` | `Data2VecAudioModel.from_pretrained` |
| `numpy` / `scipy` / `pandas` | metrics, stats, results DataFrames |
| `scikit-learn` | KMeans, RidgeCV, KNN, cosine similarity |
| `matplotlib` / `seaborn` | all figures |
| `soundfile` | wav backend |
| `tabulate` | DataFrame → markdown tables |
| `omegaconf` | unpickle fairseq-format checkpoint configs **without fairseq** |
| `umap-learn` | optional clustering scatter projection |

**Not needed:** `fairseq`, `hydra-core`, `apex`, `submitit`.

On Geoffrey the `spectralfm_env` conda environment already has all of these.

---

## Data Setup

Manifests live at `/mnt5/noy/SpectralFM/fairseq/data/nova_data/<subset>/` (Geoffrey path).
TSV roots use `/storage/noy/...` — the same NFS as RunAI mounts; eval code remaps
`/storage/` → `/mnt5/` automatically.

| Subset | Train | Valid | Wav files |
|--------|-------|-------|-----------|
| `single_channel_100` | 90 | 10 | 100 |
| `single_channel_1k` | 950 | 50 | 1,000 |
| `single_channel_10k` | 10,611 | 500 | 11,111 |
| `single_channel_one` | 999,000 | 1,000 | 1,000,000 |
| `single_channel_5m` | 4,995,000 | 5,000 | 5,000,000 |
| `single_channel_all` | 9,099,930 | 10,000 | 9,109,930 |

To regenerate manifests: `python fairseq/create_manifests.py --help`
To submit RunAI sweep: `bash sweep_dataset.sh`

Training config: `fairseq/examples/data2vec/config/audio/pretraining/spectralfm_base.yaml`
Key fix vs base librispeech config: `min_sample_size: 1` (base uses 32000, which silently
drops all 245-frame SpectralFM wavs).
