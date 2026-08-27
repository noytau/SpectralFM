# SpectralFM Eval Package — Overview

The current, actively-maintained evaluation package for SpectralFM models, under `code/eval/`.
**No fairseq required.** This is the single doc for evaluation — installation, how to run it,
every method in detail, checkpoint formats, and output structure.

(There's an older, fairseq-dependent eval system — `code/evaluation_runner.py` and friends —
that predates this package. It's not documented here; see `TASKS.md`'s cleanup task for its
status. Everything below is about `code/eval/` only.)

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
├── requirements.txt               # minimal install (see table below)
└── README.md                     # short pointer to this file
```

---

## Installation

Requires Python ≥ 3.10. No fairseq, no conda dependency — a plain venv works:

```bash
git clone https://github.com/noytau/SpectralFM.git && cd SpectralFM
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/eval/requirements.txt
```

First run downloads `facebook/data2vec-audio-base` (~360 MB) from HuggingFace;
set `HF_HOME` if you need a custom cache location.

On **Geoffrey**, the `spectralfm_env` conda environment already has everything installed
(torch 2.8, transformers 4.57) — no separate install needed there.

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

### Running on Geoffrey — a real worked example

```bash
ssh Geoffry

# Pick a free GPU first (GPU 3 is dead — currently use 4):
nvidia-smi

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

Outputs land in `/mnt5/noy/SpectralFM/code/eval_outputs/<YYYY-MM-DD_HH-MM-SS>/`.
Open `eval_report.html` in a browser — it is fully self-contained (figures base64-embedded).
For long runs, wrap with `nohup ... > /tmp/eval_run.log 2>&1 &` and `tail -f` the log.

**Runtime:** ~10 min per checkpoint on one RTX 2080 Ti (label regression on 2,000 samples
is the slowest stage) + ~1 min for reconstruction.

### Data paths on the cluster

The NFS volume is mounted at `/mnt5/noy/` on Geoffrey and `/storage/noy/` on RunAI (same
data) — eval code remaps `/storage/` → `/mnt5/` automatically.

| Purpose | Path |
|---|---|
| Eval data (`--data_source`) | `/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav` |
| Canonical panel root (`--nova_data_dir`) | `/mnt5/noy/SpectralFM/fairseq/data/nova_data` |
| Labeled data (`--labeled_data_dir`) | `/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data` |

---

## Block 1 — Checkpoint Loader (`checkpoint_loader.py`)

Four loading modes, all using raw `torch.load` or HuggingFace — zero fairseq dependency.

| Mode | `--checkpoint_path` | When to use |
|------|---|-------------|
| `hf` | — | Pull `facebook/data2vec-audio-base` from HuggingFace (no local file) |
| `file` | `.pt` file | Single checkpoint — format auto-detected by key inspection (below) |
| `dir` | directory | Picks `checkpoint_best.pt` → `checkpoint_last.pt` → highest numbered |
| `multiple` | directory (or `--checkpoint_paths f1.pt f2.pt ...`) | All checkpoints for `checkpoint_comparison` |

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

Pure functions (numpy/scipy/sklearn): `knn_retrieval`, `compute_component_clustering_metrics`
(ARI/NMI/V-measure/silhouette/KNN-precision/mAP/variance-ratio), `compute_linear_probing_metrics`
(RidgeCV probe), `compute_extended_signal_completion`.

---

## The 7 evaluation methods

Every method runs **standalone** (`--evals <name>`) and (except `signal_reconstruction`)
**per-checkpoint inside `checkpoint_comparison`** — identical figures and CSVs either way.

### 1. `embedding_similarity` — same-stack query (k-NN retrieval)

**Question:** Does the embedding space group same-stack samples better than raw input space?

For each held-out sample: find its top-k nearest neighbors by cosine similarity in (a) raw
input space and (b) embedding space (mean-pooled `last_hidden_state`), and count how many
neighbors come from the same stack (observation).

```python
match_score = ((len(emb_matches) - len(inp_matches) + k) / (2 * k)) * 100  # 0–100
```

- **Metrics:** `embedding_stack_match_rate`, `input_stack_match_rate`, `match_score_avg`
  (0–100; >50 ⇒ embeddings preserve stack structure better than raw signals).
- **Figures:** cosine similarity maps (raw + softmax, input & embedding space), intra/inter-stack
  similarity distributions, match-count histograms, `similarity_comparison_{best,worst}_query*.png`
  — 3×(k+1) grids: query signal, its embedding-space neighbors, its input-space neighbors with
  similarity values.
- **CSV:** `match_df[_<ckpt>].csv` — per-query neighbors, similarities, stack matches,
  `match_diff`, `match_score`.

### 2. `signal_reconstruction` — true reconstruction through the pipeline

**Question:** Can each pipeline stage be decoded back into the original signal?

Reconstructs the raw signal from three pipeline stages and scores per-sample MSE against the
(optionally layer-normed) input. Takes its **own checkpoint args**, independent of
`--checkpoint_mode`, because the decoders are trained separately from the eval backbone:

| Pathway | Tap point (training hook) | Decoder | Ckpt key |
|---------|---------------------------|---------|----------|
| **FE** | post-LN FE output `[B,47,512]` (`backbone.layer_norm` hook) | `MirrorDecoder` | `fe_mirror` |
| **Projection** | `post_extract_proj` output `[B,47,768]` | `TransformerMirrorDecoder` | `proj_mirror` |
| **Transformer** | encoder output `[B,47,768]` | `TransformerMirrorDecoder` | `transformer_mirror` |

Checkpoint options:

```bash
--recon_ckpt    3ae.pt       # single 3AE file: embedded backbone + all heads → ALL pathways
--recon_fe_ckpt fe.pt        # standalone FE AE (encoder/layer_norm/decoder) → FE pathway
--recon_tr_ckpt tr.pt        # transformer head (transformer_mirror/backbone_ckpt) → TR pathway
--recon_normalize true|false # override the ckpt's recorded per-sample layer_norm flag
```

The 3AE path runs one backbone forward per batch with hooks at the three tap points — exactly
mirroring training. Score: per-sample `MSE(recon, target)` where target is the (optionally
normalized) input.

#### Dataset-level metrics

Per-sample MSE alone cannot separate a working reconstruction from a decoder that emits each
sample's own mean value as a flat line — that degenerate model posts a respectable MSE while
carrying no information. Three levels are computed, all against the same target the head was
trained on (post `F.layer_norm` when the checkpoint recorded `normalize`):

**L1, per component** — one row per sample in `recon_df.csv`:

| Column | Meaning | Catches |
|---|---|---|
| `{h}_mse`, `{h}_mae` | error magnitude | baseline |
| `{h}_r2` | `1 − MSE / var(target)` | skill vs the mean-predictor baseline; `≤ 0` = degenerate |
| `{h}_pearson` | `corr(target, pred)` | shape agreement, scale-invariant |
| `{h}_amp_ratio` | `std(pred) / std(target)` | dynamic-range collapse |
| `{h}_peak_err` | signed error at the target's argmax | spectral-line fidelity |

Plus per-sample signal descriptors used as stratification axes (`contrast`, `peak_count`,
`peak_prominence`, `centroid`, `peak_position`; see `signal_features.py`) and, on
multi-component datasets, the parsed `dataset_id` / `comp` / `spec` / `n_comps`.

**L2, per spectrum** — multi-component datasets only, `spectrum_df.csv`. Every wav in
`multi_channel`, `sampled_data` and `labeled_data` is a single *component*; the physical sample
is every wav sharing a `spec` index. Collapsing a spectrum's components into `mse_mean`,
`mse_max` (worst component) and `mse_spread` separates "this whole spectrum reconstructs badly"
from "one weak component in an otherwise fine spectrum" — a question single-component data
cannot pose.

**L3, group summary** — `summary_df.csv` per dataset, `recon_summary_all_datasets.csv` across
all of them. Median leads and mean follows: on the multi-component sets a handful of outliers
pull the mean far above the typical sample (the recorded T6 round has `samples` `fe_mse` mean
1.96 against a median of 0.36). Single-component and multi-component results are reported as
separate blocks and never averaged together.

#### Figures

Sample level (unchanged): `recon_overlay.png` (target vs each pathway, per-row y-limits) and
`recon_mse_bars.png` (per-sample, log scale).

Dataset level, in `<model>/reconstruction_<dataset>/` and `<model>/reconstruction_summary/`:

| Figure | Answers |
|---|---|
| `recon_error_distribution` | how big the error is, and how much the single→multi-component shift costs — ECDF plus violins, single and multi blocks divided |
| `recon_summary_heatmap` | all datasets × all metrics at a glance, colored by within-column rank |
| `recon_skill_vs_baseline` | **check this first** — is the model beating a flat line at the sample mean at all? |
| `recon_position_profile` | where along the 245 bins the error sits: conv edge artifacts, bias, period-5 ripple from the final `(512,5,5)` FE stage |
| `recon_amplitude_calibration` | is dynamic range preserved, or has the output collapsed toward the mean? Hexbin with a best-fit slope |
| `recon_spectral_fidelity` | do narrow spectral lines survive, or only the smooth envelope? |
| `recon_error_vs_signal_properties` | *which* spectra fail — per dataset, with component index and the per-spectrum view on multi-component data |

Every figure carries its own explanation as a footnote drawn onto the PNG, so a figure read
outside the report is still self-explanatory; `FIGURES.md` next to the PNGs holds the full
write-up of all of them. `_FIG_DOC` in `recon_plots.py` is the single source for that text, the
report caption and `FIGURES.md` — a figure added without a doc entry raises rather than shipping
unexplained.

**One caveat is stated on every cross-head figure** (TASKS.md T7): the FE decoder is a different
architecture on a narrower input (47×512, `MirrorDecoder`) than the projection and transformer
decoders (47×768, `TransformerMirrorDecoder`), so a gap between heads mixes encoder information
content with decoder capacity. Comparing one head across datasets is clean; comparing heads to
each other is not.

#### Report layout

The reconstruction section of `eval_report.md` / `.html` is a narrative rather than a figure
dump: **1. One sample at a time** (what a reconstruction looks like) → **2. The whole dataset**
(headline table per head, then which spectra fail) → **3. Across datasets** (the six aggregate
figures and the combined summary table).

#### Relevant flags

```bash
--recon_max_samples 1000      # default; the dataset-level figures are distributions and
                              # stratified medians, so 200 leaves the bins too thin
--recon_n_examples 6          # sample-level overlay traces
--recon_seed 42               # which samples get drawn
--no_recon_component_meta     # skip the manifest scan (~17s on multi_channel); every dataset
                              # is then treated as single-component, disabling the
                              # per-spectrum and component-index views
```

#### Component metadata

`data_loader.parse_component_metadata(dataset_dir, split)` scans the **full** manifest, not the
drawn subset: `load_manifest_subset` takes contiguous 10-row blocks and the multi-component
manifests are shuffled, so a subset never holds a whole spectrum. It reports two counts —
`n_comps` (components of this spectrum across the whole dataset) and `n_comps_in_split` (those
present in this split), which diverge sharply: `multi_channel/valid.tsv` holds a single
component for 93k of its 96k spectra while the dataset provides 6 or 10 per spectrum.

It also drops verified byte-identical duplicate components: in both `multi_channel` and
`labeled_data`, `comp20 ≡ comp14` and `comp21 ≡ comp15`. Left in, they double-count ~2/12 and
~2/14 of the mass and inflate every components-per-spectrum count by 2. `sampled_data` has no
duplicates.

#### Self-test

```bash
python -m eval.recon_plots /tmp/recon_selftest
```

Renders every figure from synthetic data with known pathologies (one head faithful, one
amplitude-compressed to 50%, one emitting the sample mean) and asserts the metric signatures —
`r2 ≈ 0` and `amp_ratio ≈ 0` for the mean predictor, `amp_ratio ≈ 0.5` with `pearson ≈ 1` for
the compressed head, `r2 ≈ 1` for the faithful one. No checkpoint or GPU needed.

### 3. `noise_robustness` — embedding stability under input noise

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

- **Metrics:** mean cosine similarity per noise type.
- **Figures:** `noise_example_grid[_<ckpt>].png` — 2×3 grid over all noise types, clean vs noisy
  signal + embedding similarity per cell; `noisy_vs_clean_<type>*.png` — best/worst 3 samples per
  noise type; summary bar chart per noise type (across checkpoints in comparison mode).
- **CSV:** `noise_df[_<ckpt>].csv` — per-sample per-noise-type similarity.

### 4. `clustering` — stack recovery with KMeans

**Question:** Does KMeans on embeddings recover the stack structure?

KMeans with k = n_stacks, scored against true `stack_idx` via `metrics.compute_component_clustering_metrics`.

- **Metrics:** ARI, NMI, V-measure, silhouette, same-stack KNN precision, retrieval mAP,
  inter/intra variance ratio, embedding–input distance alignment.
- **Figures:** `kmeans_clustered_then_tsne[_<ckpt>].png` and `..._umap...png` — PCA(50) →
  t-SNE/UMAP scatter colored by KMeans cluster.

### 5. `label_regression` — parameter_0 linear probe (multi-channel)

**Question:** Do embeddings encode `parameter_0` better than the raw signal?

RidgeCV (5-fold cross-val) probe on inputs vs embeddings, using `--labeled_data_dir`
(`labels.tsv` + wavs, default 2,000 spectra). Wavs follow `dataset<D>_comp<C>_spec_<S>.wav`;
components of the same spectrum share one label. The eval runs three configurations on the
SAME spectra (only those having all 3 components):

| Config | Raw input | Embedding |
|--------|-----------|-----------|
| 1-comp (C0) | 245 | 768 |
| 2-comp (C0,C1) | 490 | 1536 |
| 3-comp (C0,C1,C2) | 735 | 2304 |

Raw channels are concatenated; embeddings are extracted per component then concatenated.

- **Metrics:** `label_reg_input_*` / `label_reg_emb_*` (r2/mse/rmse/mae/pearson/spearman) per
  config (suffix `''` / `_2c` / `_3c`), `label_reg_improvement_r2*` = emb R² − input R² (positive
  ⇒ embeddings add label-relevant information).
- **Figures:** `label_reg_true_vs_pred[_<ckpt>].png` — true-vs-predicted scatter per probe;
  `label_regression_comparison.png` — R² + ΔR² bars across checkpoints.

### 6. `structured_similarity` — canonical 100-sample panel

**Question:** How does similarity structure evolve through the pipeline stages?

Deterministic 100-sample panel (seed=42): 0–29 `single_channel_all` (3 stacks×10), 30–59
`multi_channel` (3 comps×10), 60–79 `sampled_data` (2×10), 80–99 `labeled_data` (2×10). One
forward pass extracts 4 representations: input (245) → FE output (512) → projection (768) →
embedding (768).

- **Figures:** `struct_sim[_<ckpt>].png` — 2×2 cosine-similarity heatmap panel: Input / FE
  output / Projection / Embedding, with white block boundaries and dataset tick labels.
- Requires `--nova_data_dir`.

### 7. `checkpoint_comparison` — everything across multiple checkpoints

Wraps Evals 1, 3, 4, 5, 6 and runs them for every checkpoint from `CheckpointLoader.load_multiple()`
(directory glob or `--checkpoint_paths` list). Produces `comparison_df` — one row per checkpoint
with all scalar metrics — plus a per-checkpoint report section containing every figure. Sub-evals
toggle via `EvalConfig.run_{noise,clustering,label_regression}_in_comparison`.
`signal_reconstruction` runs alongside (once per run), not per-checkpoint. Combine both in one
command for the complete suite.

---

## Checkpoint formats and modes — quick reference

| Format | Detection keys | Loaded as |
|---|---|---|
| `hf` | — (`--checkpoint_mode hf`) | HF pretrained `facebook/data2vec-audio-base` |
| `fairseq` | `cfg` + `model` | keys remapped into the HF model |
| `3ae` | `data2vec_audio` | embedded fine-tuned backbone (+ mirror heads for reconstruction) |
| `fe_recon` | `encoder`+`layer_norm`+`decoder` | FE + LN into the HF model (recon: + MirrorDecoder) |
| `tr_recon` | `transformer_mirror`+`backbone_ckpt` | backbone from recorded path (recon: + mirror head) |
| `state_dict` | anything else | direct `load_state_dict` |

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

Every method runs both standalone and inside `checkpoint_comparison` — same figures and CSVs.
`python -m eval.runner --help` shows all args + usage-mode examples.

---

## Block 6 — Report Generator (`report.py`) / output directory contents

Each run writes a timestamped directory `output_dir/<YYYY-MM-DD_HH-MM-SS>/`:

| File | Content |
|---|---|
| `eval_report.html` / `eval_report.md` | self-contained report (figures base64-embedded) / markdown version |
| `run_info.md` | what was analyzed — data, checkpoints, evals, device |
| `checkpoint_comparison.png` + `*_comparison_df.csv` | scalar metric grid + table (Eval 7) |
| `recon_overlay.png`, `recon_mse_bars.png`, `recon_df.csv` | signal reconstruction (Eval 2) |
| `noise_example_grid*.png`, `noisy_vs_clean_*.png`, `noise_df*.csv` | noise robustness (Eval 3) |
| `kmeans_clustered_then_{tsne,umap}*.png` | clustering scatters (Eval 4) |
| `label_reg_true_vs_pred*.png`, `label_regression_comparison.png` | label regression (Eval 5) |
| `struct_sim*.png` | structured similarity panels (Eval 6) |
| `similarity_comparison_{best,worst}_query*.png`, `match_df*.csv`, `cosine_map_*.png`, `cosine_sim_*.png`, `emb_*.png` | same-stack query (Eval 1) |

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

---

## Data Setup

Manifests live at `/mnt5/noy/SpectralFM/fairseq/data/nova_data/<subset>/` (Geoffrey path).
TSV roots use `/storage/noy/...` — the same NFS as RunAI mounts; eval code remaps `/storage/`
→ `/mnt5/` automatically.

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

Training config: `fairseq/examples/data2vec/config/audio/pretraining/spectralfm_base.yaml`.
Key fix vs base librispeech config: `min_sample_size: 1` (base uses 32000, which silently
drops all 245-frame SpectralFM wavs).

---

## Key constraints

- All eval code must stay importable **without fairseq**.
- SpectralFM wavs are 245 frames; embedding evals layer-norm inputs to match data2vec
  training, but reconstruction receives **raw** signals (normalization applied only if the
  recon checkpoint was trained with it).
- The eval `data_loader` assigns `stack_idx = file_index // 10` (files sorted by name) —
  stacks are consecutive groups of 10.
