# SpectralFM Eval Package — Overview

Lightweight evaluation package under `code/eval/`. **No fairseq required.**

---

## Architecture

```
code/eval/
├── checkpoint_loader.py          # Block 1: load any checkpoint (HF / .pt file / dir / multiple)
├── model.py                      # Block 2: model components — no fairseq
├── data_loader.py                # Block 3: load wav / CSV, masking, dataloaders
├── evaluations/
│   ├── embedding_similarity.py   # Eval 1: do embeddings cluster same-stack samples?
│   ├── signal_completion.py      # Eval 2: can the model predict masked signal parts?
│   ├── noise_robustness.py       # Eval 3: how stable are embeddings under noise?
│   └── checkpoint_comparison.py  # Eval 4: run Evals 1+3 across multiple checkpoints
├── runner.py                     # EvalRunner + EvalConfig + CLI entry point
├── report.py                     # markdown + PNG + CSV report generator
└── requirements_eval.txt         # minimal install (see table below)
```

---

## Quick Start

```bash
pip install -r code/eval/requirements_eval.txt

# Single checkpoint — embedding similarity + noise robustness
python -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode dir \
  --checkpoint_path /mnt5/noy/fairseq/outputs/2025-10-29/15-57-29/checkpoints/ \
  --evals embedding_similarity noise_robustness \
  --output_dir /mnt5/noy/code/eval_outputs/

# Compare multiple checkpoints across training
python -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode multiple \
  --checkpoint_path /mnt5/noy/fairseq/outputs/2025-10-29/15-57-29/checkpoints/ \
  --checkpoint_pattern "checkpoint*.pt" \
  --evals checkpoint_comparison \
  --output_dir /mnt5/noy/code/eval_outputs/
```

Python API:

```python
from eval.runner import EvalRunner, EvalConfig

runner = EvalRunner(EvalConfig(
    data_source="/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav",
    checkpoint_mode="multiple",
    checkpoint_path="/mnt5/noy/fairseq/outputs/2025-10-29/15-57-29/checkpoints/",
    evals=["checkpoint_comparison"],
    output_dir="/mnt5/noy/code/eval_outputs/",
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
| `file` | Single `.pt` file — auto-detects state_dict vs fairseq format by key inspection |
| `dir` | Directory scan: prefers `checkpoint_best.pt` → `checkpoint_last.pt` → highest numbered |
| `multiple` | All files matching a glob pattern → returns `[(label, model), ...]` for comparison eval |

```python
# Auto-detect fairseq vs plain state_dict
state = torch.load(path, map_location="cpu")
if "cfg" in state and "model" in state:   # fairseq format
    model.load_state_dict(state["model"], strict=False)
else:                                      # plain state_dict
    model.load_state_dict(state, strict=strict)
```

---

## Block 2 — Model (`model.py`)

Custom CNN frontend (replaces data2vec's original conv stack) and optional completion head.

```python
class CustomFeatureExtractor(nn.Module):
    # Supported archs: conv1d | 2_conv1d_relu | 2_conv1d | 3_conv1d_relu | 3_conv1d

class CompletionHead(nn.Module):
    # Linear(768 → 245) — used only by signal_completion eval

def build_model(hf_model, arch="conv1d", with_completion_head=False):
    hf_model.feature_extractor = CustomFeatureExtractor(arch)
    # Freezes all params except the custom feature extractor
    return hf_model
```

---

## Block 3 — Data Loader (`data_loader.py`)

```python
load_data(source)                   # auto: .wav dir → DataFrame, or CSV → DataFrame
split_stack_holdout(df, n=5)        # hold out first N stacks for eval
split_partial_stack(df, ratio=0.3)  # hold out 30% of each stack
build_dataloader(df, mask_ratio, masking_type, batch_size)
    # masking_type: random | grid | span | span_start | span_end | low_energy | high_energy
    # returns (DataLoader, df_with_masked_data_column)
```

---

## Eval 1 — Embedding Similarity (`evaluations/embedding_similarity.py`)

**Question:** Does the embedding space group same-stack samples better than raw input space?

For each sample: find top-k neighbors in input space and embedding space. Count how many are from the same "stack" (group of 10 spectrogram rows from the same file). A match score > 50 means the model improves on the raw-input baseline.

```python
input_sims = cosine_similarity(query_input, all_inputs)
emb_sims   = cosine_similarity(query_emb,   all_embeddings)

emb_stack_matches = [i for i in topk_emb   if df.iloc[i]["stack_idx"] == query_stack]
inp_stack_matches = [i for i in topk_input if df.iloc[i]["stack_idx"] == query_stack]

match_score = ((len(emb_matches) - len(inp_matches) + k) / (2 * k)) * 100  # 0–100
```

Returns: `embedding_stack_match_rate`, `input_stack_match_rate`, `match_score_avg` (0–100), `results_df`, `embeddings`.

---

## Eval 2 — Signal Completion (`evaluations/signal_completion.py`)

**Question:** Can the model predict the masked-out parts of a signal?

Requires a `completion_head` on the model (a `Linear(768 → 245)` head trained for reconstruction). Skips gracefully if not present — check `has_completion_head(model)` before running.

```python
emb  = model(masked_input).last_hidden_state.mean(dim=1)  # [B, 768]
pred = model.completion_head(emb)                          # [B, 245]
mse  = F.mse_loss(pred, clean_input, reduction="none").mean(dim=1)
```

Returns: `avg_mse`, `results_df` (with `inputs`, `masked`, `predicted`, `mse` per sample), or `{"skipped": True}`.

---

## Eval 3 — Noise Robustness (`evaluations/noise_robustness.py`)

**Question:** How stable are embeddings when noise is added to the input?

Score = cosine similarity between clean embedding and noisy embedding. Higher = more robust.

| Noise type | Formula |
|-----------|---------|
| `gaussian_std` | `x + N(0, 0.01)` |
| `gaussian_mean` | `x + N(2, 0.001)` |
| `shot_low` | `Poisson(x * 0.1) / 0.1` |
| `shot_high` | `Poisson(x * 0.05) / 0.05` |
| `gain_low` | `x * N(1, 0.05)` |
| `gain_high` | `x * N(1, 0.1)` |

Returns: `summary = {noise_type: mean_cosine_sim}`, `results_df`.

---

## Eval 4 — Checkpoint Comparison (`evaluations/checkpoint_comparison.py`)

Wraps Eval 1 + Eval 3 and runs them for every checkpoint loaded by `CheckpointLoader.load_multiple()`. Produces a table showing how scores evolve during training.

```python
checkpoints = CheckpointLoader.load_multiple(checkpoint_dir, pattern="checkpoint*.pt")
# → [("checkpoint1", model1), ("checkpoint_best", model2), ...]

results = CheckpointComparisonEval.run(checkpoints, df, dataloader)
# results["comparison_df"]:
#   checkpoint | emb_stack_match_rate | match_score_avg | noise_gaussian_std | ...
```

---

## Block 5 — Unified Runner (`runner.py`)

`EvalConfig` dataclass covers all parameters. `EvalRunner` handles data loading, checkpoint loading, dispatching evals, and report generation.

```python
@dataclass
class EvalConfig:
    data_source:        str           # .wav dir or CSV path
    checkpoint_mode:    str           # hf | file | dir | multiple
    checkpoint_path:    str | None    # path to .pt / dir
    checkpoint_pattern: str           # glob for 'multiple' mode
    evals:              List[str]     # subset of: embedding_similarity, signal_completion,
                                      #   noise_robustness, checkpoint_comparison
    split_mode:         str           # stack_holdout | partial_stack | none
    k:                  int  = 5
    mask_ratio:         float = 0.15
    masking_type:       str  = "random"
    batch_size:         int  = 16
    device:             str  = "auto"  # auto-selects cuda if available
    output_dir:         str  = "eval_outputs"
```

---

## Block 6 — Report Generator (`report.py`)

Writes to `output_dir/`:

```
eval_report_<timestamp>.md          ← main report with inline figure references
emb_similarity_histogram.png        ← Eval 1: input vs embedding match count histogram
emb_match_score_dist.png            ← Eval 1: match score distribution
signal_completion_mse.png           ← Eval 2: MSE distribution
signal_completion_best/worst.png    ← Eval 2: best/worst predicted signals
noise_robustness.png                ← Eval 3: bar chart per noise type
checkpoint_comparison.png           ← Eval 4: metric curves across checkpoints
embedding_similarity_results_df.csv
noise_robustness_results_df.csv
```

---

## Requirements (`requirements_eval.txt`)

```
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
soundfile>=0.12.0
tabulate>=0.9.0
```

| Package | Used by |
|---------|---------|
| `torch` | **Everything** — tensors, model inference, masking, MSE loss |
| `torchaudio` | `data_loader.py` — MelSpectrogram transform, wav file loading |
| `transformers` | `checkpoint_loader.py`, `model.py` — `Data2VecAudioModel.from_pretrained` |
| `numpy` | All evals — cosine similarity matrices, noise generation, array ops |
| `pandas` | All evals + report — results DataFrames, CSV export |
| `scikit-learn` | `embedding_similarity.py`, `noise_robustness.py` — `cosine_similarity` |
| `matplotlib` | `report.py` — all PNG figure generation |
| `seaborn` | `report.py` — histogram styling (optional; drop if minimising deps) |
| `soundfile` | `data_loader.py` — low-level wav backend for torchaudio |
| `tabulate` | `report.py` — DataFrame → markdown table |

**Not needed:** `fairseq`, `hydra-core`, `omegaconf`, `apex`, `submitit`.

On Geoffrey the `spectralfm_env` conda environment already has all of these.

---

## Data Setup (RunAI)

Manifests live at `/mnt5/noy/SpectralFM/fairseq/data/nova_data/<subset>/` (Geoffrey path).  
TSV roots use `/storage/noy/...` — the same NFS, as RunAI mounts it.

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
Key fix vs base librispeech config: `min_sample_size: 1` (base uses 32000, which silently drops all 245-frame SpectralFM wavs).
