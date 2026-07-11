# SpectralFM Evaluation

Lightweight evaluation package — **no fairseq required**.

## Install

```bash
pip install -r requirements_eval.txt
```

## Quick start

```bash
# Run embedding similarity eval on a checkpoint directory
python -m eval.runner \
  --data_source /mnt5/noy/nova_samples/debug_chnl/ \
  --checkpoint_mode dir \
  --checkpoint_path /mnt5/noy/fairseq/outputs/2025-10-29/15-57-29/checkpoints/ \
  --evals embedding_similarity noise_robustness \
  --output_dir /mnt5/noy/code/eval_outputs/

# Compare multiple checkpoints across training
python -m eval.runner \
  --data_source /mnt5/noy/nova_samples/one_chnl/ \
  --checkpoint_mode multiple \
  --checkpoint_path /mnt5/noy/fairseq/outputs/2025-10-29/15-57-29/checkpoints/ \
  --checkpoint_pattern "checkpoint*.pt" \
  --evals checkpoint_comparison \
  --output_dir /mnt5/noy/code/eval_outputs/

# Use HuggingFace pretrained (no local checkpoint)
python -m eval.runner \
  --data_source /mnt5/noy/nova_samples/debug_chnl/ \
  --checkpoint_mode hf \
  --evals embedding_similarity signal_completion \
  --output_dir /mnt5/noy/code/eval_outputs/
```

## Evaluations

| Name | Description | Requires |
|------|-------------|----------|
| `embedding_similarity` | Top-k neighbor overlap between input and embedding space (same-stack) | any model |
| `signal_completion` | MSE of predicting masked signal regions | model with `completion_head` |
| `noise_robustness` | Cosine similarity of clean vs noisy embeddings | any model |
| `checkpoint_comparison` | Runs embedding_similarity + noise_robustness across multiple checkpoints | multiple .pt files |

## Checkpoint modes

| Mode | `--checkpoint_path` | Description |
|------|---------------------|-------------|
| `hf` | — | Load `facebook/data2vec-audio-base` from HuggingFace |
| `file` | path to `.pt` | Load single checkpoint (state_dict or fairseq format) |
| `dir` | path to directory | Auto-detect `checkpoint_best.pt` → `checkpoint_last.pt` → latest |
| `multiple` | path to directory | Load all matching `--checkpoint_pattern` for comparison |

## Output

Results are saved to `--output_dir/`:
- `eval_report_<timestamp>.md` — main report with figures
- `*.png` — figures referenced in report
- `*.csv` — raw results per evaluation
