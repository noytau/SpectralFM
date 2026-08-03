# SpectralFM — CLAUDE.md

Project context and pointers for AI-assisted development.

---

## What this project is

SpectralFM trains a data2vec-audio backbone (`facebook/data2vec-audio-base`) on 1D spectrograms from the NOVA radio observatory. Each sample is a 245-point 1D signal. A custom CNN feature extractor replaces data2vec's original conv stack.

Training runs on RunAI (GPU cluster). Data lives on shared NFS:
- Geoffrey sees it as `/mnt5/noy/`
- RunAI sees the same volume as `/storage/noy/`

---

## Repo layout

```
SpectralFM/
├── code/
│   ├── eval/                  ← lightweight eval package (no fairseq)
│   │   ├── EVAL_OVERVIEW.md   ← full eval docs — READ THIS FIRST
│   │   ├── checkpoint_loader.py
│   │   ├── model.py
│   │   ├── data_loader.py
│   │   ├── evaluations/
│   │   │   ├── embedding_similarity.py
│   │   │   ├── signal_completion.py
│   │   │   ├── noise_robustness.py
│   │   │   └── checkpoint_comparison.py
│   │   ├── runner.py
│   │   ├── report.py
│   │   └── requirements_eval.txt
│   ├── run_experiment.py      ← training entry point (requires fairseq)
│   ├── evaluate.py            ← old eval (being replaced by code/eval/)
│   ├── model_loader.py        ← model loading + training helpers
│   ├── compute_stats.py       ← visualisation utilities
│   └── data_parser.py         ← original data loader (pkl / csv)
├── fairseq/                   ← forked fairseq with SpectralFM modifications
│   ├── create_manifests.py    ← generate train.tsv / valid.tsv for any wav dir
│   ├── setup_data.sh          ← one-shot: regenerate all nova_data manifests
│   └── examples/data2vec/config/audio/pretraining/
│       └── spectralfm_base.yaml  ← correct training config (min_sample_size: 1)
└── sweep_dataset.sh           ← submit RunAI training jobs across dataset subsets
```

---

## Eval package — key facts

See **[`code/eval/EVAL_OVERVIEW.md`](code/eval/EVAL_OVERVIEW.md)** for full detail.

- **Zero fairseq dependency** — imports only `torch`, `transformers`, `torchaudio`, `sklearn`, `pandas`, `matplotlib`
- **Four evaluations:** embedding similarity, signal completion, noise robustness, checkpoint comparison
- **Four checkpoint modes:** `hf` (HuggingFace), `file`, `dir`, `multiple` (for comparison across training steps)
- **Report:** markdown + PNG figures + CSV exports written to `output_dir/`

Quick run:
```bash
python -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode multiple \
  --checkpoint_path /mnt5/noy/fairseq/outputs/<date>/<time>/checkpoints/ \
  --evals checkpoint_comparison \
  --output_dir /mnt5/noy/code/eval_outputs/
```

---

## Datasets

All datasets live under the shared NFS at:
- **Geoffrey:** `/mnt5/noy/SpectralFM/fairseq/data/nova_data/<subset>/`
- **RunAI:** `/storage/noy/SpectralFM/fairseq/data/nova_data/<subset>/` (same volume, different mount)

Each subset directory contains `train.tsv`, `valid.tsv`, and a `wav/` subdirectory.  
TSV line 1 is the root path written with the `/storage/` prefix so RunAI jobs resolve files correctly.  
All wavs are 245 frames, 16 kHz, single channel, values in `[0, 1]` (mean ≈ 0.46, std ≈ 0.25).

### Main training subsets

These are the subsets used in `sweep_dataset.sh`. Manifests use `/storage/` roots.

| Subset | Wav files | Train | Valid | Wav naming |
|--------|----------:|------:|------:|------------|
| `single_channel_100` | 100 | 90 | 10 | `spec_N.wav` |
| `single_channel_1k` | 1,000 | 950 | 50 | `spec_N.wav` |
| `single_channel_10k` | 11,111 | 10,611 | 500 | `spectra0000_batch0_spec_N.wav` |
| `single_channel_one` | 1,000,000 | 999,000 | 1,000 | `spec_N.wav` |
| `single_channel_5m` | 5,000,000 | 4,995,000 | 5,000 | (batch-prefixed) |
| `single_channel_all` | 9,109,930 | 9,099,930 | 10,000 | `spectra0000_<batch>_spec_N.wav` |

`single_channel_all` aggregates all batches (`batch0`–`batch8`, `e5`, etc.) into one dataset.  
`single_channel_one` is a clean 1M-sample draw of `spec_N.wav` files (single provenance, no batch prefix).

### Special / reference datasets

These use `/mnt5/` roots in their TSVs (Geoffrey-only, not submitted to RunAI).

| Subset | Wav files | Train | Valid | Purpose |
|--------|----------:|------:|------:|---------|
| `multi_channel` | 3,412,476 | 3,241,852 | 170,624 | Multi-component data; wavs named `datasetNNNN_compK_spec_N.wav` |
| `labeled_data` | 66,024 | 62,722 | 66,023 | Has `labels.tsv` (float label per wav); used for supervised eval |
| `sampled_data` | ~21,000 | 21,306 | — | Diverse sample across batches |
| `single_channel_100_var` | 100 | 100 | — | Variance-selected 100 samples |
| `base_libri_100` | 34 | 90 | — | LibriSpeech sanity-check reference (not SpectralFM data) |
| `single_sample` | 2 | 1 | 1 | Single-sample smoke test |

### Signal distribution (sampled across subsets)

All single-channel subsets share the same distribution — data comes from the same source.

```
frames per wav : 245
sample rate    : 16,000 Hz
value range    : [0.0, ~0.99]  (normalised, not audio-range [-1,1])
mean           : ~0.46
std            : ~0.25
```

Note: the eval `data_loader.py` re-normalises to `[-1, 1]` via `normalize_to_audio_range()` before model inference.

### Manifest format

```
/storage/noy/SpectralFM/fairseq/data/nova_data/<subset>/wav   ← line 1: root dir
spec_0.wav    245                                              ← filename \t num_frames
spec_1.wav    245
...
```

Regenerate for any subset: `python fairseq/create_manifests.py --help`  
Regenerate all at once: `bash fairseq/setup_data.sh` (run on Geoffrey)

---

## Server access

```
ssh Geoffry   # 132.66.52.64, user noy — configured in ~/.ssh/config
```

Python on Geoffrey: `/mnt5/noy/miniconda3/envs/spectralfm_env/bin/python3`  
This env has torch 2.8, transformers 4.57, torchaudio 2.8 — all eval deps satisfied.

---

## Key constraints

- `min_sample_size` in training config **must be 1** — SpectralFM wavs are 245 frames; the base librispeech config uses 32000 which silently drops everything.
- `signal_completion` eval requires a `completion_head` on the model — it skips gracefully if not present.
- The old `model_loader.py` imports `from fairseq import checkpoint_utils` — do not use it in the eval path.
- All new eval code must remain importable without fairseq installed.
