# SpectralFM — CLAUDE.md

Project context and pointers for AI-assisted development.

---

## What this project is

SpectralFM trains a data2vec-audio backbone (`facebook/data2vec-audio-base`) on 1D spectrograms from the NOVA radio observatory. Each sample is a 245-point 1D signal. A custom CNN feature extractor replaces data2vec's original conv stack.

Training runs on RunAI (GPU cluster). Data lives on shared NFS:
- Geoffrey sees it as `/mnt5/noy/`
- RunAI sees the same volume as `/storage/noy/`

Two things get trained here, and they're separate code paths:
1. **The backbone itself** — self-supervised data2vec pretraining on raw signals. This is "regular training." See [Getting started](#getting-started-new-teammate-setup) below.
2. **Reconstruction decoders on top of a backbone** ("autoencoder" / "3AE" training) — teaches one or more decoder heads to rebuild the original 245-point signal from different depths of an already-trained (or randomly initialized) backbone. See [Autoencoder / reconstruction training](#autoencoder--reconstruction-training).

---

## Documentation index

**This file is the one place to start.** Everything below is every doc in the
project, grouped by what you're trying to do. If you only read one other
thing, read `code/eval/EVAL_OVERVIEW.md` (eval) or `ARCHITECTURE.md`
(architecture) next, depending on which side you're touching.

| Doc | What it's for |
|---|---|
| **Start here** | |
| [`CLAUDE.md`](CLAUDE.md) | This file — setup, training commands, project map. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Technical reference: model architecture, loss setup, checkpoint loading, what's actually been swept. |
| [`TASKS.md`](TASKS.md) | The live experiment ledger — what's done, what's active, what's next, in priority order. |
| [`docs/html/`](docs/html/) | Same content as above two, as standalone HTML pages — see [HTML docs](#html-docs) below. |
| **Training** | |
| [`fairseq/examples/data2vec/config/audio/pretraining/recon_loss/HOW_TO_RUN.md`](fairseq/examples/data2vec/config/audio/pretraining/recon_loss/HOW_TO_RUN.md) | Reconstruction-loss training — the single doc for it: Hydra commands, RunAI file sync, artifact pull, all in one place. |
| [`code/docs/RECON_2AE_BASEMERGE.md`](code/docs/RECON_2AE_BASEMERGE.md) | Design rationale and API surface for the per-component init/freeze/composite-optimizer machinery — merged and current (also see the HTML version). |
| **Evaluation** | |
| [`code/eval/EVAL_OVERVIEW.md`](code/eval/EVAL_OVERVIEW.md) | **The single eval doc** — installation, running it, all 7 methods, checkpoint formats, examples. `code/eval/README.md` is a one-paragraph pointer to this file, kept only because GitHub renders it by default when browsing the folder. |
| **Reference / provenance** | |
| [`code/README.md`](code/README.md) | One-line project tagline (oldest doc in the repo). |
| [`fairseq/examples/data2vec/README.md`](fairseq/examples/data2vec/README.md) | Upstream fairseq data2vec docs (not SpectralFM-specific — background reading on the base architecture). |
| **Lower priority / historical** — still correct, just less likely to be what you need today | |
| [`docs/AUTOENCODER_EXPERIMENTS.md`](docs/AUTOENCODER_EXPERIMENTS.md) | Autoencoder/reconstruction experiment history — mostly superseded by `RECON_2AE_BASEMERGE.md`; kept for the phase 1–3 sweep results and operational notes. |
| [`fairseq/examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse/HOW_TO_RUN.md`](fairseq/examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse/HOW_TO_RUN.md) | Commands for the FE-vs-transformer embedding-collapse ablation — a narrower, one-off experiment. |
| [`docs/RUNAI_TRAINING_FOR_CLAUDE.md`](docs/RUNAI_TRAINING_FOR_CLAUDE.md) | RunAI operational reference (PVC paths, checkpoint copy recipes, submit-script inventory) — less relevant since the cluster/server setup moved on from what this was written against; check current server details in [Server access](#server-access) first. |

Everything else under `fairseq/` outside `examples/data2vec/` is vendored
upstream fairseq documentation (translation, speech-to-text, other model
families) — not part of SpectralFM, safe to ignore unless you're touching
that specific fairseq subsystem.

### HTML docs

All standalone HTML documentation lives in one place: [`docs/html/`](docs/html/).

| File | Mirrors |
|---|---|
| `docs/html/index.html` | **Start here** — hub page linking every doc and resource in the project. |
| `docs/html/architecture.html` | `ARCHITECTURE.md`, styled for reading rather than editing. |
| `docs/html/project-story.html` | A non-technical project overview — what SpectralFM is and why, for someone outside the immediate team. |
| `docs/html/RECON_2AE_BASEMERGE.html` | `code/docs/RECON_2AE_BASEMERGE.md`. |
| `docs/html/recon_loss_explained.html` | Background on the original data2vec reconstruction-loss design. |
| `docs/html/transformer_signal_recon_training.html` | Transformer-stage signal-reconstruction training walkthrough. |

---

## Getting started (new teammate setup)

Do this once, in order, on **Geoffrey** (`ssh Geoffry`, see [Server access](#server-access)):

1. **Environment.** Training needs the `spectralfm` conda env (fairseq + its deps), defined in `spectralfm.yml` at the repo root (also what the RunAI training image bakes in — see `Dockerfile`). This is a **different env** from the one used for evaluation (`spectralfm_env`, see [Server access](#server-access)) — eval is deliberately fairseq-free, training is not.
2. **Data.** Confirm the dataset subset you want to train on already has manifests under `/mnt5/noy/SpectralFM/fairseq/data/nova_data/<subset>/` (check for `train.tsv`/`valid.tsv`). If not, generate them first — see [Manifest generation](#manifest-generation) below. This is a one-time step per dataset subset, not part of every training run.
3. **Launch a training run** — see the command in the next section.
4. **Evaluate the result** once you have a checkpoint — see `code/eval/EVAL_OVERVIEW.md`.

### Regular (backbone) training

This is fairseq's native `data2vec_audio` self-supervised pretraining — the model config lives in `fairseq/examples/data2vec/config/audio/pretraining/spectralfm_base.yaml`, and it's launched with fairseq's own `hydra_train` entry point via `runai submit`. You do not write data paths or job config into the yaml — everything dataset/step/GPU-specific is passed as a hydra override at launch time.

The easiest way to launch is `sweep_dataset.sh` at the repo root, which submits one RunAI job per dataset subset listed in its `SUBSETS=(...)` array:

```bash
bash sweep_dataset.sh
```

To launch a single custom run instead of the sweep, submit directly:

```bash
runai submit spectral-my-run \
  --image noyhassid/spectralfm-lean:v6 \
  --gpu 1 --project raja \
  --existing-pvc claimname=storage,path=/storage \
  --preemptible \
  --command -- bash -c "cd /storage/noy/SpectralFM/fairseq && \
    WANDB_NAME=my_run_name python -m fairseq_cli.hydra_train \
    --config-dir examples/data2vec/config/audio/pretraining \
    --config-name spectralfm_base \
    task.data=/storage/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
    common.user_dir=/storage/noy/SpectralFM/fairseq/examples/ \
    optimization.lr=[0.0001] optimization.max_update=10000 optimization.max_epoch=0 \
    lr_scheduler._name=cosine +lr_scheduler.warmup_updates=1000 \
    +lr_scheduler.warmup_init_lr=0 +lr_scheduler.min_lr=0 +lr_scheduler.max_update=10000 \
    common.log_interval=10"
```

Swap `task.data=` to point at any dataset subset with a manifest (see below). Checkpoints save every 5000 updates (keeping the last 3) into the job's RunAI output dir under `/storage`; `wandb_run_name` in the run log is what to search for in W&B.

**One config gotcha that will silently break a run if missed:** `spectralfm_base.yaml` explicitly sets `model.train_only_fe: false`, because the field defaults to `true` (added by the autoencoder-training merge) and would otherwise freeze everything except the feature extractor. If you ever start a *new* config from scratch for full backbone training rather than copying `spectralfm_base.yaml`, set this explicitly.

### Manifest generation

A one-time step per dataset subset, independent of any specific training run — not something you redo before every launch.

`.tsv` manifests (`train.tsv`/`valid.tsv`) point fairseq at a directory of `.wav` files. Generate them from a wav directory with:

```bash
python fairseq/create_manifests.py \
  --wav_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/<subset>/wav \
  --out_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/<subset> \
  --runai_root /storage/noy/SpectralFM/fairseq/data/nova_data/<subset>/wav \
  --max_train 10000 --valid_frac 0.05
```

`--runai_root` is what gets written as the manifest's root path (line 1 of the `.tsv`) — always the `/storage/...` RunAI path, even though you're generating the manifest from Geoffrey, since that's what RunAI jobs will actually read.

To regenerate manifests for every existing subset at once: `bash fairseq/setup_data.sh` (run on Geoffrey). See the [Datasets](#datasets) section below for which subsets already have manifests and which are wired into `sweep_dataset.sh`'s default sweep.

**Example — a small smoke-test subset**, capped to 200 files so a launch finishes in minutes instead of hours:
```bash
python fairseq/create_manifests.py \
  --wav_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_one/wav \
  --out_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/smoke_200 \
  --runai_root /storage/noy/SpectralFM/fairseq/data/nova_data/smoke_200/wav \
  --max_train 200 --valid_frac 0.1
```

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
│   │   │   ├── signal_reconstruction.py
│   │   │   ├── noise_robustness.py
│   │   │   └── checkpoint_comparison.py
│   │   ├── recon_plots.py     ← dataset-level reconstruction figures
│   │   ├── recon_analysis.py  ← metrics those figures and findings share
│   │   ├── signal_features.py
│   │   ├── findings.py        ← the report's generated closing section
│   │   ├── runner.py
│   │   ├── report.py
│   │   ├── report_pdf.py      ← e-ink PDF build of the reconstruction report
│   │   └── requirements_eval.txt
│   ├── train_reconstruction.py   ← standalone autoencoder/reconstruction trainer (requires fairseq)
│   ├── recon_components.py       ← per-component checkpoint loaders used by both training paths
│   ├── model_loader.py           ← checkpoint loading + /storage↔/mnt5 remap (used by evaluation_runner.py etc.)
│   ├── evaluation_runner.py      ← older, fairseq-dependent multi-checkpoint eval workhorse (predates code/eval/)
│   └── docs/                     ← deep-dive docs for specific pieces of code/ (RECON_2AE_BASEMERGE.md, ...)
├── docs/                       ← project-level deep-dive docs (AUTOENCODER_EXPERIMENTS.md, RUNAI_TRAINING_FOR_CLAUDE.md, ...)
├── fairseq/                    ← forked fairseq with SpectralFM modifications
│   ├── create_manifests.py     ← generate train.tsv / valid.tsv for any wav dir
│   ├── setup_data.sh           ← one-shot: regenerate all nova_data manifests
│   ├── submit_signal_recon_*.sh, submit_recon_*.sh  ← RunAI launchers for autoencoder/reconstruction training rounds
│   └── examples/data2vec/config/audio/pretraining/
│       ├── spectralfm_base.yaml     ← canonical backbone training config (min_sample_size: 1)
│       └── recon_loss/              ← Hydra configs for reconstruction-loss training
└── sweep_dataset.sh            ← submit RunAI backbone-training jobs across dataset subsets
```

A note on provenance: `code/train_reconstruction.py`, `code/recon_components.py`, `code/evaluation_runner.py` and the `docs/` folder arrived via merging a previously-separate `recon/2ae-basemerge` branch (and other autoencoder-training work already on `main`) into this branch. There is a fair amount of exploratory/one-off tooling in `code/` from that work (analysis scripts, debug scripts, alternate eval entry points) that hasn't been triaged yet — see the `TASKS.md` cleanup item before assuming every `code/*.py` file is load-bearing.

---

## Eval package — key facts

See **[`code/eval/EVAL_OVERVIEW.md`](code/eval/EVAL_OVERVIEW.md)** for full detail.

- **Zero fairseq dependency** — imports only `torch`, `transformers`, `torchaudio`, `sklearn`, `pandas`, `matplotlib`
- **Four evaluations:** embedding similarity, signal completion, noise robustness, checkpoint comparison
- **Four checkpoint modes:** `hf` (HuggingFace), `file`, `dir`, `multiple` (for comparison across training steps)
- **Report:** markdown + PNG figures + CSV exports written to `output_dir/`
- **Reconstruction (3AE)** additionally gets eight dataset-level figures across all four
  recon datasets, a generated closing section, and an e-ink PDF — see
  [`EVAL_OVERVIEW.md`](code/eval/EVAL_OVERVIEW.md#2-signal_reconstruction--true-reconstruction-through-the-pipeline):
  ```bash
  python -m eval.runner --evals signal_reconstruction --recon_ckpt <3ae_ckpt.pt> \
    --nova_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data \
    --eval_set_size all --recon_max_samples 0 --output_dir <out>/
  ```

Quick run — compare every checkpoint in a training run's output dir:
```bash
python -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode multiple \
  --checkpoint_path /mnt5/noy/fairseq/outputs/<date>/<time>/checkpoints/ \
  --evals checkpoint_comparison \
  --output_dir /mnt5/noy/code/eval_outputs/
```

Single checkpoint, all four evals, straight from a HuggingFace-format model:
```bash
python -m eval.runner \
  --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \
  --checkpoint_mode hf \
  --checkpoint_path facebook/data2vec-audio-base \
  --evals embedding_similarity signal_completion noise_robustness \
  --output_dir /mnt5/noy/code/eval_outputs/baseline/
```

---

## Autoencoder / reconstruction training

Trains one or more decoder heads to reconstruct the original 245-point signal from a (frozen or trainable) backbone. Two independent entry points exist, at different levels of composability. Full write-up (design rationale, param-group tagging, the 6 experiment configurations tried so far): `code/docs/RECON_2AE_BASEMERGE.md`. Experiment history and results: `docs/AUTOENCODER_EXPERIMENTS.md`.

### Standalone script — `code/train_reconstruction.py`

Simpler, no Hydra. Two modes:
```bash
# FE-only: conv feature extractor + LayerNorm + a ConvTranspose1d mirror decoder
python code/train_reconstruction.py --mode train --recon_path fe \
  --lr 1e-4 --n_samples 1000 --steps 10000 --data_dir <wav_dir>

# Transformer: full backbone (FE + transformer) + mirror decoder on the full sequence
python code/train_reconstruction.py --mode train --recon_path transformer \
  --ckpt none --n_samples 1000 --steps 2000 --data_dir <wav_dir>
  # --ckpt none = random backbone; pass a fairseq .pt instead to warm-start it
```
`--init_*_ckpt` / `--freeze_*` / `--lr_*` flags (per-component: fe, ln, proj, transformer, and each decoder head) let you mix warm-started and frozen components — e.g. "reconstruction heads on top of a frozen, already-trained SSL backbone." `--lambda_recon_fe` / `--lambda_recon_trans` weight each head's loss; `--lambda_tv_fe` adds a total-variation smoothness penalty on the FE decoder's output.

**Worked example — freeze a pretrained backbone, train only the two decoder heads on top of it:**
```bash
python code/train_reconstruction.py --mode train --recon_path transformer \
  --ckpt /mnt5/noy/SpectralFM/checkpoints/runai/my_ssl_backbone.pt \
  --init_fe_ckpt /mnt5/noy/SpectralFM/checkpoints/runai/my_ssl_backbone.pt \
  --init_transformer_ckpt /mnt5/noy/SpectralFM/checkpoints/runai/my_ssl_backbone.pt \
  --freeze_fe_v2 --freeze_transformer \
  --lambda_recon_fe 1.0 --lambda_recon_trans 1.0 --lambda_tv_fe 0.1 \
  --lr_head_fe 1e-4 --lr_head_trans 1e-4 \
  --data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_1k/wav \
  --n_samples 950 --steps 10000 --batch_size 128 \
  --wandb_project spectralfm-autoencoder --out_dir autoencoder_experiments/my_run
```
Only the FE decoder and transformer decoder receive gradients here (`--freeze_fe_v2 --freeze_transformer`); the backbone stays exactly as it was trained. This is the "T5-style" recipe from `TASKS.md` — reconstruction heads on top of a frozen, already-good SSL backbone, the direction the T6 finding (see [Loss setup](ARCHITECTURE.md#2-loss-setup)) points toward.

### Hydra path — same `data2vec_audio` model, launched like backbone training

For sweeps that need RunAI's job queue and W&B integration the way regular backbone training does. Configs live under `fairseq/examples/data2vec/config/audio/pretraining/recon_loss/`; launch with the matching `fairseq/submit_signal_recon_*.sh` / `fairseq/submit_recon_*.sh` script, or directly:
```bash
cd fairseq
PYTHONPATH=/mnt5/noy/SpectralFM/code fairseq-hydra-train \
  --config-dir examples/data2vec/config/audio/pretraining/recon_loss \
  --config-name spectralfm_recon_loss_basemerge
```
`PYTHONPATH` must include `code/` — the model lazy-imports `code/recon_components.py` for the per-component checkpoint loaders; without it, init/freeze/param-group-tagging fields silently become no-ops (a warning, not a crash — you'll get random init instead of the warm-start you asked for). Override per run, e.g. `model.lambda_recon_fe=0.5`, `optimizer.groups.transformer.lr=[3e-5]`.

### Checkpoint loading

Three different mechanisms, depending on what's loading what:
- **Warm-starting a component during training** (`--init_fe_ckpt` / `model.init_fe_ckpt` etc.): `code/recon_components.py`'s loaders auto-detect which of several known checkpoint layouts a `.pt` file uses (plain fairseq audio checkpoint, `data2vec_multi`, or the older standalone `apr28_fe_recon`-style save) and remap keys accordingly — point at the file, you don't need to know its exact save format up front.
- **Loading a full checkpoint for eval/inference** (`code/model_loader.py:load_fairseq_checkpoint`): uses fairseq's own `checkpoint_utils.load_model_ensemble_and_task`. Before loading, it inspects the checkpoint's embedded config for any `/storage/noy/...` path (e.g. a `model_path` pointing at the base checkpoint it was warm-started from) and remaps it to `/mnt5/noy/...` when that's where you're running, auto-setting `skip_pretrained_weights` if no local copy of that base checkpoint exists — so an eval run doesn't fail just because the original warm-start file isn't on your machine. It also backfills config keys missing from older checkpoints (`model_path`, `skip_pretrained_weights`, `train_only_fe`) so old and new checkpoints load the same way.
- **The zero-fairseq eval package** (`code/eval/checkpoint_loader.py`): a separate, from-scratch loader — see `code/eval/EVAL_OVERVIEW.md`. It detects checkpoint format from its keys and rebuilds an equivalent HuggingFace model shell, entirely without a fairseq install.

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
- `model.train_only_fe` on `data2vec_audio` **defaults to `true`** (freezes everything except the feature extractor). `spectralfm_base.yaml` overrides it to `false` for full backbone training — any new full-training config must do the same, or training will silently become FE-only.
- `signal_completion` eval requires a `completion_head` on the model — it skips gracefully if not present.
- `code/model_loader.py` imports `from fairseq import checkpoint_utils`, so it needs fairseq installed — it's used by `code/evaluation_runner.py` and the autoencoder training scripts, not by the zero-fairseq `code/eval/` package. Don't import it from `code/eval/`.
- All code under `code/eval/` must remain importable without fairseq installed.

---

## Overnight mode

When the user says they're going to sleep (or asks to run things overnight), do ALL of the following:

1. **Keep the Mac awake:** start `caffeinate -dis` as a background Bash task
   (prevents idle sleep while the session runs; remind the user: plugged in, lid open).
2. **Make the work session-independent:** every long-running job must run on
   Geoffrey under `nohup`, and any multi-step chain (train → eval → next launch)
   must be orchestrated by a `nohup`'d watcher script ON GEOFFREY — never only by
   local background tasks. Watchers key on recorded PIDs (`ps -p`) and
   checkpoint-file existence — NEVER `pgrep -f <pattern>` (the ssh wrapper's own
   command line matches the pattern and the watcher hangs forever).
3. **VPN-drop tolerance:** local monitoring loops must retry `ssh Geoffry` (poll
   interval ≥ 30 min for overnight horizons) so a VPN reconnect only delays a ping.
4. **Sign-off summary:** before ending the turn, list what is running (GPU, run
   name, ETA), what is auto-chained on the server, and what the morning
   deliverables will be. Worst case (Mac dies) must lose nothing but reporting.
