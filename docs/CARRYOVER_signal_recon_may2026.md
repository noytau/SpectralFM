# Carry-over: signal reconstruction tooling (May 2026)

Use this file when moving to another machine or chat. **Experiment logs and checkpoints** live under `autoencoder_experiments/` (large); they are **not** committed to git by default—copy that tree or sync PVC if needed.

## Launch the four 100k runs (FE×2 + transformer×2)

From repo root `/mnt5/noy/SpectralFM` (or set `REPO`):

```bash
cd /mnt5/noy/SpectralFM
export PYTHONPATH=code:fairseq:fairseq/examples
# Detached tmux (survives SSH disconnect); 4 GPUs → parallel, <4 → sequential queue on GPU 0
USE_TMUX=1 bash scripts/launch_recon_signal_4x100k.sh
```

- **Defaults:** `n_samples=steps=100000`, `lr=1e-4`, `warmup=10000` (cosine), `batch_size=512`, `grad_accum_steps=4`, manifest `fairseq/data/nova_data/single_channel_one/train.tsv`, init checkpoint `fairseq/base_libri_official.pt` for the “pretrained” pair, `--ckpt none` for random.
- **W&B project:** `spectralfm-runai-recon-signal-100k` (override with `WANDB_PROJECT=...`).
- **tmux:** `tmux ls` → sessions like `sfm_<UTC>_fe_rand`, `_fe_lib`, `_tr_rand`, `_tr_lib`; attach with `tmux attach -t <name>` (detach: `Ctrl-b` `d`).
- **Pointer file** listing output dirs (example stamp): `autoencoder_experiments/LAST_RECON_SIGNAL_4WAY_20260511_151415Z.txt`.

Same launcher **without** tmux (nohup + `wait`): omit `USE_TMUX=1` (script kills existing `train_reconstruction.py` first).

## Key code / config touched

| Area | Path |
|------|------|
| Standalone recon train + analyze | `code/train_reconstruction.py` |
| 4-way launcher | `scripts/launch_recon_signal_4x100k.sh` |
| Fairseq recon RunAI submit | `fairseq/submit_recon_loss_experiments.sh`, `fairseq/submit_recon_decoder_experiments.sh` |
| Data2Vec audio model + Hydra | `fairseq/examples/data2vec/models/data2vec_audio.py`, `spectralfm_full_train.yaml`, `spectralfm_base.yaml`, `recon_loss/spectralfm_recon_loss.yaml`, `recon_loss/HOW_TO_RUN.md` |
| Autoencoder doc | `docs/AUTOENCODER_EXPERIMENTS.md` |

### `train_reconstruction.py` (high level)

- **Transformer backbone:** always build SpectralFM `Data2VecAudioModel`, merge checkpoint tensors like Fairseq `build_model` (not full `load_model_ensemble` for wrong `_name`). Optional remap `modality_encoders.AUDIO.local_encoder.*` → `feature_extractor` with **shape-safe** load for `base_libri_official.pt`.
- **FE path:** `build_fe_standalone` can load same multi `local_encoder` keys (shape-filtered).
- **CLI:** `--wandb_run_name`, `--run_suffix` (unique tags / W&B display), default `--lr 1e-4`, single-GPU pin `_configure_single_gpu`.
- **Logging interval:** `log_interval = steps // 20` → for 100k steps, file logs every **5000** steps; W&B logs more often.

## RunAI

Cluster jobs: `runai list jobs` (project `raja`). The **four tmux 100k jobs above are local**, not RunAI, unless you submit equivalent jobs yourself.

## Known gotchas

- **Duplicate `python ... train_reconstruction.py` processes** with the same `--out_dir` have been observed—keep **one** process per directory.
- **`base_libri_official.pt`** is not a full structural match to SpectralFM FE; only **compatible-shaped** tensors load; rest stays at init.
