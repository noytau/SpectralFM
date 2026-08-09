# Reconstruction Loss — How to Run

## Quick Start: `recon_only` mode (FE reconstruction only)

The default YAML (`spectralfm_recon_loss.yaml`) runs **FE-only reconstruction**:
- Transformer, EMA, masking, and regression loss are all **skipped**
- Only the CNN feature extractor and `fe_recon_decoder` are trainable
- Loss: L1 (configurable to L2 via CLI)

### Local (Geoffrey server)

```bash
cd /mnt5/noy/SpectralFM/fairseq

python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/recon_loss \
  --config-name spectralfm_recon_loss
```

### RunAI

```bash
bash submit_recon_loss_experiments.sh
```

The submit script overrides `/mnt5/...` paths to `/storage/...` for the PVC.

## CLI Overrides

### Switch loss to L2
```bash
model.recon_loss_type=l2
```

### Change learning rate
```bash
optimization.lr='[0.0001]'
```

### Enable epoch cosine heatmaps
```bash
model.epoch_cosim_enable=true
```

### Change dataset
```bash
task.data=/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_all
```

### Override WandB run name
```bash
common.wandb_run_name=my_experiment_name
```

### Reduce checkpoint frequency (save disk)
```bash
checkpoint.no_epoch_checkpoints=true \
checkpoint.keep_interval_updates=1 \
checkpoint.save_interval_updates=5000
```

### FE→transformer projection (`post_extract_proj`)

SpectralFM maps CNN width **512** → transformer width **768** with `post_extract_proj` in
`examples/data2vec/models/data2vec_audio.py` (`build_post_extract_proj`, used from `__init__`).
Same role as wav2vec2’s conditional `post_extract_proj` in `fairseq/models/wav2vec/wav2vec2.py`
when `embed != encoder_embed_dim`.

- **Linear (default):** single `Linear(512, 768)`.
- **MLP + GELU:** two linear layers with a hidden width (default 1536).

```bash
model.post_extract_proj_type=mlp_gelu \
model.post_extract_proj_mlp_hidden=2048
```

When `model.recon_only=true`, the forward path returns after FE reconstruction and **does not**
run `post_extract_proj`; the scalars `post_extract_*` are then absent from `net_output` (criterion
skips missing `log_keys`). For full data2vec training (`spectralfm_full_train.yaml`), they are
logged every step if listed under `criterion.log_keys`.

## What `recon_only` does

When `model.recon_only=true`:

1. **`__init__`**: Sets `final_proj = None` (prevents EMA creation), freezes all parameters except `feature_extractor` and `fe_recon_decoder`
2. **`forward()`**: Runs FE → layer_norm → mean_pool → `fe_recon_decoder` → L1/L2 loss, then returns immediately (no transformer forward pass)
3. **`set_num_updates()`**: Skips EMA teacher creation

### Trainable parameters

| Component | Params | Trainable |
|-----------|--------|-----------|
| Feature Extractor (5 conv layers) | ~3.8M | Yes |
| fe_recon_decoder (MLP 512→512→245) | ~265K | Yes |
| Transformer encoder (12 layers) | ~85M | No (frozen) |
| trans_recon_decoder | ~460K | No (frozen) |
| EMA teacher | — | Not created |

## Comparison: `train_only_fe` vs `recon_only`

| Aspect | `train_only_fe` | `recon_only` |
|--------|-----------------|--------------|
| FE trainable | Yes | Yes |
| Transformer runs | Yes (frozen forward) | **No** (skipped) |
| EMA teacher | Created & updated | **Not created** |
| Masking | Applied | **Skipped** |
| Regression loss | Computed | **Skipped** |
| Recon loss | Optional (lambda > 0) | **Primary loss** |
| Loss function | Configurable (L1/L2) | Configurable (L1/L2) |
| Compute | ~3× slower (transformer + EMA) | Fast (FE only) |

## Evaluation

Current option — the `signal_reconstruction` eval in `code/eval/` (zero-fairseq,
handles FE/projection/transformer pathways alike):
```bash
python -m eval.runner --evals signal_reconstruction --recon_fe_ckpt <ckpt.pt> ...
```
See `code/eval/EVAL_OVERVIEW.md` for the full invocation. The older, standalone
`code/eval_fe_decoder.py` also still works if you need it specifically:
```bash
python code/eval_fe_decoder.py \
  --checkpoint checkpoints/runai/recon_only_l1/ \
  --device cuda \
  --output_dir code/eval_results/fe_decoder_recon_only
```

---

## RunAI: working directory (don't run this from repo root)

The trainer is `fairseq/fairseq_cli/hydra_train.py`, not `SpectralFM/fairseq_cli/...`.
Always:
```bash
cd /storage/noy/SpectralFM/fairseq
python fairseq_cli/hydra_train.py --config-dir examples/data2vec/config/...
```
Running from `/storage/noy/SpectralFM` (repo root) fails — Python looks for a
non-existent `fairseq_cli/` next to it.

## RunAI: syncing files to the PVC (epoch-cosine-similarity + recon_loss)

If `model.epoch_cosim_enable=true` or you're running `spectralfm_recon_loss.yaml`
fresh on a PVC that hasn't seen this code yet, these files must exist under
`/storage/noy/SpectralFM/fairseq/` (same relative paths as local):

- `examples/data2vec/models/data2vec_audio.py`
- `examples/data2vec/cosim_epoch_utils.py`
- `fairseq/tasks/audio_pretraining.py`
- `examples/data2vec/config/audio/pretraining/recon_loss/spectralfm_recon_loss.yaml`
- `data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy`
- `fairseq_cli/train.py` (recommended — the training loop calls `task.end_epoch`)

Copy with `kubectl cp <local path> <pod>:<pvc path> -n <namespace>` per file (or use
`fairseq/copy_epoch_cosim_subset_to_shell.sh` for the `.npy`). **Preflight check**
before submitting — run from `fairseq/` on the RunAI shell (or locally with
`FAIRSEQ_ROOT=...`):
```bash
./check_runai_recon_epoch_cosim_ready.sh
```
Exit 0 = all checks passed; exit 1 = fix the listed items and rerun.

## RunAI: pulling results back

`artifacts/` at the repo root is gitignored — a landing spot for `kubectl cp` pulls
(PNGs, logs) so they stay local without polluting git:
```bash
DST=/mnt5/noy/SpectralFM/artifacts/runai_pull/<run_name>
kubectl cp "<ns>/<pod>:/storage/noy/SpectralFM/fairseq/outputs/recon_loss/<run>/hydra_train.log" "$DST/hydra_train.log"
kubectl cp "<ns>/<pod>:/storage/noy/SpectralFM/fairseq/outputs/recon_loss/<run>/.hydra" "$DST/.hydra"
```
Don't pull the full `checkpoint_last.pt` (~1.5 GB) unless you actually need it.

**Tracking experiments:** `submit_recon_loss_experiments.sh` lists all job names and
overrides (source of truth — `git pull` on the PVC to update what operators run).
WandB project `spectralfm_recon_loss`, run names `Exp1`–`Exp4`. `runai list jobs | grep sfm-recon` to see them on the cluster.
