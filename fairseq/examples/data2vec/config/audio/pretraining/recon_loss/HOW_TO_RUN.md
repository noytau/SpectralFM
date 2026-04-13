# Reconstruction loss experiments

**Defaults are in `spectralfm_recon_loss.yaml`** (RunAI `/storage/...` paths). Submit script only sets WandB name and experiment-specific knobs (`lambda_recon_trans`, `freeze_encoder`).

**Batching:** `dataset.batch_size=512`, `optimization.update_freq=[4]` → **effective batch = 512 × 4** samples per optimizer step (single GPU).

## Paths

| Environment | Override on CLI |
|-------------|-----------------|
| **RunAI** | None (YAML already uses `/storage/noy/SpectralFM/...`) |
| **Local** | `task.data`, `common.user_dir`, `model.epoch_cosim_subset_path`, `model.model_path`, `hydra.run.dir` → `/mnt5/noy/SpectralFM/...` |

## Cosine similarity maps (what actually runs)

When `model.epoch_cosim_enable=true` and `model.epoch_cosim_subset_path` points to a valid file:

1. **When:** If `model.epoch_cosim_interval_updates` **> 0** (default **1000**), rank 0 runs after each successful `train_step` when `num_updates % N == 0`. If **0**, the same pipeline runs **once per training epoch** in `task.end_epoch` instead (no mid-training maps).
2. **Data:** Load capped train indices from the subset file (see below); batch them with `epoch_cosim_micro_batch`; collate from `task.dataset("train")`.
3. **Forward:** `model.eval()` + `torch.no_grad()`; `extract_cosim_epoch_features` → pooled **input**, **FE**, and **transformer embedding** vectors.
4. **Metrics:** `sklearn.metrics.pairwise.cosine_similarity` on each representation; mean/std of upper triangle per panel.
5. **Artifacts:** PNG `step_{updates:07d}_cosim_triple.png` (interval mode) or `epoch_{epoch:03d}_cosim_triple.png` (epoch-only mode) under `checkpoint.save_dir` / `epoch_cosim_output_subdir`; WandB `epoch_cosim/*` and image at `trainer.get_num_updates()` step.

Implementation: `examples/data2vec/cosim_epoch_utils.py` (logic), `fairseq_cli/train.py` (interval hook), `fairseq/tasks/audio_pretraining.py` (`maybe_run_epoch_cosim_on_interval` + `end_epoch` when interval is 0).

## Epoch cosine subset (precomputed structured panel)

This is **not** “all of train” — it is the **structured similarity subset** from `eval_utils.build_structured_similarity_subset` (e.g. 3×10 stack samples from `single_channel_all`), **filtered to `task.data` and remapped** to indices in the **train** `FileAudioDataset` (`structured_subset_epoch_cosim_train_indices`). The old filename `*_train.npy` only meant “train-split indices,” which was easy to misread.

**Canonical file** (local and RunAI — same relative path under the repo root):

| Environment | Path |
|-------------|------|
| Local (`/mnt5`) | `/mnt5/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy` |
| RunAI (`/storage`) | `/storage/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy` |

Training expects this file when `model.epoch_cosim_enable` is true. If it is missing, create it **once** (or copy from a machine that already ran the command):

```bash
# Local: create directory and write ~30 train indices for single_channel_all
mkdir -p /mnt5/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim

cd /mnt5/noy/SpectralFM/code
python precompute_epoch_cosim_indices.py \
  --nova_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data \
  --task_data /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_all \
  --min_sample_size 200 \
  --prefer_manifest train \
  --out /mnt5/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy
```

You should see `[+] Wrote 30 indices to ...`. Verify:

```bash
ls -la /mnt5/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy
```

**RunAI — get the `.npy` onto `/storage`**

1. **Avoid `scp` from inside the shell pod** (`scp noy@132.66…:/mnt5/... /storage/...`): that pulls from your workstation over SSH **from the cluster**. It often fails (firewall, SSH not reachable from cluster IPs, no keys). Copy **from** the machine that has the file instead, or use `kubectl cp` below.

2. **One-liner script** (from repo, with `kubectl` configured):

   ```bash
   cd /mnt5/noy/SpectralFM/fairseq
   ./copy_epoch_cosim_subset_to_shell.sh
   ```

   Auto-picks the first **Running** pod whose name contains `spectralfm-shell`. Or pass namespace and pod explicitly:

   ```bash
   ./copy_epoch_cosim_subset_to_shell.sh runai-raja spectralfm-shell-0-6
   ```

3. **Manual `kubectl cp`** from your dev machine (same paths):

   ```bash
   POD=spectralfm-shell-0-6          # your shell pod name
   NS=runai-raja                     # your namespace if not default
   DST=/storage/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy
   SRC=/mnt5/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy

   kubectl exec -n "$NS" "$POD" -- mkdir -p "$(dirname "$DST")"
   kubectl cp "$SRC" "${POD}:${DST}" -n "$NS"
   kubectl exec -n "$NS" "$POD" -- ls -la "$DST"
   ```

4. **Or generate on the cluster** (no copy): in a pod with `/storage` mounted and `nova_data` present, run the same `precompute_epoch_cosim_indices.py` command with `--nova_data_dir` / `--task_data` / `--out` under `/storage/noy/SpectralFM/fairseq/...`.

## Local smoke run (short; paths + Hydra output on `/mnt5`)

```bash
cd /mnt5/noy/SpectralFM/fairseq
python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/recon_loss \
  --config-name spectralfm_recon_loss \
  task.data=/mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_all \
  common.user_dir=/mnt5/noy/SpectralFM/fairseq/examples/ \
  model.model_path=/mnt5/noy/SpectralFM/fairseq/base_libri_official.pt \
  model.epoch_cosim_subset_path=/mnt5/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy \
  hydra.run.dir=/mnt5/noy/SpectralFM/fairseq/outputs/recon_loss/smoke \
  optimization.max_update=200
```

## Experiment matrix (submit only adds WandB name + one or two overrides)

| Job | Extra overrides |
|-----|------------------|
| Exp 1 | (YAML default: λ_fe=1, λ_trans=0, train encoder) |
| Exp 2 | `model.lambda_recon_trans=1.0` |
| Exp 3 | `model.freeze_encoder=true` |

## RunAI

```bash
cd /mnt5/noy/SpectralFM/fairseq
bash submit_recon_loss_experiments.sh
```

## WandB

Project: `spectralfm_recon_loss`. Watch `loss_recon_fe`, `loss_recon_trans`, `loss`, `epoch_cosim/*`.
