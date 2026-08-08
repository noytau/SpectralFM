# Files to copy to RunAI (`/storage`) for recon loss + epoch cosine maps

Repo root = `fairseq/` (the directory that contains `fairseq_cli/`, `fairseq/`, `examples/`).

Replace `<REPO>` with:
- **Local:** `/mnt5/noy/SpectralFM/fairseq`
- **RunAI PVC:** `/storage/noy/SpectralFM/fairseq`

Copy **local → `/storage`** (same relative path under each root).

---

## Required (epoch cosine + Hydra config)

| Relative path (under `fairseq/`) |
|----------------------------------|
| `examples/data2vec/models/data2vec_audio.py` |
| `examples/data2vec/cosim_epoch_utils.py` |
| `fairseq/tasks/audio_pretraining.py` |
| `examples/data2vec/config/audio/pretraining/recon_loss/spectralfm_recon_loss.yaml` |
| `data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy` |

---

## Recommended (training loop calls `task.end_epoch`)

| Relative path (under `fairseq/`) |
|----------------------------------|
| `fairseq_cli/train.py` |

---

## Example: `kubectl cp` from workstation (paths are local SRC)

```bash
REPO=/mnt5/noy/SpectralFM/fairseq
POD=spectralfm-shell-0-6
NS=runai-raja
DST=/storage/noy/SpectralFM/fairseq

for f in \
  examples/data2vec/models/data2vec_audio.py \
  examples/data2vec/cosim_epoch_utils.py \
  fairseq/tasks/audio_pretraining.py \
  fairseq_cli/train.py \
  examples/data2vec/config/audio/pretraining/recon_loss/spectralfm_recon_loss.yaml \
  data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy
do
  kubectl exec -n "$NS" "$POD" -- mkdir -p "$(dirname "$DST/$f")"
  kubectl cp "$REPO/$f" "${POD}:${DST}/$f" -n "$NS"
done
```

Or use `fairseq/copy_epoch_cosim_subset_to_shell.sh` for the `.npy` only, then copy the rest the same way.

---

## Preflight check (before `submit_recon_loss_experiments.sh`)

From repo `fairseq/` on the **RunAI shell** (or locally with `FAIRSEQ_ROOT`):

```bash
cd /storage/noy/SpectralFM/fairseq   # or: cd /mnt5/noy/SpectralFM/fairseq
./check_runai_recon_epoch_cosim_ready.sh
# Local workstation:
# FAIRSEQ_ROOT=/mnt5/noy/SpectralFM/fairseq ./check_runai_recon_epoch_cosim_ready.sh
```

Exit code **0** = all checks passed; **1** = fix listed items then rerun.

## Manual verify on the pod

```bash
ls -la /storage/noy/SpectralFM/fairseq/examples/data2vec/cosim_epoch_utils.py
ls -la /storage/noy/SpectralFM/fairseq/data/nova_data/metadata/epoch_cosim/structured_similarity_single_channel_all.npy
grep -q epoch_cosim_enable /storage/noy/SpectralFM/fairseq/examples/data2vec/config/audio/pretraining/recon_loss/spectralfm_recon_loss.yaml && echo yaml ok
grep -q maybe_run_epoch_cosim /storage/noy/SpectralFM/fairseq/fairseq/tasks/audio_pretraining.py && echo audio_pretraining ok
```
