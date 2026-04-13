# RunAI artifacts — copy to dev machine

`artifacts/` at repo root is **gitignored**; use it to store `kubectl cp` pulls so PNGs/logs stay local.

## Working directory for `hydra_train.py`

The trainer is **`fairseq/fairseq_cli/hydra_train.py`**, not `SpectralFM/fairseq_cli/...`. Always:

```bash
cd /storage/noy/SpectralFM/fairseq
python fairseq_cli/hydra_train.py --config-dir examples/data2vec/config/...
```

If you run from `/storage/noy/SpectralFM` (repo root), Python will look for a non-existent `fairseq_cli/` next to it and fail with “No such file or directory”.

## Pull smoke / verification outputs from the shell pod

Replace `<NS>` / `<POD>` (e.g. `runai-raja`, `spectralfm-shell-0-6`).

```bash
DST=/mnt5/noy/SpectralFM/artifacts/runai_pull/smoke_verify
mkdir -p "$DST/cosim_epoch"

kubectl cp "<NS>/<POD>:/storage/noy/SpectralFM/fairseq/outputs/recon_loss/smoke_verify_pipeline/checkpoints/cosim_epoch/." \
  "$DST/cosim_epoch/"

kubectl cp "<NS>/<POD>:/storage/noy/SpectralFM/fairseq/outputs/recon_loss/smoke_verify_pipeline/hydra_train.log" \
  "$DST/hydra_train.log"

kubectl cp "<NS>/<POD>:/storage/noy/SpectralFM/fairseq/outputs/recon_loss/smoke_verify_pipeline/.hydra" \
  "$DST/.hydra"
```

Do **not** copy full `checkpoint_last.pt` unless you need it (~1.5 GB).

## Tracking experiments on RunAI

- **Git:** `submit_recon_loss_experiments.sh` and `HOW_TO_RUN.md` list all job names and overrides; `git pull` on `/storage/noy/SpectralFM` updates what operators run.
- **WandB:** project `spectralfm_recon_loss`; each job sets `common.wandb_run_name=...` (Exp1–Exp4).
- **Cluster:** `runai list jobs | grep sfm-recon` to see `sfm-recon-exp1-fe` … `sfm-recon-exp4-train-fe-only`.
