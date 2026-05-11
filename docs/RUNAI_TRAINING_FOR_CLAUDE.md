# RunAI training in SpectralFM — reference for assistants

This file summarizes how **RunAI** (GPU cluster with PVC at `/storage`) is used for **Hydra / fairseq** training in this repo. It is meant for humans and for AI assistants (e.g. Claude) so they do not have to rediscover paths and workflows from scattered scripts.

---

## 1. Path layout: RunAI vs local (Geoffrey)

| Role | RunAI (cluster PVC) | Local (Geoffrey / dev) |
|------|---------------------|-------------------------|
| Project prefix | `/storage/noy` | `/mnt5/noy` |
| Fairseq + code | `/storage/noy/SpectralFM/fairseq` | `/mnt5/noy/SpectralFM/fairseq` |
| Nova audio data | `/storage/noy/SpectralFM/fairseq/data/nova_data/` | `/mnt5/noy/SpectralFM/fairseq/data/nova_data/` |
| Exported checkpoints (canonical copy target) | `/storage/noy/SpectralFM/checkpoints/runai/` | Same logical tree under `/mnt5/...` after sync |

Submit scripts set `FAIRSEQ_ROOT="/storage/noy/SpectralFM/fairseq"` and mount PVC as `--existing-pvc "claimname=storage,path=/storage"` (image and PVC names may vary; operators copy from a working shell job via `runai describe job ...`).

**Trainer entrypoint:** always run from **inside** `fairseq/`:

```bash
cd /storage/noy/SpectralFM/fairseq
python fairseq_cli/hydra_train.py --config-dir ... --config-name ...
```

Running from repo root `/storage/noy/SpectralFM` fails because `fairseq_cli/` lives under `fairseq/`, not next to the repo root (see `fairseq/.../recon_loss/RUNAI_ARTIFACTS.md`).

---

## 2. Critical NFS behavior: `/storage` vs `/mnt5`

`/storage` (RunAI) and `/mnt5` (local) are **different mounts** of related storage. Files written on the cluster under `/storage/...` are **not** guaranteed to appear immediately at the same path under `/mnt5/...`.

**Reliable checkpoint export:** use `kubectl exec` into a **running** shell pod on the cluster and `cp` within `/storage`, or follow the scripted pipe pattern in `.cursor/rules/copy-runai-checkpoints.mdc` (also summarized below).

---

## 3. Copying checkpoints off long-running outputs

- **Shell pod:** `kubectl get pods | grep spectralfm-shell` → e.g. `spectralfm-shell-0-5`.
- **Batch copy:** pipe a bash script to `kubectl exec -i <pod> -- bash` (avoids awkward multi-line `kubectl exec -c` usage). Example logic:
  - Scan Hydra output dirs under something like `/storage/noy/SpectralFM/fairseq/outputs/...`.
  - Prefer `checkpoints/checkpoint_best.pt` if validation ran; with `dataset.disable_validation=true` there is **no** `checkpoint_best.pt` — use latest `checkpoint_<step>.pt` or `checkpoint_last.pt`.
  - Name exports using `wandb_run_name` from `hydra_train.log` plus run folder timestamp: `<wandb_run_name>_<YYYY-MM-DD>_<HH-MM-SS>.pt`.
- **Destination directory on PVC:** `/storage/noy/SpectralFM/checkpoints/runai/` (mirrors to `/mnt5/noy/SpectralFM/checkpoints/runai/` when NFS has synced).
- **Checkpoint sizes:** full checkpoints ~1.4 GB; **frozen-encoder** runs omit much optimizer state → ~730 MB.
- Re-run copy jobs after long runs finish (e.g. 20k steps) so interim checkpoints get replaced by final ones.

Full command template: `.cursor/rules/copy-runai-checkpoints.mdc`.

---

## 4. RunAI submit scripts (under `fairseq/`)

All use **`runai submit`** with a Docker **image** (commonly `noyhassid/spectralfm-lean:v6`), **`--gpu-memory 40G`**, **`--node-pools faculty,raja`**, conda env **`spectralfm`**, and `conda run -n spectralfm python fairseq_cli/hydra_train.py ...`.

| Script | Purpose |
|--------|---------|
| `submit_collapse_experiments.sh` | **FE vs transformer collapse** ablations (`fe_vs_transformer_collapse` / `spectralfm_collapse_ablation`). Short (1k) and long (20k) runs; dataset often `single_channel_one`; comments document FE-identity `max_sample_size=47` for fair memory vs FE-train. |
| `submit_variance_experiments.sh` | **Variance / uniformity** regularization long runs (20k), same config family as collapse. |
| `submit_recon_loss_experiments.sh` | **Reconstruction loss** experiments (`recon_loss/spectralfm_recon_loss.yaml`). Jobs `sfm-recon-exp1-fe`, `exp2`, `exp3` (optional exp4 commented). WandB project **`spectralfm_recon_loss`**. |
| `submit_recon_decoder_experiments.sh` | Recon **decoder architecture** grid (~1k steps each). |

Common **monitoring:** `watch -n 30 'runai list jobs'`, `runai logs -f <job-name>`, `runai delete job <name>` before resubmit if stale.

---

## 5. Reconstruction loss + epoch cosine on RunAI

- **Config:** `fairseq/examples/data2vec/config/audio/pretraining/recon_loss/spectralfm_recon_loss.yaml` — comments note RunAI overrides for `task.data`, `common.user_dir`, `hydra.run.dir` under `/storage`.
- **How to run locally vs RunAI:** `fairseq/examples/data2vec/config/audio/pretraining/recon_loss/HOW_TO_RUN.md` — RunAI section is `bash submit_recon_loss_experiments.sh` (script remaps `/mnt5` → `/storage` as needed).
- **Files that must exist on PVC** for recon + epoch cosine: `fairseq/examples/data2vec/config/audio/pretraining/recon_loss/RUNAI_SYNC_PATHS.md` (Python modules, YAML, `structured_similarity_single_channel_all.npy`, etc.).
- **Preflight:** `fairseq/check_runai_recon_epoch_cosim_ready.sh` from shell or locally with `FAIRSEQ_ROOT=...`.
- **Small artifact pulls:** `RUNAI_ARTIFACTS.md` (cosim PNGs, `hydra_train.log`, `.hydra` — avoid copying full 1.5 GB checkpoints unless necessary).
- **Cursor plan** (authoritative for recon work): `/home/noy/.cursor/plans/reconstruction_loss_experiments_1ec3ff93.plan.md` (referenced from `.cursor/rules/efficiency-tips.mdc`).

---

## 6. Full-train style config (WandB naming)

`fairseq/examples/data2vec/config/audio/pretraining/spectralfm_full_train.yaml` sets `common.user_dir` and data under `/storage/...`, `wandb_project: spectralfm_runai_full_train`, and checkpoint intervals tuned for long runs.

---

## 7. Environment detection and path translation (training / data)

`fairseq/fairseq/tasks/audio_pretraining.py` implements:

- **`detect_current_environment()`** — RunAI vs **Geoffrey**: GPU names (A5000/A6000 vs RTX 2080), Geoffrey IP prefix `132.66.52.*`, else presence of `/mnt5/noy` vs `/storage/noy`.
- **`translate_data_path()` / checkpoint-origin helpers** — map `/storage/noy` ↔ `/mnt5/noy` when loading data or evaluating across machines.

This keeps datasets and configs consistent when moving checkpoints between cluster and lab servers.

---

## 8. Evaluation and tooling after RunAI training

- **Checkpoint roots on disk:** `/mnt5/noy/SpectralFM/checkpoints/runai/` — flat date folders, per-run names, grouped dirs like `2026-03-10-compare-single-to-multi/`, `recon_loss_experiment_*`, etc. (`datasets-and-checkpoints.mdc`, `code/docs/EVALUATION_FLOW.md`).
- **`code/model_loader.py`** — when loading fairseq checkpoints locally, remaps model paths that start with `/storage/noy` to `/mnt5/noy` if the local file exists; can set `skip_pretrained_weights` if base weights are absent locally but finetuned weights are inside the checkpoint.
- **Eval docs** with RunAI examples: `code/HOW_TO_EVALUATE.md`, `code/docs/EVALUATION_FLOW.md` (path remapping, `eval_fe_decoder.py` / `evaluation_runner.py` `--checkpoint_dir` patterns).

---

## 9. Data prep and copying utilities

- **`fairseq/scripts/convert_all_datasets_runai.py`** — converts Nova datasets on the server; auto-detects `/storage` vs `/mnt5`.
- **`scripts/create_and_copy_subset.py`** — creates subsets and can push to RunAI via `runai submit` / `kubectl cp` / `runai exec` (namespace e.g. `runai-raja`).
- **`fairseq/copy_epoch_cosim_subset_to_shell.sh`** — copies the epoch-cosim `.npy` into a shell pod.
- **`fairseq/copy_checkpoints.sh`** — related checkpoint staging under `/storage/noy/SpectralFM/checkpoints/runai`.

---

## 10. Operational checklist (short)

1. Code + data on PVC under `/storage/noy/SpectralFM/fairseq/...`.
2. `cd /storage/noy/SpectralFM/fairseq` before `hydra_train.py`.
3. Image + PVC + `WANDB_API_KEY` (if using WandB) configured like existing jobs.
4. Submit via the appropriate `submit_*.sh` or manual `runai submit`.
5. After training, copy checkpoints with **in-cluster** `cp`/`kubectl exec` if `/mnt5` visibility is delayed.
6. For local eval, rely on `/mnt5/.../checkpoints/runai/` and path remapping in `model_loader` / task code.

---

## 11. Further reading (in-repo)

| Topic | Location |
|-------|----------|
| Checkpoint copy recipe | `.cursor/rules/copy-runai-checkpoints.mdc` |
| Dataset sizes and `--eval_data_dir` | `.cursor/rules/datasets-and-checkpoints.mdc` |
| Recon Run / sync / artifacts | `fairseq/examples/data2vec/config/audio/pretraining/recon_loss/RUNAI_SYNC_PATHS.md`, `RUNAI_ARTIFACTS.md`, `HOW_TO_RUN.md` |
| Eval flow and RunAI checkpoint layouts | `code/docs/EVALUATION_FLOW.md` |

---

*Generated as a consolidated assistant-oriented note; update this file when RunAI image names, namespaces, or PVC claim names change.*
