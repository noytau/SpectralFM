# Reconstruction 2-AE basemerge — status & how-to

**Branch:** `recon/2ae-basemerge` (off `main`, **not** pushed)
**Commit:** `fdfeea4` — "Add 2-AE reconstruction (head_fe + head_trans) and Hydra per-component LR"
**Summary plot:** `code/eval_results/recon_components_overview.png`

This doc captures everything done in the SpectralFM 2-AE basemerge work so you
can pick it back up cold. Pair it with the plot above (Panels A–D + the 6 per-
configuration strips).

---

## 1.  TL;DR — what's new

Two parallel reconstruction heads on top of the basemerge transformer backbone:

```
            ┌──────────────────────┐       head_trans (existing)
            │ Transformer encoder  │ ────► ConvTranspose MirrorDecoder
            └──────────────────────┘       → recon_trans_loss   (λ_trans)
   FE ─► LN ─┬─► post_extract_proj ─► (transformer above)
             │
             └─► head_fe   (NEW, when λ_fe > 0)
                 ConvTranspose MirrorDecoder
                 → recon_fe_loss   (λ_fe)
```

Loss:
```
argparse  (code/train_reconstruction.py):
    total = λ_trans · recon_trans_loss + λ_fe · recon_fe_loss        (L2)

Hydra  (data2vec_audio + new YAML preset):
    total = regression_loss                                          (always weight 1)
          + λ_recon_trans · recon_trans_loss                         (L2)
          + λ_recon_fe    · recon_fe_loss                            (L2)
```

Per-component LR is now real on both sides:

| group        | members (joint encoder + decoder share a group, per spec)        | default LR |
|--------------|------------------------------------------------------------------|------------|
| `fe`         | `feature_extractor` + `fe_recon_decoder`                         | **1e-5**   |
| `ln`         | `layer_norm`                                                     | **1e-5**   |
| `proj`       | `post_extract_proj`                                              | **1e-4**   |
| `transformer`| `encoder` + `trans_recon_decoder`                                | **1e-5**   |
| `other`      | `mask_emb`, `final_proj`                                         | **1e-4**   |

Argparse path uses `torch.optim.Adam` param-groups; Hydra path uses fairseq's
`optimizer._name: composite` (one Adam + cosine LR scheduler per group).

---

## 2.  Files changed / added

| Path | What |
|---|---|
| `code/train_reconstruction.py` | new λ flags, `head_fe` build, hook-based fe_seq capture, weighted L2 loss, renamed WandB keys, expanded checkpoint format |
| `code/recon_components.py` | `head_trans` + optional `head_fe` everywhere; `load_head_from_ckpt` rewritten for flat state dicts and stem-less heads |
| `fairseq/examples/data2vec/models/data2vec_audio.py` | new `init_*_ckpt` / `freeze_*` / `tag_param_groups` config fields; `_maybe_apply_recon_components` helper; `MirrorReconDecoder` + `_TransMirrorWrap`; new `recon_decoder_type=mirror` wiring; `recon_trans` forward respects `needs_full_sequence` |
| `fairseq/examples/data2vec/config/audio/pretraining/recon_loss/spectralfm_recon_loss_basemerge.yaml` | new Hydra preset: composite optimizer, 5 LR groups, basemerge inits, both λ_recon_* = 1, L2 |
| `fairseq/submit_signal_recon_2ae_runai.sh` | new RunAI submitter for the argparse 2-AE grid |
| `fairseq/submit_signal_recon_basemerge_runai.sh` | (earlier in this work) basemerge transformer-AE submitter |
| `fairseq/copy_signal_recon_grid_ckpts_from_runai.sh` | (earlier in this work) helper to pull RunAI ckpts back to `/mnt5` |

---

## 3.  Argparse path — `code/train_reconstruction.py`

### 3.1  New CLI flags (all back-compat preserved)

```bash
# reconstruction loss weights
--lambda_recon_fe FLOAT          # default: Option B (see §3.3)
--lambda_recon_trans FLOAT       # default: Option B (see §3.3)

# head_fe (the new MirrorDecoder branch)
--init_head_fe_ckpt PATH         # FE-AE decoder dict or transformer_mirror.decoder.*
--freeze_head_fe                 # frozen → λ_fe auto-zeroed
--lr_head_fe FLOAT

# head_trans (was --*_head before; old names still work)
--init_head_trans_ckpt PATH      # alias for legacy --init_head_ckpt
--freeze_head_trans              # alias for legacy --freeze_head
--lr_head_trans FLOAT            # alias for legacy --lr_head
```

### 3.2  Auto-zero rules

`train_reconstruction.py` enforces these with a warning, so the optimizer never
sees a useless loss term:

| condition                                                  | enforced |
|------------------------------------------------------------|----------|
| `head_trans` frozen                                        | `λ_trans = 0` |
| `head_fe` frozen or not built                              | `λ_fe = 0`    |
| both end up 0                                              | **raises** with a clear message |

### 3.3  Default behaviour (Option B)

* `--recon_path transformer`:
  * neither λ flag set → `λ_trans = 1`, `λ_fe = 1` iff `head_fe` is wired (via `--init_head_fe_ckpt` or explicit `--lambda_recon_fe`), else `λ_fe = 0`
  * any λ flag set explicitly → use that value (still subject to auto-zero rules)
* `--recon_path fe`: unchanged — single MirrorDecoder on FE+LN, no transformer.

### 3.4  WandB keys

```
train/loss_total
train/loss                         # back-compat alias of loss_total
train/recon_trans_loss             # logged only when λ_trans > 0
train/recon_fe_loss                # logged only when λ_fe > 0
train/lambda_trans / lambda_fe     # constants per run
train/target_var
train/pred_var_trans / pred_var_fe # variance along spectrogram axis, batch-averaged
train/lr   train/lr/{fe,ln,proj,transformer,head_trans,head_fe}
{name}/grad_norm / param_norm / lr # every 100 steps via per_component_norms
```

### 3.5  Checkpoint format bump (`transformer_ae_format=3`)

In addition to the existing fields:

```python
"fe_mirror":            head_fe.state_dict()              # only when head_fe is enabled
"lambda_recon_trans":   float
"lambda_recon_fe":      float
"head_fe_enabled":      bool
"losses_trans":         list[float]                       # per-step recon_trans loss
"losses_fe":            list[float]                       # per-step recon_fe loss
```

The existing analyze-mode loader (`_load_transformer_mirror_from_ckpt`) ignores
the new fields gracefully — old checkpoints still load.

---

## 4.  Hydra path — `data2vec_audio.py` + new YAML

### 4.1  New `Data2VecAudioConfig` fields (all default `None` / `False`)

```yaml
model:
  # per-component init (paths to .pt files; loaders auto-detect
  # fairseq_audio / data2vec_multi / apr28_fe_recon layouts and
  # remap ViT blocks.X.* → fairseq layers.X.* with QKV split)
  init_fe_ckpt:                  /path/to/apr28_fe_recon_best.pt
  init_ln_ckpt:                  /path/to/apr28_fe_recon_best.pt
  init_proj_ckpt:                null
  init_transformer_ckpt:         /path/to/base_libri_official.pt
  init_fe_recon_decoder_ckpt:    /path/to/apr28_fe_recon_best.pt  # head_fe init
  init_trans_recon_decoder_ckpt: null

  # per-component freeze
  freeze_fe_v2:               true        # also sets feature_grad_mult=0
  freeze_ln:                  true
  freeze_proj:                false
  freeze_transformer_v2:      false       # (alternative to existing freeze_encoder)
  freeze_fe_recon_decoder:    false
  freeze_trans_recon_decoder: false

  # tag p.param_group for composite optimizer
  tag_param_groups:           true

  # decoder choice — wires MirrorReconDecoder (needs_full_sequence=True)
  recon_decoder_type:         mirror      # mlp | linear | conv1d | interp | flat | mirror
```

`build_model()` calls a new `_maybe_apply_recon_components(model, cfg)` after
the legacy `model_path` load. It uses `code/recon_components.py` for the
shape-safe loaders (lazy import — falls back to a no-op + warning if
`code/` is not on `PYTHONPATH`).

### 4.2  Param-group tagging map (joint encoder + decoder per spec)

```python
_PARAM_GROUP_MAP = {
    "feature_extractor":    "fe",
    "fe_recon_decoder":     "fe",            # joint with FE encoder
    "layer_norm":           "ln",
    "post_extract_proj":    "proj",
    "encoder":              "transformer",
    "trans_recon_decoder":  "transformer",   # joint with transformer encoder
    # anything else → "other"  (mask_emb, final_proj, …)
}
```

Validated: `{fe: 33 tensors, ln: 2, proj: 2, transformer: 219, other: 3}`.

### 4.3  New `MirrorReconDecoder` (in `data2vec_audio.py`)

Same 5-layer `ConvTranspose1d` stack as `code/train_reconstruction.MirrorDecoder`
(245 → 47 conv stack reversed). Sets `needs_full_sequence = True`, so the
existing `if getattr(self.fe_recon_decoder, "needs_full_sequence", False):`
branch in the data2vec forward feeds it the full `fe_seq` (no mean-pool).

`recon_decoder_type=mirror` also swaps `trans_recon_decoder` to a wrapped
`_TransMirrorWrap` (stem 1×1 conv 768→512 + LN → `MirrorReconDecoder`) and the
`recon_trans` forward path now honours `needs_full_sequence` instead of always
mean-pooling.

### 4.4  Composite optimizer YAML pattern

```yaml
optimizer:
  _name: composite
  groups:
    fe:          { lr: [1e-5], optimizer: {_name: adam, ...}, lr_scheduler: {_name: cosine, warmup_updates: 1000, max_update: 100000} }
    ln:          { lr: [1e-5], ... }
    proj:        { lr: [1e-4], ... }
    transformer: { lr: [1e-5], ... }
    other:       { lr: [1e-4], ... }
lr_scheduler: pass_through

optimization:
  lr: [0]              # required global field; composite ignores it
```

Full file: `fairseq/examples/data2vec/config/audio/pretraining/recon_loss/spectralfm_recon_loss_basemerge.yaml`.

---

## 5.  Validation done (both smoke tests passed)

### 5.1  Argparse 2-AE smoke (local RTX 2080 Ti)

```bash
PYTHONPATH=code:fairseq/examples python3 code/train_reconstruction.py \
  --mode train --recon_path transformer --ckpt none \
  --init_fe_ckpt          fairseq/apr28_fe_recon_best.pt \
  --init_ln_ckpt          fairseq/apr28_fe_recon_best.pt \
  --init_transformer_ckpt fairseq/base_libri_official.pt \
  --init_head_fe_ckpt     fairseq/apr28_fe_recon_best.pt \
  --freeze_fe_v2 --freeze_ln \
  --lambda_recon_fe 1.0 --lambda_recon_trans 1.0 \
  --lr 1e-4 --lr_transformer 1e-5 --lr_proj 1e-4 \
  --lr_head_trans 1e-4 --lr_head_fe 1e-4 \
  --data_dir fairseq/data/nova_data/single_channel_100/wav \
  --n_samples 16 --steps 4 --warmup 1 \
  --batch_size 16 --grad_accum_steps 1 \
  --out_dir /tmp/smoke_2ae --device cuda
```

* All 239 trainable parameters received gradients (`grad-audit ok`).
* 4 param groups built with the right LRs (proj 1e-4, transformer 1e-5, head_trans 1e-4, head_fe 1e-4).
* `L2_total = L2_trans + L2_fe` per step.
* With `--init_head_fe_ckpt fairseq/apr28_fe_recon_best.pt`: `L2_fe` started at **0.02** at step 1 (Apr-28 decoder paired with its own Apr-28 FE+LN inputs), confirming the head loader round-trips correctly.

### 5.2  Hydra build_model smoke (synthetic batch)

```python
from omegaconf import OmegaConf, open_dict
from data2vec.models.data2vec_audio import Data2VecAudioModel, Data2VecAudioConfig
sc = OmegaConf.structured(Data2VecAudioConfig)
with open_dict(sc):
    sc.conv_feature_layers = "[(512, 3, 1)]*4 + [(512, 5, 5)]"   # pseudo
    sc.recon_decoder_type = "mirror"
    sc.init_fe_ckpt = "...apr28_fe_recon_best.pt"
    sc.init_ln_ckpt = "...apr28_fe_recon_best.pt"
    sc.init_transformer_ckpt = "...base_libri_official.pt"
    sc.init_fe_recon_decoder_ckpt = "...apr28_fe_recon_best.pt"
    sc.freeze_fe_v2 = True; sc.freeze_ln = True
    sc.tag_param_groups = True
m = Data2VecAudioModel.build_model(sc)
```

Result:
* FE + LN frozen, `feature_grad_mult=0`
* `transformer` 89.78M trainable, `fe_recon_decoder` 3.68M trainable (MirrorReconDecoder, full sequence), `trans_recon_decoder` 4.07M trainable (`_TransMirrorWrap`, full sequence)
* `param_group → tensor counts: {fe: 33, ln: 2, proj: 2, transformer: 219, other: 3}`
* `recon_only` forward returns `losses = {'recon_fe': tensor(19.11)}` on random input — sanity check that the decoder runs end-to-end on `fe_seq` instead of mean-pooled features.

---

## 6.  How to launch the experiments

### 6.1  Argparse 2-AE on RunAI (no Hydra)

```bash
bash fairseq/submit_signal_recon_2ae_runai.sh
```

Submits 2 jobs:
* `sfm-sr2ae-s10k`  — 10 k optimizer steps, 1 k warmup
* `sfm-sr2ae-s100k` — 100 k steps, 10 k warmup

Both run with: FE+LN frozen from Apr-28, transformer from base_libri (8→12 layer
remap), `head_fe` pre-loaded from the Apr-28 decoder, λ_fe = λ_trans = 1.

WandB project (default): `spectralfm-runai-signal-recon-2ae`.

**Prerequisites on the RunAI PVC `/storage/noy/SpectralFM/`:**
* latest `code/` synced (knows the new flags),
* `fairseq/apr28_fe_recon_best.pt` and `fairseq/base_libri_official.pt`,
* manifest `fairseq/data/nova_data/single_channel_one/train_runai_150k.tsv`.

### 6.2  Hydra 2-AE basemerge (new preset)

```bash
cd fairseq
PYTHONPATH=/mnt5/noy/SpectralFM/code \
fairseq-hydra-train \
  --config-dir examples/data2vec/config/audio/pretraining/recon_loss \
  --config-name spectralfm_recon_loss_basemerge
```

`PYTHONPATH` must include `code/` so `data2vec_audio.py` can lazy-import
`recon_components.py` for the per-component loaders. Without it, the
init/freeze/tag flags become no-ops (with a warning) and you get random init.

Pass `model.lambda_recon_fe=0.5 model.lambda_recon_trans=0.5` etc. to sweep λ.
Override LRs per group via `optimizer.groups.transformer.lr=[3e-5]` etc.

---

## 7.  The 6 configurations (from Panel B of the summary plot)

| # | name                          | FE       | LN       | proj    | transformer  | head_fe | head_trans | regression | losses                                  |
|---|-------------------------------|----------|----------|---------|--------------|---------|------------|------------|-----------------------------------------|
| ① | `sfm-sr-fe-*`                 | T Feb-25 | T Feb-25 | —       | —            | T rand  | —          | —          | recon_fe                                |
| ② | `sfm-sr-tr-*`                 | F Feb-25 | F Feb-25 | T rand  | T rand       | —       | T rand     | —          | recon_trans                             |
| ③ | `sfm-srbm-*`   (basemerge)    | F Apr-28 | F Apr-28 | T rand  | T base_libri | —       | T rand     | —          | recon_trans                             |
| ④ | PROPOSED 2-AE  (argparse)     | F Apr-28 | F Apr-28 | T rand  | T base_libri | T Apr-28| T rand     | —          | recon_fe + recon_trans                  |
| ⑤ | Hydra data2vec (baseline)     | T any    | T any    | T any   | T any        | —       | —          | T          | regression                              |
| ⑥ | PROPOSED Hydra 2-AE           | F Apr-28 | F Apr-28 | T rand  | T base_libri | T rand  | T rand     | T          | regression + recon_fe + recon_trans     |

Legend: **T** trainable, **F** frozen. Init source after the slash.

---

## 8.  Status & open items

**Done**

* Branch `recon/2ae-basemerge` created off `main`, all changes committed (`fdfeea4`).
* Argparse: 2-AE forward, weighted L2 loss, auto-zero rules, new WandB keys, per-component LR groups, audit, checkpoint bump.
* Hydra: new config fields, lazy `recon_components` loader, param-group tagging, `MirrorReconDecoder`, `recon_decoder_type=mirror`, composite-optimizer YAML preset.
* RunAI submitter for the argparse 2-AE grid.
* Both smoke tests green.

**Not done (deliberately, until you say go)**

* `git push` — branch is local only.
* RunAI submission — `submit_signal_recon_2ae_runai.sh` is ready but not invoked.
* Hydra preset has not been launched end-to-end on RunAI yet; the smoke test exercises `build_model` + `recon_only` forward but not a full `fairseq-hydra-train` loop with composite optimizer.
* `head_fe` is currently a separate component in the argparse path; if you eventually want to fold the FE-only path's `decoder` under the same name for symmetry, that's a follow-up rename.

---

## 9.  Quick recovery checklist when you come back

1. `git checkout recon/2ae-basemerge` and `git log --stat -1` — confirm you're on commit `fdfeea4`.
2. Open `code/eval_results/recon_components_overview.png` — Panels A–D + the 6 strips show the whole design at a glance.
3. Skim §3.1 (argparse flags) and §4.1 (Hydra fields) — those are the API surface.
4. If you want to run an experiment:
   * Argparse on RunAI → §6.1.
   * Hydra locally / on RunAI → §6.2.
5. If something looks off in a checkpoint, the new metadata (`lambda_recon_*`, `head_fe_enabled`, `losses_trans`, `losses_fe`, `init_manifest`, `freeze_map`, `audit_gradient_flow`) is right there to diagnose it.
