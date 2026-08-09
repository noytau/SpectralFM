# SpectralFM — Training Architecture Reference

Technical map of what trains today: model code, config, loss, checkpoint
formats. For setup and exact launch commands, see
[`CLAUDE.md`](CLAUDE.md#getting-started-new-teammate-setup) — this doc is the
"how it works and why" companion to that "how to run it" quick-start. Eval
side: [`code/eval/EVAL_OVERVIEW.md`](code/eval/EVAL_OVERVIEW.md).

## Two things get trained here

| | Entry point | Trains |
|---|---|---|
| **Backbone (regular) training** | `fairseq_cli.hydra_train` + `spectralfm_base.yaml`, launched by `sweep_dataset.sh` | The `data2vec_audio` backbone itself, self-supervised, no labels |
| **Autoencoder / reconstruction training** | `code/train_reconstruction.py` (standalone), or the same `hydra_train` entry point with `recon_loss/*.yaml` configs | One or more decoder heads that reconstruct the original 245-point signal from a (frozen or trainable) backbone |

Both use the same underlying model class — fairseq's `data2vec_audio`
(`fairseq/examples/data2vec/models/data2vec_audio.py`) — just configured
differently. There is also an older, HuggingFace-based training path
(`code/model_loader.py`'s `load_custom_data2vec_audio_model`,
`code/run_experiment.py`) that predates both of the above and is not part of
either flow described here; it's slated for removal (see `TASKS.md`).

---

## 1. Model architecture

### Feature extractor

A 5-layer 1D conv stack, explicitly sized for 245-frame input (245 → 243 →
241 → 239 → 237 → **47** tokens):

```
conv_feature_layers = [(512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 5, 5)]
```

512 channels, `layer_norm` mode, no conv bias. This value must be set
explicitly everywhere the model is built — the class default
(`Wav2Vec2Config.conv_feature_layers`) is the stock wav2vec2/speech stack
(~320× downsampling), which collapses a 245-frame input to a degenerate
~0-length sequence. `spectralfm_base.yaml`, `spectralfm_full_train.yaml`, and
`code/train_reconstruction.py`'s `_FE_CONV_LAYERS` constant all set the same
5-layer value independently — if you write a new config from scratch, copy it
from one of those rather than relying on the default.

### Backbone transformer

12-layer transformer encoder, `encoder_embed_dim=768`, EMA teacher for
self-distillation (see Loss setup below). `post_extract_proj` projects the
512-d conv output up to 768-d — either a bare `nn.Linear` (`post_extract_proj_type: linear`,
the default) or a small MLP (`post_extract_proj_type: mlp_gelu`, hidden dim
via `post_extract_proj_mlp_hidden`) when a config asks for it.

### `train_only_fe` — read this before writing a new config

`Data2VecAudioConfig.train_only_fe` **defaults to `true`**, which calls
`freeze_all_except_feature_extractor()` in `build_model()` — every parameter
except the conv feature extractor gets `requires_grad=False`. This exists to
support FE-only reconstruction experiments (frozen backbone, only the FE and
its decoder train). `spectralfm_base.yaml` explicitly sets it to `false` for
full backbone training. **Any new config for full end-to-end training must do
the same**, or training will silently run as FE-only.

### Reconstruction decoder heads (autoencoder training only)

Up to three decoder heads, each reconstructing the 245-point signal from a
different depth of the backbone, all sharing the same 5-layer
`ConvTranspose1d` "mirror" architecture (47 → 245, reversing the FE):

| Head | Reads from | Wired up by |
|---|---|---|
| FE decoder | conv FE output (post-LayerNorm) | `--recon_path fe` (standalone) or `lambda_recon_fe > 0` (Hydra) |
| Proj decoder | post-`post_extract_proj` features | `lambda_recon_proj > 0` (Hydra only) |
| Transformer decoder | full transformer output | `--recon_path transformer` (standalone) or `lambda_recon_trans > 0` (Hydra) |

Each decoder is a stem (a 1×1 conv when the input is wider than 512, e.g. the
768-d transformer output) followed by the same `MirrorReconDecoder`/`MirrorDecoder`
stack: `ConvTranspose1d(512,512,k=5,s=5)` then four `ConvTranspose1d(512,512,k=3,s=1)`
stages (each `LayerNorm`+`GELU`), ending in a final projection to 1 channel —
47 → 237 → 239 → 241 → 243 → 245. Parameter count: ~3.68M encoder-side FE,
~3.68M per decoder head.

Per-component **init** (`--init_fe_ckpt`, `--init_transformer_ckpt`, …, or
the Hydra equivalents `model.init_fe_ckpt`, …) and **freeze**
(`--freeze_fe_v2`, `--freeze_transformer`, …) flags let any of FE / LayerNorm
/ proj / transformer / each decoder head be independently warm-started from a
different checkpoint and independently frozen — this is how "reconstruction
heads on top of a frozen, already-trained SSL backbone" experiments get
built. On the Hydra side there's also a composite optimizer
(`optimizer._name: composite`) with one param group + LR per component
(`fe`, `ln`, `proj`, `transformer`, `other`), driven by `tag_param_groups: true`
tagging each parameter's `param_group` attribute in `build_model()`.

---

## 2. Loss setup

**Backbone (SSL) training:** standard data2vec self-distillation, computed
inside `Data2VecAudioModel` itself — fairseq's `model` criterion just reads
whatever loss dict the model reports, there's no separate criterion file.
Masked-student output is regressed against the EMA teacher's average of the
top `average_top_k_layers=8` transformer layers, target-normalized
(`instance_norm_target_layer: true`). `loss_beta: 0` ⇒ plain L2/MSE.

**Reconstruction training:** a weighted sum of up to three independent
per-head MSE losses against the original signal — `lambda_recon_fe`,
`lambda_recon_trans`, `lambda_recon_proj` (any can be zero to disable that
head's loss without removing the head, e.g. to monitor a frozen head's
reconstruction quality via `--monitor_recon_fe`). The FE head additionally
supports a total-variation smoothness penalty:

```
TV(x) = mean(|x[:, 1:] - x[:, :-1]|)
loss  = λ_fe·(MSE(head_fe) + λ_tv_fe·TV(head_fe))  +  λ_trans·MSE(head_trans)  +  λ_proj·MSE(head_proj)
```

Best setting found so far: `lambda_tv_fe=0.1` (mild regularizer; higher
over-smooths — see `TASKS.md`'s T4 results).

**Known finding, not yet reflected in defaults:** reconstruction quality and
downstream representation quality are decoupled — the original SSL-pretrained
backbone remains the only one that's label-informative (R²=0.44 on label
regression); every reconstruction-trained variant tried so far scores ≈0.
Prefer adding decoders on top of a frozen SSL backbone over training through
it, until that finding changes. See `TASKS.md`'s T6 write-up.

---

## 3. Data sources

`.tsv` manifests (`train.tsv`/`valid.tsv`) over `.wav` directories, built by
`fairseq/create_manifests.py` — see `CLAUDE.md`'s
[Manifest generation](CLAUDE.md#manifest-generation) for the exact command;
manifest generation is a one-time per-subset step, independent of any given
training run. Dataset subsets and which are wired into training vs. eval-only
are documented in `CLAUDE.md`'s [Datasets](CLAUDE.md#datasets) section.

---

## 4. Checkpoint loading

Three different mechanisms exist, because they solve different problems —
picking the wrong one for the job is the most common way to get a silently
wrong (not crashing) result:

1. **Warm-starting one component while building a new model**
   (`code/recon_components.py`, used by both `train_reconstruction.py`'s
   `--init_*_ckpt` flags and Hydra's `model.init_*_ckpt` fields). Each loader
   (`load_fe_from_ckpt`, `load_transformer_from_ckpt`, `load_head_from_ckpt`,
   …) auto-detects which of several known checkpoint layouts the source `.pt`
   uses — a plain fairseq audio checkpoint, `data2vec_multi`'s layout, or the
   older standalone `apr28_fe_recon`-style save — and remaps state-dict keys
   accordingly (e.g. ViT-style `blocks.X.*` → fairseq's `layers.X.*` with QKV
   split, when the source is a different model family). This is lazy-imported
   from `data2vec_audio.py`'s `build_model()` — if `code/` isn't on
   `PYTHONPATH` when launching via Hydra, these fields silently no-op with a
   warning instead of failing, and you get random init where you expected a
   warm-start.

2. **Loading a full checkpoint for eval/inference**
   (`code/model_loader.py:load_fairseq_checkpoint`, used by
   `code/evaluation_runner.py` and other `code/eval_*.py` scripts — the
   older, fairseq-dependent eval tooling, not the zero-fairseq `code/eval/`
   package). Uses fairseq's own `checkpoint_utils.load_model_ensemble_and_task`.
   Before loading, it inspects the checkpoint's embedded config for any
   `/storage/noy/...` path (e.g. a `model_path` pointing at the base
   checkpoint this one was warm-started from, or a cosim-subset path) and
   remaps it to `/mnt5/noy/...` when running on Geoffrey — auto-setting
   `skip_pretrained_weights=True` if no local copy of that base checkpoint
   exists, so evaluating a checkpoint doesn't fail just because the original
   warm-start file isn't on your machine. It also backfills three config keys
   that don't exist on older checkpoints (`model_path`,
   `skip_pretrained_weights`, `train_only_fe`, the last defaulting to `False`
   here specifically so old checkpoints evaluate as full models rather than
   silently as FE-only) — so old and new checkpoints load through the same
   code path with the same behavior.

3. **The zero-fairseq eval package** (`code/eval/checkpoint_loader.py`) — a
   separate, from-scratch loader with no relationship to the two above. It
   detects checkpoint format purely from its state-dict keys (`fairseq`,
   `3ae`, `fe_recon`, `tr_recon`) and rebuilds an equivalent HuggingFace
   `Data2VecAudioModel` shell with the matching feature extractor swapped in,
   entirely without a fairseq install. See `code/eval/EVAL_OVERVIEW.md`.

**Worked example of mechanism 2's path remap:** a checkpoint trained on
RunAI has `cfg.model.model_path = "/storage/noy/SpectralFM/checkpoints/runai/base_libri_official.pt"`
baked in (the base checkpoint it was warm-started from). Load that same
checkpoint from Geoffrey with:
```python
from model_loader import load_fairseq_checkpoint
model, cfg, info = load_fairseq_checkpoint(
    "/mnt5/noy/SpectralFM/checkpoints/runai/my_run/checkpoint_best.pt"
)
```
Internally this rewrites `model_path` to
`/mnt5/noy/SpectralFM/checkpoints/runai/base_libri_official.pt` before
loading. If that file happens not to exist locally, it instead sets
`skip_pretrained_weights=True` and loads only the fine-tuned weights already
inside `checkpoint_best.pt` — either way you get a working model without
manually editing the checkpoint's embedded config.

---

## 5. What can be trained — axes of variation

1. **Dataset subset** — `task.data=` (Hydra) or `--data_dir`/`--manifest`
   (standalone); see `CLAUDE.md`'s Datasets section for what's available.
2. **Backbone capacity** — encoder layers/dim/heads, EMA schedule; all
   hydra-overridable, not varied in any sweep run so far.
3. **Which components train vs. freeze** — via `train_only_fe` (backbone-wide)
   or the per-component `freeze_*` flags (reconstruction training only).
4. **Projection head** — `linear` vs. MLP (`post_extract_proj_type` /
   `proj_mlp_hidden_dim`), swept as `linear`/`mlp768`/`mlp2048` in past
   rounds (`TASKS.md` T2).
5. **Reconstruction loss weights** — `lambda_recon_fe/trans/proj`,
   `lambda_tv_fe` — TV swept over {0, 0.01, 0.1, 1.0} in `TASKS.md` T4.
6. **Per-component checkpoint init** — mix-and-match warm starts per
   component, e.g. "SSL-pretrained transformer, frozen; freshly-initialized
   decoder heads." Worked example (full command):
   `CLAUDE.md`'s [Autoencoder / reconstruction training](CLAUDE.md#autoencoder--reconstruction-training) section.
7. **Masking** — `mask_prob`/`mask_length`/`mask_selection` (backbone
   training only; reconstruction training doesn't mask).
8. **LR schedule and steps** — `tri_stage` (yaml default) vs. `cosine` +
   warmup (`sweep_dataset.sh`'s launch-time override); per-component LR
   overrides exist only for reconstruction training.

---

## Train → eval bridge

Once you have a checkpoint (from either training path), evaluate it with
`code/eval/runner.py` — see the quick-run command in `CLAUDE.md`. It works
from any of the three checkpoint formats above without needing to know which
training path produced it.
