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

```bash
python code/eval_fe_decoder.py \
  --checkpoint checkpoints/runai/recon_only_l1/ \
  --device cuda \
  --output_dir code/eval_results/fe_decoder_recon_only
```
