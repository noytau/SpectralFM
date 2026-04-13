# fe_vs_transformer_collapse — How to Run

All experiments use a single config file `spectralfm_collapse_ablation.yaml` with CLI overrides.
Run each command from the **fairseq root**:

```bash
cd /mnt5/noy/SpectralFM/fairseq
```

WandB run names are auto-composed as:
`fe-{fe_mode}_trans-frozen-{freeze_encoder}_lv{lambda_var}_lc{lambda_cov}`

Outputs land in:
`outputs/fe_vs_transformer_collapse/fe-{fe_mode}_trans-frozen-{freeze_encoder}_lv{lambda_var}_lc{lambda_cov}/{timestamp}/`

---

## Experiment Matrix

| # | Exp ID | Purpose |
|---|--------|---------|
| 1 | `fe-identity_trans-train_base` | Baseline: can transformer collapse alone? |
| 2 | `fe-identity_trans-train_var` | Does variance loss prevent transformer collapse? |
| 3 | `fe-identity_trans-train_vicreg` | Full VICReg regularization on transformer-only |
| 4 | `fe-frozen_trans-train_base` | Does frozen (pretrained) FE preserve input structure? |
| 5 | `fe-frozen_trans-train_var` | Frozen FE + variance regularization |
| 6 | `fe-frozen_trans-train_vicreg` | Frozen FE + full VICReg |
| 7 | `fe-train_trans-frozen_base` | Does FE collapse on its own? |
| 8 | `fe-train_trans-train_var` | Full training + variance (**main candidate**) |
| 9 | `fe-train_trans-train_vicreg` | Full training + full VICReg |

---

## Commands

### 1 — fe-identity_trans-train_base
Identity FE (no CNN), transformer trains freely, base data2vec loss only.

```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse \
  --config-name spectralfm_collapse_ablation \
  model.fe_mode=identity \
  model.freeze_encoder=false \
  model.lambda_var=0.0 \
  model.lambda_cov=0.0
```

---

### 2 — fe-identity_trans-train_var
Identity FE, transformer trains, + variance regularization.

```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse \
  --config-name spectralfm_collapse_ablation \
  model.fe_mode=identity \
  model.freeze_encoder=false \
  model.lambda_var=1.0 \
  model.lambda_cov=0.0
```

---

### 3 — fe-identity_trans-train_vicreg
Identity FE, transformer trains, + VICReg (variance + covariance).

```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse \
  --config-name spectralfm_collapse_ablation \
  model.fe_mode=identity \
  model.freeze_encoder=false \
  model.lambda_var=25.0 \
  model.lambda_cov=1.0
```

---

### 4 — fe-frozen_trans-train_base
FE frozen with **base_libri pretrained weights**, transformer trains, base loss only.

```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse \
  --config-name spectralfm_collapse_ablation \
  model.fe_mode=frozen \
  model.freeze_encoder=false \
  model.lambda_var=0.0 \
  model.lambda_cov=0.0 \
  model.model_path=/mnt5/noy/SpectralFM/fairseq/base_libri_official.pt \
  model.skip_pretrained_weights=false
```

---

### 5 — fe-frozen_trans-train_var
FE frozen (pretrained), transformer trains, + variance regularization.

```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse \
  --config-name spectralfm_collapse_ablation \
  model.fe_mode=frozen \
  model.freeze_encoder=false \
  model.lambda_var=1.0 \
  model.lambda_cov=0.0 \
  model.model_path=/mnt5/noy/SpectralFM/fairseq/base_libri_official.pt \
  model.skip_pretrained_weights=false
```

---

### 6 — fe-frozen_trans-train_vicreg
FE frozen (pretrained), transformer trains, + VICReg.

```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse \
  --config-name spectralfm_collapse_ablation \
  model.fe_mode=frozen \
  model.freeze_encoder=false \
  model.lambda_var=25.0 \
  model.lambda_cov=1.0 \
  model.model_path=/mnt5/noy/SpectralFM/fairseq/base_libri_official.pt \
  model.skip_pretrained_weights=false
```

---

### 7 — fe-train_trans-frozen_base
FE trains freely, transformer frozen (random init), base loss only.
Answers: does the FE collapse on its own?

```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse \
  --config-name spectralfm_collapse_ablation \
  model.fe_mode=train \
  model.freeze_encoder=true \
  model.lambda_var=0.0 \
  model.lambda_cov=0.0
```

---

### 8 — fe-train_trans-train_var ⭐ main candidate
Full training (FE + transformer), + variance regularization.

```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse \
  --config-name spectralfm_collapse_ablation \
  model.fe_mode=train \
  model.freeze_encoder=false \
  model.lambda_var=1.0 \
  model.lambda_cov=0.0
```

---

### 9 — fe-train_trans-train_vicreg
Full training (FE + transformer), + full VICReg regularization.

```bash
python fairseq_cli/hydra_train.py \
  --config-dir examples/data2vec/config/audio/pretraining/fe_vs_transformer_collapse \
  --config-name spectralfm_collapse_ablation \
  model.fe_mode=train \
  model.freeze_encoder=false \
  model.lambda_var=25.0 \
  model.lambda_cov=1.0
```

---

## Collapse Monitoring

Watch these keys in the WandB logs:

| Metric | Healthy | Warning | Collapsed |
|--------|---------|---------|-----------|
| `target_var` | > 1.0 | 0.3–1.0 | < 0.1 |
| `pred_var` | > 0.5 | 0.1–0.5 | < 0.01 |
| `var` (loss) | decreasing | flat | — |
| `cov` (loss) | decreasing | flat | — |

If `target_var` drops below `min_target_var` (0.1) after 5k updates, training aborts automatically.

## Loss Weight Reference

| Config | `lambda_var` | `lambda_cov` | `lambda_uniform` | `var_gamma` |
|--------|-------------|-------------|-----------------|------------|
| `base` | 0 | 0 | 0 | — |
| `var` | 1.0 | 0 | 0 | 1.0 |
| `vicreg` | 25.0 | 1.0 | 0 | 1.0 |
| `uniform` | 0 | 0 | 1.0 | — |

If variance loss dominates early training, reduce `lambda_var` to 0.1–0.5.
If covariance loss dominates, reduce `lambda_cov` to 0.1.

## Notes

- **Frozen FE** always uses `/mnt5/noy/SpectralFM/fairseq/base_libri_official.pt` (Librispeech pretrained weights). Only `feature_extractor.*` keys are loaded; the transformer is randomly initialised.
- **Identity FE** replaces the CNN with a single linear projection `Linear(1 → 512)`. Sequence length is preserved (no striding), so the transformer sees longer sequences than normal.
- **Frozen encoder** freezes all `self.encoder` parameters. The EMA teacher is unaffected and still receives momentum updates from the encoder's current state (which is fixed).
- Outputs are written to `outputs/fe_vs_transformer_collapse/` relative to the fairseq root.
