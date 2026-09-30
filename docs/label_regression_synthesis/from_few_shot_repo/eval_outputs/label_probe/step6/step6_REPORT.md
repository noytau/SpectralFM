# Step 6 — Readout & geometry sweep for few-shot label efficiency

**Verdict: NO** (0/6 cells won under the rule pre-registered in `docs/superpowers/specs/2026-09-09-few-shot-readout-geometry-design.md` §2).

Screened 456 readouts on `eval_a` (30 draws) to pick a finalist set (3 embedding readouts, 2 raw readouts per cell -- an asymmetry that if anything favours the embedding). The **per-cell result** table below is an argmax over those finalists x probes, computed on the disjoint **`eval_b`** (100 draws), so the finalist set never saw the data it is scored on; the argmax itself is symmetric across both arms and run identically for embedding and raw. The later sections on stage/pooling/normalizer choice and the diagnostic-correlation table are `eval_a` **screen** numbers, not `eval_b` confirm numbers.


## Per-cell result (eval_b, paired draws)

| n_comp | n_train | embedding readout | probe | emb R² | raw readout | probe | raw R² | ΔR² | boot P(Δ>0) | emb frac>0 | won | n_pairs | n_dropped |
|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|:--:|---:|---:|
| 1 | 20 | `emb|layer0|segment4|whiten` | `ridge_strong` | +0.001 | `raw|raw|-|whiten` | `ridge_strong` | +0.249 | -0.237 | 0.00 | 0.51 | no | 100 | 0 |
| 1 | 50 | `emb|layer0|segment4|whiten` | `ridge_strong` | +0.119 | `raw|raw|-|whiten` | `gp_rbf` | +0.582 | -0.459 | 0.00 | 0.93 | no | 100 | 0 |
| 2 | 20 | `emb|layer3|segment4|standardize` | `pca10_ridge` | +0.025 | `raw|moments|-|whiten` | `ridge_strong` | +0.306 | -0.288 | 0.00 | 0.58 | no | 100 | 0 |
| 2 | 50 | `emb|layer3|segment4|standardize` | `ridge_strong` | +0.239 | `raw|raw|-|standardize` | `ridge_strong` | +0.589 | -0.358 | 0.00 | 0.93 | no | 100 | 0 |
| 3 | 20 | `emb|layer1|mean_std|standardize` | `tabpfn_pca50` | +0.015 | `raw|moments|-|whiten` | `ridge_strong` | +0.298 | -0.296 | 0.00 | 0.58 | no | 100 | 0 |
| 3 | 50 | `emb|layer1|segment4|none` | `ridge_strong` | +0.197 | `raw|raw|-|standardize` | `ridge_strong` | +0.674 | -0.488 | 0.00 | 0.90 | no | 100 | 0 |

## Which hypothesis moved the number (eval_a screen numbers)

- **1-comp** — best stages by screen score: layer1, layer3, layer0, layer2.
- **2-comp** — best stages by screen score: layer1, layer3, layer2, layer0.
- **3-comp** — best stages by screen score: fe, layer1, layer2, layer0.
- **Normalizer (H-C)** — best screen R² per normalizer: `none` +0.250, `standardize` +0.249, `l2` +0.178, `whiten` +0.144.
- **Pooling (H-A)** — best screen R² per pooling: `segment4` +0.250, `mean` +0.246, `mean_std` +0.242, `mean_max_min` +0.202, `mean_std_max_min` +0.201, `first_last` +0.138.

## Do the cheap diagnostics predict few-shot performance? (eval_a screen numbers)

| diagnostic | Spearman ρ vs screen R² | p | n readouts |
|---|---:|---:|---:|
| `whitened_topk_r2_20` | +0.631 | 4.9e-52 | 456 |
| `whitened_topk_r2_10` | +0.626 | 4.8e-51 | 456 |
| `whitened_topk_r2_50` | +0.626 | 6.2e-51 | 456 |
| `whitened_topk_r2_5` | +0.579 | 4e-42 | 456 |
| `ridgecv_full_r2` | +0.338 | 1.1e-13 | 456 |
| `effective_rank` | +0.142 | 0.0024 | 456 |
| `participation_ratio` | +0.109 | 0.02 | 456 |

## Scope

- One checkpoint (Feb-25 SSL), components (0,1,2), frozen backbone, random label draws.
- Layer and pooling have no raw-input analogue; the raw family gets the normalizer, reduction and full probe panel instead. That asymmetry favours the embedding.
- A NO verdict bounds this readout family on this backbone; it does not prove no readout exists.
