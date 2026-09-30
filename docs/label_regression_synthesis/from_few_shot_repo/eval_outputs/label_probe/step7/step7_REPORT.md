# Step 7 — Where does the embedding overtake raw input?

Pre-registered rule and estimator: `docs/superpowers/specs/2026-09-09-label-efficiency-crossover-design.md` §2. Readouts are re-selected per rung on `eval_a`; every number below is measured on the disjoint **`eval_b`** with independent draws.


## Run configuration

- **Component count(s) laddered: 1.**
  2-component was **not** laddered: step 1's large-n anchor (n ≈ 3773) already shows raw ahead there (embedding +0.757 vs raw +0.831), so no crossing is possible to find in that regime.
  3-component was **not** laddered: step 1's large-n anchor (n ≈ 3773) already shows raw ahead there (embedding +0.793 vs raw +0.867), so no crossing is possible to find in that regime.
- **Rung(s) 20, 50 were not (re-)measured here** — they are already published cells from the prior step-6 study, cited below as reference rows rather than re-measured.
- **The selection pass (which picks the readout per rung on `eval_a`) ran at a reduced budget** (not recorded in this results file — taken from the run's launch configuration, see `TASKS.md` T15): 10 draws per rung, ranking only the top 3 embedding and top 2 raw candidates (plus the historical arm, never trimmed). The reporting pass (the table below) always kept the full per-rung draw budget (`draws_by_n`) and reported whichever candidate that reduced selection pass picked. Selection only has to rank candidates; reporting has to estimate a value, which needs more draws.

## 1-component

**No crossing on this ladder** — raw input leads at every rung up to n = 2000. Largest median ΔR² was -0.113 at n = 2000; the crossing, if any, lies above the ladder.

| n_train | draws | emb R² | raw R² | historical R² | ΔR² | boot P(Δ>0) | won | emb readout | emb probe | raw probe | reliability (emb frac+/fail, raw frac+/fail, dropped pairs) |
|---:|---:|---:|---:|---:|---:|---:|:--:|---|---|---|---|
| 100 | 60 | +0.268 | +0.762 | +0.092 | -0.487 | 0.00 | no | `emb|layer2|segment4|whiten` | `krr_rbf` | `gp_rbf` | 0.98/0, 1.00/0, 0 |
| 200 | 40 | +0.484 | +0.827 | +0.242 | -0.341 | 0.00 | no | `emb|layer0|segment4|whiten` | `krr_rbf` | `gp_rbf` | 1.00/0, 1.00/0, 0 |
| 500 | 25 | +0.652 | +0.877 | +0.364 | -0.226 | 0.00 | no | `emb|layer1|segment4|whiten` | `krr_rbf` | `gp_rbf` | 1.00/0, 1.00/0, 0 |
| 1000 | 15 | +0.695 | +0.848 | +0.498 | -0.151 | 0.00 | no | `emb|layer1|segment4|whiten` | `ridgecv` | `ridgecv` | 1.00/0, 1.00/0, 0 |
| 2000 | 10 | +0.739 | +0.853 | +0.603 | -0.113 | 0.00 | no | `emb|layer1|segment4|whiten` | `ridgecv` | `ridgecv` | 1.00/0, 1.00/0, 0 |

Step-1 anchor at n ≈ 3773 (pooled 5-fold, different protocol, plotted not scored): embedding +0.628 vs raw +0.468.

**Only 3 of 6 embedding candidates were ever ranked during selection** (`sel_top_emb=3`), and the winning embedding readout changed 2 time(s) across the 5 rungs on this ladder (layer2 → layer0 → layer1) — i.e. the pick sat on the boundary of the trimmed pool rather than settling on one clear winner. "The best embedding readout stays behind raw" is therefore bounded by this 3-candidate pool, not verified against the full panel at every rung.

**No unwhitened raw readout was ever measured at n ≥ 100 on this ladder.** The raw selection pool was trimmed to the top 2 (of 4) step-6-ranked raw candidates (`sel_top_raw=2`), and all of them use `whiten` normalization (`raw|raw|-|whiten`, `raw|z|-|whiten`). The attribution of the study's original premise to an under-whitened raw baseline therefore rests on the *previous* step-6 study's n=20/50 evidence for an unwhitened raw readout, not on a measurement from this ladder.


## How to read this

- **The reported number is also argmax-selected, on `eval_b` itself.** The *readout* (which representation) is chosen per rung on `eval_a`, as stated above — but for whichever readout wins, the reported R² is the maximum over every probe in that rung's panel scored on `eval_b` (9-12 probes over 10-60 draws). That is a second, unadvertised selection step on the same set the numbers are reported from. It inflates *both* arms — embedding and raw always see the identical panel and the identical argmax procedure at a given rung — so it cannot flip the sign of the embedding-vs-raw comparison, but it does mean the columns below are not unbiased held-out estimates of any single probe's performance.
- **Draws above n = 1000 overlap heavily.** They are drawn without replacement from a 3,716-spectrum pool, so two draws at n = 2000 share ~54% of their rows. The spread at the top two rungs therefore understates true sampling variance and their bootstrap intervals are optimistically narrow.
- **The probe panel varies with n** — `ridge_strong`'s alpha grid is biased for a ~20-label fit, a rung not run on this ladder (n = 20, 50 are cited from step 6, not re-measured here); it is likely already over-regularized by the smallest rung actually measured (n = 100) and drops from the panel entirely at n ≥ 1000. Exact GP and the kernel-ridge grid search are cubic and are dropped there too. Both arms always get the identical panel at a given rung, so the confound is between rungs, not between arms.
- **The historical column is `layer12`/mean**, the tap every prior measurement of this backbone used. Its gap to the embedding column is how much of the story was the readout rather than the regime.
- `parameter_0` is a 168-level designed grid, pre-standardized (mean 0.000, std 0.9999). R² is measured against the most favourable target variance it can have; the crossing *n* is more transportable than the R² values either side of it.
- Exact-label overlap: 100% of `eval_b` labels also occur in the pool, and a 20-label draw matches the exact label of ~15% of `eval_b` rows (~33% at n = 50). This inflates both arms and is not corrected here.
