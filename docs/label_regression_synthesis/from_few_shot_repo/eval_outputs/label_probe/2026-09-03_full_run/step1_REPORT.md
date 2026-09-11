# Step 1 — Four-Stage Baseline

Checkpoint: `/mnt5/noy/SpectralFM/checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt`  
n=4716 spectra, comps=(0, 1, 2, 3, 8, 9, 14, 15, 27, 28, 30, 31)

**Input ceiling (best linear, 12-comp): R²=0.9885**


## Stage table (best-of-panel R², recovery ratio vs ceiling)

| n-comp | Raw input | Input (z-scored) | Post-FE (pre-LN) | Post-FE (post-LN) | Post-Projection | Post-Transformer |
|---|---|---|---|---|---|---|
| 1 | 0.409 (41%) | 0.468 (47%) | 0.538 (54%) | 0.567 (57%) | 0.538 (54%) | 0.628 (63%) |
| 2 | 0.831 (84%) | 0.757 (77%) | 0.753 (76%) | 0.762 (77%) | 0.754 (76%) | 0.757 (77%) |
| 3 | 0.867 (88%) | 0.814 (82%) | 0.791 (80%) | 0.789 (80%) | 0.789 (80%) | 0.793 (80%) |
| 7 | 0.935 (95%) | 0.924 (93%) | 0.896 (91%) | 0.896 (91%) | 0.893 (90%) | 0.866 (88%) |
| 12 | 0.988 (100%) | 0.980 (99%) | 0.954 (97%) | 0.952 (96%) | 0.949 (96%) | 0.939 (95%) |

## Legacy row (n=1000, seed=42, KFold(5,shuffle=False), RidgeCV)

| n-comp | raw input | z-scored input | transformer |
|---|---|---|---|
| 1 | 0.3691 | 0.3772 | 0.4412 |
| 2 | 0.7649 | 0.7076 | 0.4754 |
| 3 | 0.8235 | 0.7565 | 0.5062 |

Historical recorded values: 0.3772 / 0.7060 / 0.7568 (z-scored input, 1/2/3-comp).


## Verdict on the T6 headline claim

- 1-comp transformer R²=0.6276 vs input (z-scored, best probe)=0.4684 vs input (raw, best probe)=0.4089
- Embedding beats z-scored input: **True**
- Embedding beats raw input: **True**
