# Step 2 — XGBoost vs Ridge

Paired delta (xgb − ridge), identical folds, 5 repeats. `tie` = |delta| < 2×delta_sd.

| n-comp | Raw input | Input (z-scored) | Post-FE (pre-LN) | Post-FE (post-LN) | Post-Projection | Post-Transformer |
|---|---|---|---|---|---|---|
| 1 | +0.001 (tie) | -0.094 | +0.057 | -0.005 (tie) | +0.015 | -0.042 |
| 2 | +0.173 | -0.092 | -0.019 | -0.057 | -0.031 | -0.090 |
| 3 | +0.080 | -0.137 | -0.059 | -0.103 | -0.078 | -0.127 |
| 7 | +0.122 | -0.240 | -0.109 | -0.171 | -0.145 | -0.181 |
| 12 | -0.049 | -0.234 | -0.207 | -0.248 | -0.238 | -0.298 |

Positive = XGBoost beats Ridge at that stage/config.
