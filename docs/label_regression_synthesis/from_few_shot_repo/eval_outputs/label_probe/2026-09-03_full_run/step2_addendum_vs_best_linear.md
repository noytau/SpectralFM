# Step 2 addendum — XGBoost vs the BEST linear probe (not fixed Ridge(1.0))

| n-comp | input_raw | input_z | fe | extract_features | proj | transformer |
|---|---|---|---|---|---|---|
| 1 | -0.202 | -0.276 | -0.280 | -0.309 | -0.275 | -0.359 |
| 2 | -0.165 | -0.254 | -0.249 | -0.258 | -0.244 | -0.299 |
| 3 | -0.210 | -0.303 | -0.282 | -0.279 | -0.275 | -0.329 |
| 7 | -0.168 | -0.388 | -0.370 | -0.371 | -0.355 | -0.377 |
| 12 | -0.196 | -0.299 | -0.350 | -0.346 | -0.345 | -0.410 |

Positive = XGBoost beats the best-of-panel linear probe (OLS/Ridge/RidgeCV) at that stage/config -- the harder, more informative bar.
