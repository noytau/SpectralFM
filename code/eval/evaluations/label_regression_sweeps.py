"""
Label Regression Sweeps — checkpoint side-by-side comparison views
--------------------------------------------------------------------
Two grid-style comparisons matching the legacy label_reg_evaluation.py plots
(_plot_train_size / _plot_n_components): a single LBFGS-trained linear probe
(not the RidgeCV cross-val probe used by label_regression.py) evaluated at
fixed train/eval splits, so results can be laid out as [config] x [Raw input +
one column per checkpoint].

  run_train_size_sweep — rows = train size (default 100/1000/2000), fixed
                          2-comp, eval=1000
  run_component_sweep  — rows = component count 1/2/3, fixed train size
                          (default 1000), eval=1000

Both take `checkpoints` = [(label, model, path), ...] (the same structure
CheckpointLoader.load_multiple / checkpoint_comparison.run use) plus
`labeled_data_dir`, and return a dict consumed by
report._plot_label_reg_sweep.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

from ..data_loader import load_labeled_data
from .label_regression import _normalize_like_fairseq, _embed


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return 1.0 - ((y - yhat) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


_WD_CANDIDATES = (0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def _fit_linear_once(X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, seed: int,
                      max_iter: int, weight_decay: float) -> np.ndarray:
    """LBFGS-trained single Linear layer on standardized labels, with an
    optional L2 penalty on the weights. weight_decay=0 matches the legacy
    label_reg_evaluation.py::_fit_linear exactly (convex problem, so the seed
    only affects init, not the converged optimum)."""
    mu, sigma = y_tr.mean(), y_tr.std() + 1e-8
    y_n = (y_tr - mu) / sigma
    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_n).float().unsqueeze(1)
    Xe = torch.from_numpy(X_ev).float()
    torch.manual_seed(seed)
    probe = nn.Linear(X_tr.shape[1], 1)
    nn.init.xavier_uniform_(probe.weight)
    nn.init.zeros_(probe.bias)
    opt = optim.LBFGS(probe.parameters(), lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")
    loss_fn = nn.MSELoss()

    def closure():
        opt.zero_grad()
        pred = probe(Xt)
        l = loss_fn(pred, yt)
        if weight_decay > 0:
            l = l + weight_decay * probe.weight.pow(2).sum()
        l.backward()
        return l

    probe.train()
    opt.step(closure)
    probe.eval()
    with torch.no_grad():
        p = probe(Xe).squeeze(-1).numpy()
    return p * sigma + mu


def _fit_linear_probe(X_tr: np.ndarray, y_tr: np.ndarray, X_ev: np.ndarray, seed: int = 42,
                       max_iter: int = 300, wd_candidates=_WD_CANDIDATES) -> np.ndarray:
    """
    LBFGS linear probe with an internal 80/20 split to pick an L2 weight_decay
    from wd_candidates (mirrors the RidgeCV alpha search used elsewhere in this
    package). Needed because a plain unregularized fit (weight_decay=0) actively
    OVERFITS once feature dim approaches/exceeds n_train (768-2304-dim
    embeddings vs typical train sizes of 100-2000): letting LBFGS run to
    convergence on those columns makes held-out R2 get WORSE, not better,
    while low-dim raw-input columns (245-735-dim) are unaffected/best at
    weight_decay=0. Confirmed empirically 2026-08-04 (see TASKS.md/eval notes).
    """
    n_tr = len(y_tr)
    if n_tr < 20 or len(wd_candidates) == 1:
        return _fit_linear_once(X_tr, y_tr, X_ev, seed, max_iter, wd_candidates[0])

    n_inner = max(10, int(n_tr * 0.8))
    X_in, y_in = X_tr[:n_inner], y_tr[:n_inner]
    X_val, y_val = X_tr[n_inner:], y_tr[n_inner:]

    best_wd, best_mse = wd_candidates[0], float("inf")
    for wd in wd_candidates:
        pred_val = _fit_linear_once(X_in, y_in, X_val, seed, max_iter, wd)
        mse = float(np.mean((y_val - pred_val) ** 2))
        if mse < best_mse:
            best_mse, best_wd = mse, wd

    return _fit_linear_once(X_tr, y_tr, X_ev, seed, max_iter, best_wd)


def _get_split(X: np.ndarray, y: np.ndarray, rng_idx: np.ndarray, n_train: int, n_eval: int):
    Xs, ys = X[rng_idx], y[rng_idx]
    return Xs[:n_train], ys[:n_train], Xs[n_train:n_train + n_eval], ys[n_train:n_train + n_eval]


def _cell_metrics(y_ev: np.ndarray, preds: np.ndarray) -> dict:
    pr, _ = pearsonr(y_ev, preds)
    return {
        "r2": float(_r2(y_ev, preds)),
        "pearson_r": float(pr),
        "mae": float(np.mean(np.abs(y_ev - preds))),
    }


def _load_and_embed(
    checkpoints: list, labeled_data_dir: str, device: str,
    max_samples: int, seed: int, comps: tuple, batch_size: int = 64,
) -> tuple:
    """Load raw + per-checkpoint mean-pooled embeddings for one component set.
    Returns (y, columns) where columns = [("Raw input", X_raw), (label, emb), ...]."""
    inputs, y = load_labeled_data(labeled_data_dir, max_samples=max_samples, seed=seed, comps=comps)
    n, n_comps, L = inputs.shape
    X_flat = _normalize_like_fairseq(inputs.reshape(n * n_comps, L))
    X_raw = X_flat.reshape(n, n_comps * L)

    columns = [("Raw input", X_raw)]
    for item in checkpoints:
        label, model = item[0], item[1]
        emb_flat = _embed(model, X_flat, device, batch_size=batch_size)
        emb = emb_flat.reshape(n, n_comps * emb_flat.shape[-1])
        columns.append((label, emb))
    return y, columns


def run_train_size_sweep(
    checkpoints: list,
    labeled_data_dir: str,
    device: str = "cpu",
    train_sizes: list = None,
    n_eval: int = 1000,
    comps: tuple = (0, 1),
    seed: int = 42,
    batch_size: int = 64,
) -> dict:
    """
    Rows = train_sizes, cols = Raw input + one column per checkpoint (comps
    concatenated, e.g. comps=(0,1) -> 490-dim raw / 1536-dim embedding).

    Returns {"row_values", "columns", "cells", "sweep", "comps", "n_eval"} where
    cells maps (row, col) -> {y_true, y_pred, y_train, col_label, row_label, r2,
    pearson_r, mae}.
    """
    if train_sizes is None:
        train_sizes = [100, 1000, 2000]
    max_samples = max(train_sizes) + n_eval + 200
    y, columns = _load_and_embed(checkpoints, labeled_data_dir, device, max_samples, seed, comps, batch_size)
    rng_idx = np.random.default_rng(seed).permutation(len(y))

    cells = {}
    for row, n_tr in enumerate(train_sizes):
        for col, (col_label, arr) in enumerate(columns):
            X_tr, y_tr, X_ev, y_ev = _get_split(arr, y, rng_idx, n_tr, n_eval)
            preds = _fit_linear_probe(X_tr, y_tr, X_ev, seed=seed)
            m = _cell_metrics(y_ev, preds)
            cells[(row, col)] = {
                "y_true": y_ev, "y_pred": preds, "y_train": y_tr,
                "col_label": col_label, "row_label": f"train={n_tr}", **m,
            }
            print(f"[LabelRegSweep:train_size] train={n_tr:5d}  {col_label:<28} "
                  f"R2={m['r2']:+.4f}  r={m['pearson_r']:.3f}  MAE={m['mae']:.3f}")

    return {
        "row_values": train_sizes, "columns": [c[0] for c in columns], "cells": cells,
        "sweep": "train_size", "comps": comps, "n_eval": n_eval,
    }


def run_component_sweep(
    checkpoints: list,
    labeled_data_dir: str,
    device: str = "cpu",
    comp_configs: list = None,
    train_size: int = 1000,
    n_eval: int = 1000,
    seed: int = 42,
    batch_size: int = 64,
) -> dict:
    """
    Rows = component configs (default 1/2/3-comp), cols = Raw input + one
    column per checkpoint. Each row reloads data at its own dimensionality
    (245/490/735-dim raw, 768/1536/2304-dim embedding).
    """
    if comp_configs is None:
        comp_configs = [(0,), (0, 1), (0, 1, 2)]
    max_samples = train_size + n_eval + 200

    cells = {}
    row_labels = []
    columns_labels = None
    for row, comps in enumerate(comp_configs):
        y, columns = _load_and_embed(checkpoints, labeled_data_dir, device, max_samples, seed, comps, batch_size)
        if columns_labels is None:
            columns_labels = [c[0] for c in columns]
        rng_idx = np.random.default_rng(seed).permutation(len(y))
        row_label = f"{len(comps)}-comp (C{','.join(map(str, comps))})"
        row_labels.append(row_label)
        for col, (col_label, arr) in enumerate(columns):
            X_tr, y_tr, X_ev, y_ev = _get_split(arr, y, rng_idx, train_size, n_eval)
            preds = _fit_linear_probe(X_tr, y_tr, X_ev, seed=seed)
            m = _cell_metrics(y_ev, preds)
            cells[(row, col)] = {
                "y_true": y_ev, "y_pred": preds, "y_train": y_tr,
                "col_label": col_label, "row_label": row_label, **m,
            }
            print(f"[LabelRegSweep:n_components] {row_label:<20} {col_label:<28} "
                  f"R2={m['r2']:+.4f}  r={m['pearson_r']:.3f}  MAE={m['mae']:.3f}")

    return {
        "row_values": row_labels, "columns": columns_labels, "cells": cells,
        "sweep": "n_components", "train_size": train_size, "n_eval": n_eval,
    }
