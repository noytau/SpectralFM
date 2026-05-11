"""
Label Regression Evaluation — Plot Suite
=========================================
Generates all label_regression diagnostic plots for a given evaluation run.
Saves everything to  {output_dir}/plots/label_reg_plots/

Plots produced:
  1. comparison_linear_vs_mlp.png
  2. comparison_n_components.png
  3. comparison_train_size.png
  4. param0_distributions.png
  5. merged_comp_comparison_1000.png

Usage (standalone):
    python label_reg_evaluation.py --output_dir /path/to/eval_run_dir

Usage (programmatic):
    from label_reg_evaluation import run_label_reg_plots
    run_label_reg_plots(output_dir="/path/to/eval_run_dir")
"""

import os
import re
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from scipy.stats import pearsonr, gaussian_kde
from collections import defaultdict
from typing import Optional

# ── Paths (kept as constants; all data reading is from fixed cache locations) ──
SEED         = 42
N_EVAL       = 1000
DATASET_DIR  = "/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data"
CACHE_BASE   = "/mnt5/noy/fairseq/data/single_channel_1m"
LAYER_CACHE  = CACHE_BASE + "/tasks/{run}/task1_layer_sweep/layer_cache"
PATTERN      = re.compile(r"dataset(\d+)_comp(\d+)_spec_(\d+)\.wav")

_CKPT_DIR = "/mnt5/noy/SpectralFM/checkpoints/runai/recon_loss_experiment_3"

# (checkpoint_path_or_run, display_label, best_layer_or_None)
# When best_layer is None the script extracts embeddings on-the-fly from the .pt file.
CHECKPOINTS = [
    (f"{_CKPT_DIR}/recon-fe1.0_recon-tr0.0_frozen-encFalse_5k.pt",
     "fe-recon\n5k", None),
    (f"{_CKPT_DIR}/recon-fe1.0_recon-tr1.0_frozen-encFalse_5k.pt",
     "fe+tr-recon\n5k", None),
    (f"{_CKPT_DIR}/recon-fe1.0_recon-tr0.0_frozen-encTrue_5k.pt",
     "fe-recon\nfrozen-enc 5k", None),
]
COLORS = ["dimgray", "steelblue", "darkorange", "mediumseagreen"]


def _extract_embeddings(inputs: np.ndarray, checkpoint_path: str) -> np.ndarray:
    """Extract mean-pooled transformer embeddings [N, 768] from a checkpoint."""
    import torch
    import sys, os
    _fairseq = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fairseq")
    if _fairseq not in sys.path:
        sys.path.insert(0, _fairseq)
    from eval_fe_decoder import extract_embeddings_from_inputs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = extract_embeddings_from_inputs(inputs, checkpoint_path, device_str=device, batch_size=64)
    if emb is None:
        raise RuntimeError(f"Failed to extract embeddings from {checkpoint_path}")
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_row_map():
    pairs = []
    with open(f"{DATASET_DIR}/labels.tsv") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) == 2:
                pairs.append((p[0], float(p[1])))
    random.Random(SEED).shuffle(pairs)
    fname_to_valid = {}
    with open(f"{DATASET_DIR}/valid.tsv") as f:
        f.readline()
        for i, line in enumerate(f):
            fname_to_valid[line.strip().split("\t")[0]] = i
    y_cache, spec_comp_to_row = [], {}
    for fname, label in pairs:
        if fname in fname_to_valid:
            row = len(y_cache)
            y_cache.append(label)
            m = PATTERN.match(fname)
            if m:
                ds, comp, spec = int(m.group(1)), int(m.group(2)), int(m.group(3))
                spec_comp_to_row[(ds, spec, comp)] = row
    return np.array(y_cache, dtype=np.float32), spec_comp_to_row


def _build_merged(arr, y_cache, spec_comp_to_row, comps=(0, 1)):
    X, y = [], []
    for (ds, spec, comp), r0 in spec_comp_to_row.items():
        if comp == comps[0] and all((ds, spec, c) in spec_comp_to_row for c in comps):
            vecs = [arr[spec_comp_to_row[(ds, spec, c)]] for c in comps]
            X.append(np.concatenate(vecs))
            y.append(y_cache[r0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _r2(y, yhat):
    return 1.0 - ((y - yhat) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


def _get_split(X, y, rng_idx, n_train, n_eval=N_EVAL):
    Xs, ys = X[rng_idx[:len(X)]], y[rng_idx[:len(y)]]
    return Xs[:n_train], ys[:n_train], Xs[n_train:n_train + n_eval], ys[n_train:n_train + n_eval]


# ─────────────────────────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────────────────────────

def _fit_linear(X_tr, y_tr, X_ev):
    mu, sigma = y_tr.mean(), y_tr.std() + 1e-8
    y_n = (y_tr - mu) / sigma
    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_n).float().unsqueeze(1)
    Xe = torch.from_numpy(X_ev).float()
    probe = nn.Linear(X_tr.shape[1], 1)
    nn.init.xavier_uniform_(probe.weight); nn.init.zeros_(probe.bias)
    opt = optim.LBFGS(probe.parameters(), lr=0.5, max_iter=300,
                      line_search_fn="strong_wolfe")
    loss_fn = nn.MSELoss()
    def closure():
        opt.zero_grad(); l = loss_fn(probe(Xt), yt); l.backward(); return l
    probe.train(); opt.step(closure)
    probe.eval()
    with torch.no_grad():
        p = probe(Xe).squeeze().numpy()
    return p * sigma + mu


def _fit_mlp(X_tr, y_tr, X_ev, hidden=256, epochs=500, lr=1e-3, patience=20):
    mu, sigma = y_tr.mean(), y_tr.std() + 1e-8
    y_n = (y_tr - mu) / sigma
    n_val = max(10, int(0.1 * len(X_tr)))
    Xt = torch.from_numpy(X_tr[:-n_val]).float()
    yt = torch.from_numpy(y_n[:-n_val]).float().unsqueeze(1)
    Xv = torch.from_numpy(X_tr[-n_val:]).float()
    yv = torch.from_numpy(y_n[-n_val:]).float().unsqueeze(1)
    Xe = torch.from_numpy(X_ev).float()
    model = nn.Sequential(nn.Linear(X_tr.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad(); loss_fn(model(Xt), yt).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), yv).item()
        if vl < best_val - 1e-5:
            best_val, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = model(Xe).squeeze().numpy()
    return p * sigma + mu


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_panel(ax, y_true, y_pred, y_train, title, color, is_best=False):
    r2_val = float(_r2(y_true, y_pred))
    pr, _ = pearsonr(y_true, y_pred)
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    ax.scatter(y_true, y_pred, s=3, alpha=0.2, color=color, rasterized=True)
    lo = min(y_true.min(), y_pred.min()) - 0.05
    hi = max(y_true.max(), y_pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    best_tag = "★ BEST  " if is_best else ""
    ax.set_title(
        f"{best_tag}{title}\n"
        f"R²={r2_val:.3f}  r={float(pr):.3f}  MAE={mae:.3f}\n"
        f"train μ={y_train.mean():.2f} σ={y_train.std():.2f}\n"
        f"pred  μ={y_pred.mean():.2f} σ={y_pred.std():.2f}",
        fontsize=7.5, fontweight="bold" if is_best else "normal",
        color="darkgreen" if is_best else "black",
    )
    if is_best:
        for spine in ax.spines.values():
            spine.set_edgecolor("gold"); spine.set_linewidth(3)
        ax.patch.set_facecolor("#fffff0")
    ax.set_xlabel("True", fontsize=7); ax.set_ylabel("Pred", fontsize=7)
    ax.grid(True, alpha=0.2); ax.tick_params(labelsize=6)
    return r2_val


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot generators
# ─────────────────────────────────────────────────────────────────────────────

def _plot_linear_vs_mlp(out_dir, inputs, layer_embs, y_cache, spec_comp_to_row, rng_idx):
    row_configs = [
        (100,  _fit_linear, "Linear  train=100"),
        (100,  _fit_mlp,    "MLP  train=100"),
        (1000, _fit_linear, "Linear  train=1000"),
        (1000, _fit_mlp,    "MLP  train=1000"),
    ]
    col_configs = [("Raw 490-dim", inputs, (0, 1))] + \
        [(f"{t}\nL{bl} 1536-dim", layer_embs[rn], (0, 1)) for rn, t, bl in CHECKPOINTS]

    res = {}
    for row, (n_tr, probe_fn, row_label) in enumerate(row_configs):
        for col, (col_label, arr, comps) in enumerate(col_configs):
            X, y = _build_merged(arr, y_cache, spec_comp_to_row, comps=comps)
            X_tr, y_tr, X_ev, y_ev = _get_split(X, y, rng_idx, n_tr)
            preds = probe_fn(X_tr, y_tr, X_ev)
            res[(row, col)] = (y_ev, preds, y_tr, col_label, row_label)

    best = max(res, key=lambda k: float(_r2(res[k][0], res[k][1])))
    fig, axes = plt.subplots(4, 4, figsize=(4 * 4.0, 4 * 5.0), squeeze=False)
    fig.suptitle("Linear vs MLP  —  2 components  |  eval=1000", fontsize=12, fontweight="bold")
    for (row, col), (y_ev, preds, y_tr, col_label, row_label) in res.items():
        title = f"{col_label}\n{row_label}" if row == 0 else row_label
        _scatter_panel(axes[row][col], y_ev, preds, y_tr, title, COLORS[col],
                       is_best=(row, col) == best)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, os.path.join(out_dir, "comparison_linear_vs_mlp.png"))


def _plot_n_components(out_dir, inputs, layer_embs, y_cache, spec_comp_to_row, rng_idx):
    comp_rows   = [(0,), (0, 1), (0, 1, 2)]
    comp_labels = ["1-comp (C0)\nraw 245 / emb 768",
                   "2-comp (C0,C1)\nraw 490 / emb 1536",
                   "3-comp (C0,C1,C2)\nraw 735 / emb 2304"]
    res = {}
    for row, (comps, comp_label) in enumerate(zip(comp_rows, comp_labels)):
        col_cfgs = [("Raw input", inputs, comps)] + \
            [(f"{t}\nL{bl}", layer_embs[rn], comps) for rn, t, bl in CHECKPOINTS]
        for col, (col_label, arr, c) in enumerate(col_cfgs):
            X, y = _build_merged(arr, y_cache, spec_comp_to_row, comps=c)
            X_tr, y_tr, X_ev, y_ev = _get_split(X, y, rng_idx, 1000)
            preds = _fit_linear(X_tr, y_tr, X_ev)
            res[(row, col)] = (y_ev, preds, y_tr, col_label, comp_label)

    best = max(res, key=lambda k: float(_r2(res[k][0], res[k][1])))
    fig, axes = plt.subplots(3, 4, figsize=(4 * 4.0, 3 * 5.0), squeeze=False)
    fig.suptitle("1 vs 2 vs 3 components  —  Linear  |  train=1000  eval=1000",
                 fontsize=12, fontweight="bold")
    for (row, col), (y_ev, preds, y_tr, col_label, comp_label) in res.items():
        title = f"{col_label}\n{comp_label}" if row == 0 else comp_label
        _scatter_panel(axes[row][col], y_ev, preds, y_tr, title, COLORS[col],
                       is_best=(row, col) == best)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, os.path.join(out_dir, "comparison_n_components.png"))


def _plot_train_size(out_dir, inputs, layer_embs, y_cache, spec_comp_to_row, rng_idx):
    col_configs = [("Raw 490-dim", inputs, (0, 1))] + \
        [(f"{t}\nL{bl} 1536-dim", layer_embs[rn], (0, 1)) for rn, t, bl in CHECKPOINTS]
    res = {}
    for row, n_tr in enumerate([100, 1000, 2000]):
        for col, (col_label, arr, comps) in enumerate(col_configs):
            X, y = _build_merged(arr, y_cache, spec_comp_to_row, comps=comps)
            X_tr, y_tr, X_ev, y_ev = _get_split(X, y, rng_idx, n_tr)
            preds = _fit_linear(X_tr, y_tr, X_ev)
            res[(row, col)] = (y_ev, preds, y_tr, col_label, n_tr)

    best = max(res, key=lambda k: float(_r2(res[k][0], res[k][1])))
    fig, axes = plt.subplots(3, 4, figsize=(4 * 4.0, 3 * 5.0), squeeze=False)
    fig.suptitle("Training size: 100 vs 1k vs 2k  —  Linear  |  2-comp  |  eval=1000",
                 fontsize=12, fontweight="bold")
    for (row, col), (y_ev, preds, y_tr, col_label, n_tr) in res.items():
        title = f"{col_label}\ntrain={n_tr}" if row == 0 else f"train={n_tr}"
        _scatter_panel(axes[row][col], y_ev, preds, y_tr, title, COLORS[col],
                       is_best=(row, col) == best)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, os.path.join(out_dir, "comparison_train_size.png"))


def _plot_param0_distributions(out_dir, inputs, layer_embs, y_cache, spec_comp_to_row, rng_idx):
    """Full distribution overview + true vs predicted KDE per model."""
    _, y_all_raw = _build_merged(inputs, y_cache, spec_comp_to_row, comps=(0, 1))
    y_all = y_all_raw  # spec-level labels

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                            height_ratios=[1, 1.2])

    # Top panel: full distribution
    ax0 = fig.add_subplot(gs[0, :])
    ax0.hist(y_all, bins=80, color="steelblue", alpha=0.7, edgecolor="white",
             linewidth=0.4, density=True)
    kde = gaussian_kde(y_all, bw_method=0.12)
    xs  = np.linspace(y_all.min() - 0.3, y_all.max() + 0.3, 500)
    ax0.plot(xs, kde(xs), color="navy", linewidth=2.5, label="KDE")
    ax0.axvline(y_all.mean(), color="crimson", linewidth=1.8, linestyle="--",
                label=f"mean={y_all.mean():.3f}")
    ax0.axvline(np.median(y_all), color="darkorange", linewidth=1.8, linestyle=":",
                label=f"median={np.median(y_all):.3f}")
    stats_txt = (f"N={len(y_all):,}   mean={y_all.mean():.3f}   std={y_all.std():.3f}\n"
                 f"min={y_all.min():.3f}   max={y_all.max():.3f}   "
                 f"p25={np.percentile(y_all,25):.3f}   p75={np.percentile(y_all,75):.3f}")
    ax0.set_title(f"parameter_0 — full distribution\n{stats_txt}",
                  fontsize=11, fontweight="bold")
    ax0.set_xlabel("parameter_0", fontsize=10); ax0.set_ylabel("Density", fontsize=10)
    ax0.legend(fontsize=9); ax0.grid(True, alpha=0.25)

    # Bottom panels: true vs predicted KDE per model (best layer)
    n_train = int(len(y_all) * 0.30)
    y_tr_dist = y_all[:n_train]
    y_ev_dist = y_all[n_train:]

    for col, (run_name, title, best_layer) in enumerate(CHECKPOINTS):
        emb = layer_embs[run_name]
        X_e, y_e = _build_merged(emb, y_cache, spec_comp_to_row, comps=(0, 1))
        Xs, ys = X_e[rng_idx[:len(X_e)]], y_e[rng_idx[:len(y_e)]]
        X_tr_l, y_tr_l = Xs[:n_train], ys[:n_train]
        X_ev_l, y_ev_l = Xs[n_train:n_train + N_EVAL], ys[n_train:n_train + N_EVAL]
        preds = _fit_linear(X_tr_l, y_tr_l, X_ev_l)

        ax = fig.add_subplot(gs[1, col])
        # True KDE
        kde_t = gaussian_kde(y_ev_l, bw_method=0.15)
        kde_p = gaussian_kde(preds,  bw_method=0.15)
        xr = np.linspace(min(y_ev_l.min(), preds.min()) - 0.3,
                         max(y_ev_l.max(), preds.max()) + 0.3, 400)
        ax.plot(xr, kde_t(xr), color="steelblue", linewidth=2, label=f"True σ={y_ev_l.std():.3f}")
        ax.fill_between(xr, kde_t(xr), alpha=0.12, color="steelblue")
        ax.plot(xr, kde_p(xr), color="tomato", linewidth=2, linestyle="--",
                label=f"Pred σ={preds.std():.3f}")
        ax.fill_between(xr, kde_p(xr), alpha=0.12, color="tomato")
        r2v = float(_r2(y_ev_l, preds))
        pr, _ = pearsonr(y_ev_l, preds)
        ax.set_title(f"{title}\nL{best_layer}  R²={r2v:.3f}  r={float(pr):.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("parameter_0", fontsize=9); ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    fig.suptitle("parameter_0 — True vs Predicted distributions  |  30% train (linear probe)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "param0_distributions.png"))


def _plot_merged_comp_comparison(out_dir, inputs, layer_embs, y_cache, spec_comp_to_row, rng_idx):
    """Raw 490-dim vs all 3 model 1536-dim embeddings, train=1000."""
    col_configs = [("Raw input\n490-dim", inputs, (0, 1), "dimgray")] + \
        [(f"{t}\nL{bl} 1536-dim", layer_embs[rn], (0, 1), COLORS[i + 1])
         for i, (rn, t, bl) in enumerate(CHECKPOINTS)]

    results = []
    for col_label, arr, comps, color in col_configs:
        X, y = _build_merged(arr, y_cache, spec_comp_to_row, comps=comps)
        X_tr, y_tr, X_ev, y_ev = _get_split(X, y, rng_idx, 1000)
        preds  = _fit_linear(X_tr, y_tr, X_ev)
        r2v    = float(_r2(y_ev, preds))
        pr, _  = pearsonr(y_ev, preds)
        mae    = float(np.mean(np.abs(y_ev - preds)))
        results.append((col_label, y_ev, preds, y_tr, r2v, float(pr), mae, color))

    fig, axes = plt.subplots(1, len(results), figsize=(len(results) * 4.5, 5.2), squeeze=False)
    fig.suptitle(
        "True vs Predicted  —  parameter_0  |  train=1000 / eval=1000\n"
        "490-dim merged input  vs  1536-dim model embeddings (comp0‖comp1, best layer)",
        fontsize=11, fontweight="bold",
    )
    best_idx = int(np.argmax([r[4] for r in results]))
    for i, (col_label, y_ev, preds, y_tr, r2v, prv, maev, color) in enumerate(results):
        is_best = (i == best_idx)
        ax = axes[0][i]
        ax.scatter(y_ev, preds, s=5, alpha=0.25, color=color, rasterized=True)
        lo = min(y_ev.min(), preds.min()) - 0.05
        hi = max(y_ev.max(), preds.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.4)
        best_tag = "★ BEST  " if is_best else ""
        ax.set_title(
            f"{best_tag}{col_label}\nR²={r2v:.3f}  r={prv:.3f}  MAE={maev:.3f}\n"
            f"train μ={y_tr.mean():.2f} σ={y_tr.std():.2f}\n"
            f"pred  μ={preds.mean():.2f} σ={preds.std():.2f}",
            fontsize=9, fontweight="bold" if is_best else "normal",
            color="darkgreen" if is_best else "black",
        )
        if is_best:
            for spine in ax.spines.values():
                spine.set_edgecolor("gold"); spine.set_linewidth(3)
            ax.patch.set_facecolor("#fffff0")
        ax.set_xlabel("True parameter_0", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.grid(True, alpha=0.2); ax.tick_params(labelsize=7)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "merged_comp_comparison_1000.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_label_reg_plots(output_dir: str) -> str:
    """
    Generate all label_regression diagnostic plots.

    Args:
        output_dir: root dir of the evaluation run (e.g. code/eval_results/20260322_120000)

    Returns:
        Path to the label_reg_plots sub-directory.
    """
    plots_dir = os.path.join(output_dir, "plots", "label_reg_plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Label Regression Plot Suite")
    print(f"  Output → {plots_dir}")
    print(f"{'='*60}")

    print("\n[1/5] Loading shared data …")
    y_cache, spec_comp_to_row = _build_row_map()
    inputs = np.load(f"{CACHE_BASE}/label_reg_emb_cache_2026-01-07_21-50-07.npz")["inputs"]
    layer_embs = {}
    for run_name, title, best_layer in CHECKPOINTS:
        if best_layer is not None:
            layer_embs[run_name] = np.load(
                LAYER_CACHE.format(run=run_name) + f"/layer_{best_layer}.npz")["embeddings"]
        else:
            print(f"    Extracting embeddings from {Path(run_name).name} …")
            layer_embs[run_name] = _extract_embeddings(inputs, run_name)
    _, y2 = _build_merged(inputs, y_cache, spec_comp_to_row, comps=(0, 1))
    rng_idx = np.random.default_rng(SEED).permutation(len(y2))

    print("\n[2/5] Linear vs MLP …")
    _plot_linear_vs_mlp(plots_dir, inputs, layer_embs, y_cache, spec_comp_to_row, rng_idx)

    print("\n[3/5] N-components sweep …")
    _plot_n_components(plots_dir, inputs, layer_embs, y_cache, spec_comp_to_row, rng_idx)

    print("\n[4/5] Training size sweep …")
    _plot_train_size(plots_dir, inputs, layer_embs, y_cache, spec_comp_to_row, rng_idx)

    print("\n[5/5] Parameter_0 distributions …")
    _plot_param0_distributions(plots_dir, inputs, layer_embs, y_cache, spec_comp_to_row, rng_idx)

    print("\n[5b] Merged comp comparison (train=1000) …")
    _plot_merged_comp_comparison(plots_dir, inputs, layer_embs, y_cache, spec_comp_to_row, rng_idx)

    print(f"\n[✓] All label_reg plots saved → {plots_dir}\n")
    return plots_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate label_regression diagnostic plots")
    parser.add_argument("--output_dir", default="eval_results",
                        help="Base output directory (a timestamped subdir is created automatically)")
    args = parser.parse_args()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"label_reg_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    run_label_reg_plots(run_dir)
