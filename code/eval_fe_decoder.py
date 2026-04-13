"""
FE Decoder Reconstruction Evaluation
=====================================

Trains lightweight decoders on top of **frozen** CNN feature-extractor outputs
to reconstruct the original input spectrogram.  Three decoder architectures are
compared in one run (the "3 experiments"):

  Exp 1 — Linear  : Linear(512 → 245)                                 ~125k params
  Exp 2 — MLP-512 : Linear(512→512) → ReLU → Linear(512→245)         ~387k params
  Exp 3 — MLP-512-256: …→ReLU→Linear(512→256)→ReLU→Linear(256→245)  ~518k params

All three decoders receive the same mean-pooled FE representation [B, 512].
Better reconstruction with a linear decoder = the FE encodes the input in a
linearly-accessible way.  A gap between Linear and MLP variants reveals how
much non-linear structure is present in the FE encoding.

Pipeline (shared across all 3 experiments)
------------------------------------------
  input [B, 245]
    → feature_extractor (5 valid-conv layers, no padding):
        conv1 k=3 s=1  → [B, 512, 243]
        conv2 k=3 s=1  → [B, 512, 241]
        conv3 k=3 s=1  → [B, 512, 239]
        conv4 k=3 s=1  → [B, 512, 237]
        conv5 k=5 s=5  → [B, 512, 47]   (T' = floor((237-5)/5+1) = 47)
    → transpose(1,2)   → [B, 47, 512]
    → layer_norm       → [B, 47, 512]
    → mean over time   → [B, 512]       ← shared FE representation
    ↓
  Exp 1: Linear(512 → 245)
  Exp 2: Linear(512→512) → ReLU → Linear(512→245)
  Exp 3: Linear(512→512) → ReLU → Linear(512→256) → ReLU → Linear(256→245)

NOTE — padding: all CNN layers use **valid** (no-padding) convolutions.
No padding tokens are added anywhere; all sequences are fixed-length (245→47).

CLI mirrors evaluation_runner.py so the script can be dropped into the same
workflow.  The eval method name is ``decode_fe``.

Usage examples
--------------
# Run all 3 decoder experiments on a single checkpoint (default behaviour)
python code/eval_fe_decoder.py \\
    --checkpoint /path/to/checkpoint_best.pt \\
    --eval_data_dir fairseq/data/nova_data/single_channel_10k \\
    --max_eval_samples 2000

# Run only a single linear decoder (legacy behaviour)
python code/eval_fe_decoder.py \\
    --checkpoint /path/to/checkpoint_best.pt \\
    --eval_data_dir fairseq/data/nova_data/single_channel_10k \\
    --decoder_variants 0

# Custom decoder specs: linear + MLP-1024 + MLP-1024:512
python code/eval_fe_decoder.py \\
    --checkpoint /path/to/checkpoint_best.pt \\
    --eval_data_dir fairseq/data/nova_data/single_channel_10k \\
    --decoder_variants 0 1024 1024:512

# Auto-discover all checkpoints in a directory
python code/eval_fe_decoder.py \\
    --checkpoint_dir /mnt5/noy/SpectralFM/checkpoints/runai/fe_vs_transformer_collapse \\
    --eval_data_dir fairseq/data/nova_data/single_channel_10k \\
    --max_eval_samples 2000

Checkpoint directory structures supported
-----------------------------------------
1. Flat .pt files (e.g. compare-single-to-multi style):
     <checkpoint_dir>/
       2026-01-07_21-50-07_all_long.pt
       2026-02-25_13-46-46_all_long.pt
   → run_name extracted from filename: "2026-01-07_21-50-07"

2. Run subdirs with checkpoint_best.pt (e.g. runai/ style):
     <checkpoint_dir>/
       2026-01-07_21-50-07/
         checkpoint_best.pt
       2026-02-25_13-46-46/
         checkpoint_best.pt
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project path setup ──────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FAIRSEQ_PATH = os.path.join(_THIS_DIR, "..", "fairseq")
if _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eval_metrics import compute_fe_decoder_metrics
from eval_plots import (
    plot_fe_decoder_reconstruction_samples,
    plot_fe_decoder_score_distribution,
    plot_per_bin_error_heatmap,
    plot_pca_component_r2,
    plot_residual_scatter,
    plot_fe_vs_transformer_r2,
    plot_fe_vs_transformer_comparison_bar_chart,
    plot_reconstruction_triple,
    plot_all_decoder_variants,
    plot_fe_vs_transformer_by_architecture,
    plot_fe_vs_transformer_by_architecture_multi_model,
)

# Pattern that matches the date-time prefix (with optional short tag) used as
# the run name in SpectralFM checkpoint filenames.
# Examples:
#   "2026-01-07_21-50-07_all_long"      → "2026-01-07_21-50-07"
#   "2026-03-03_17-45-36-multi_channel" → "2026-03-03_17-45-36-multi"
_RUN_NAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:-[A-Za-z0-9]+)?)"
)


# ── Checkpoint discovery ─────────────────────────────────────────────────────

def _run_name_from_stem(stem: str) -> str:
    """Extract the date-time run name from a checkpoint file stem."""
    m = _RUN_NAME_RE.match(stem)
    return m.group(1) if m else stem


def discover_checkpoints(
    checkpoint_dir: str,
    best_only: bool = False,
    latest_only: bool = False,
    run_names: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Auto-discover checkpoints under checkpoint_dir.

    Supports two directory layouts:

    Layout A — flat .pt files in the directory root:
        <dir>/2026-01-07_21-50-07_all_long.pt
        <dir>/2026-02-25_13-46-46_all_long.pt

    Layout B — run subdirs, each containing a .pt file:
        <dir>/2026-01-07_21-50-07/checkpoint_best.pt
        <dir>/2026-02-25_13-46-46/checkpoint_best.pt

    For Layout B, ``best_only`` prefers checkpoint_best.pt (already the
    default); ``latest_only`` prefers checkpoint_last.pt.

    Args:
        checkpoint_dir: root directory to search
        best_only:      prefer checkpoint_best.pt in subdirs
        latest_only:    prefer checkpoint_last.pt in subdirs
        run_names:      if given, only include runs whose name is in this list

    Returns:
        Sorted list of (run_name, checkpoint_path) tuples.
    """
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint_dir does not exist: {checkpoint_dir}")

    results: List[Tuple[str, str]] = []

    # Layout A: *.pt files directly in root
    flat_pts = sorted(root.glob("*.pt"))
    if flat_pts:
        for pt in flat_pts:
            run_name = _run_name_from_stem(pt.stem)
            results.append((run_name, str(pt)))
    else:
        # Layout B: subdirectories
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            run_name = subdir.name

            # Preference order
            candidates = []
            if latest_only:
                candidates = ["checkpoint_last.pt", "checkpoint_best.pt"]
            else:
                candidates = ["checkpoint_best.pt", "checkpoint_last.pt"]

            # Also accept any single .pt in the subdir
            found = None
            for name in candidates:
                p = subdir / name
                if p.exists():
                    found = p
                    break
            if found is None:
                pts = sorted(subdir.glob("*.pt"))
                if pts:
                    found = pts[0]

            if found:
                results.append((run_name, str(found)))

    if run_names:
        run_names_set = set(run_names)
        results = [(rn, cp) for rn, cp in results if rn in run_names_set]

    if not results:
        print(f"[!] No checkpoints found in {checkpoint_dir}")
    else:
        print(f"[+] Discovered {len(results)} checkpoint(s) in {checkpoint_dir}:")
        for rn, cp in results:
            print(f"    {rn:40s}  {cp}")

    return results


# ── FE extraction ────────────────────────────────────────────────────────────

def extract_fe_outputs_from_inputs(
    inputs: np.ndarray,
    checkpoint_path: str,
    device_str: str = "cpu",
    batch_size: int = 64,
) -> Optional[np.ndarray]:
    """
    Pass inputs through the checkpoint's CNN feature extractor (frozen).

    Args:
        inputs:          float32 [N, T] — raw model inputs (245-d spectrograms)
        checkpoint_path: path to fairseq .pt checkpoint
        device_str:      'cpu' or 'cuda'
        batch_size:      samples per forward pass

    Returns:
        float32 [N, T'×512] flattened FE outputs (before post_extract_proj),
        or None on failure.
    """
    import torch
    from model_loader import load_fairseq_checkpoint

    device = torch.device(device_str)
    try:
        model, _, _ = load_fairseq_checkpoint(checkpoint_path)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    except Exception as exc:
        print(f"    [!] Could not load checkpoint: {exc}")
        return None

    fe_vecs = []
    N = len(inputs)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch = inputs[start : start + batch_size]
            src = torch.from_numpy(batch).float().to(device)   # [B, T]
            fe = model.feature_extractor(src)                  # [B, 512, T']
            fe = fe.transpose(1, 2)                            # [B, T', 512]
            fe = model.layer_norm(fe)                          # [B, T', 512]
            # Mean-pool over the time dimension → [B, 512].
            # This reduces decoder params from 5.9M (flat) to 125k (pooled),
            # making linear decoding tractable with O(1k) training samples.
            fe_pooled = fe.mean(dim=1)                         # [B, 512]
            fe_vecs.append(fe_pooled.cpu().numpy())
            print(f"    FE extraction: {min(start + batch_size, N)}/{N}", end="\r")

    print()
    return np.concatenate(fe_vecs, axis=0).astype(np.float32)


# ── Transformer embedding extraction (for D4) ────────────────────────────────

def extract_embeddings_from_inputs(
    inputs: np.ndarray,
    checkpoint_path: str,
    device_str: str = "cpu",
    batch_size: int = 64,
) -> Optional[np.ndarray]:
    """
    Pass inputs through the full transformer and mean-pool over time → [N, 768].
    Used to train a comparison decoder for D4.
    """
    import torch
    from model_loader import load_fairseq_checkpoint

    device = torch.device(device_str)
    try:
        model, _, _ = load_fairseq_checkpoint(checkpoint_path)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    except Exception as exc:
        print(f"    [!] Could not load checkpoint for embedding extraction: {exc}")
        return None

    emb_vecs = []
    N = len(inputs)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch = inputs[start : start + batch_size]
            src = torch.from_numpy(batch).float().to(device)
            out = model(src, padding_mask=None, mask=False, features_only=True)
            emb = out["x"].mean(dim=1)             # [B, 768]
            emb_vecs.append(emb.cpu().numpy())
            print(f"    Embedding extraction: {min(start + batch_size, N)}/{N}", end="\r")

    print()
    return np.concatenate(emb_vecs, axis=0).astype(np.float32)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_and_split_inputs(
    eval_data_dir: Optional[str],
    inputs_npy: Optional[str],
    max_eval_samples: int,
    n_train: Optional[int],
    target_length: int = 245,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrogram inputs and split into train / eval.

    ``max_eval_samples`` controls the **total** number of samples loaded.
    The split is 50/50 by default; override train size with ``n_train``.

    Priority for loading: --inputs_npy > --eval_data_dir WAV files.
    """
    if inputs_npy:
        print(f"[+] Loading inputs from .npy: {inputs_npy}")
        inputs = np.load(inputs_npy).astype(np.float32)
        inputs = inputs[:max_eval_samples]
    elif eval_data_dir:
        from eval_utils import load_wav_files_torchaudio
        print(f"[+] Loading up to {max_eval_samples} WAV files from {eval_data_dir} ...")
        inputs, _ = load_wav_files_torchaudio(
            eval_data_dir, max_samples=max_eval_samples, target_length=target_length
        )
        print(f"    Loaded {len(inputs)} samples, shape {inputs.shape}")
    else:
        raise ValueError("Either --eval_data_dir or --inputs_npy must be provided.")

    rng = np.random.default_rng(seed)
    inputs = inputs[rng.permutation(len(inputs))]

    split = n_train if n_train is not None else len(inputs) // 2
    split = min(split, len(inputs) - 1)

    train_inputs = inputs[:split]
    eval_inputs = inputs[split:]
    print(f"[+] Split: {len(train_inputs)} train / {len(eval_inputs)} eval")
    return train_inputs, eval_inputs


# ── Comparison bar chart ──────────────────────────────────────────────────────

def plot_comparison_bar_chart(
    all_metrics: List[Dict],
    run_names: List[str],
    output_path: str,
) -> None:
    """Side-by-side bar charts: cosine mean±std, MSE, R² per checkpoint."""
    cosine_means = [m["fe_dec_cosine_mean"] for m in all_metrics]
    cosine_stds = [m["fe_dec_cosine_std"] for m in all_metrics]
    mse_vals = [m["fe_dec_mse"] for m in all_metrics]
    r2_vals = [m["fe_dec_r2"] for m in all_metrics]

    x = np.arange(len(run_names))
    colors = ["#3498DB", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12",
              "#1ABC9C", "#E67E22", "#95A5A6"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("FE Decoder Reconstruction — Checkpoint Comparison",
                 fontsize=14, fontweight="bold")

    panels = [
        (axes[0], cosine_means, cosine_stds, "Cosine Similarity (↑ better)",
         "Reconstruction Cosine Similarity\n(mean ± std)", True),
        (axes[1], mse_vals,    None,         "MSE (↓ better)",
         "Reconstruction MSE",              False),
        (axes[2], r2_vals,     None,         "R² (↑ better)",
         "Reconstruction R²",               True),
    ]
    for ax, vals, errs, ylabel, title, fmt3 in panels:
        bars = ax.bar(
            x, vals,
            yerr=errs, capsize=5 if errs else 0,
            color=[colors[i % len(colors)] for i in range(len(run_names))],
            edgecolor="white", alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.3f}" if fmt3 else f"{val:.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.005 if fmt3 else 1e-5),
                    fmt, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison chart: {output_path}")


# ── Decoder variant helpers ───────────────────────────────────────────────────

def parse_decoder_variant(spec: str) -> List[int]:
    """
    Parse a decoder variant spec string into a list of hidden-layer dims.

    Examples
    --------
    "0"       → []          (single Linear, no hidden layers)
    "512"     → [512]       (1 hidden layer: Linear→ReLU→Linear)
    "512:256" → [512, 256]  (2 hidden layers: Linear→ReLU→Linear→ReLU→Linear)
    """
    spec = spec.strip()
    if spec == "0":
        return []
    return [int(x) for x in spec.split(":")]


def variant_label(hidden: List[int]) -> str:
    """Return a short human-readable label for a decoder variant."""
    if not hidden:
        return "Linear"
    return "MLP-" + "-".join(str(h) for h in hidden)


def plot_decoder_variants_comparison(
    variants_metrics: Dict[str, Dict],
    run_name: str,
    output_path: str,
) -> None:
    """
    Bar chart comparing 3 decoder variants (Linear / MLP-1 / MLP-2) for a single
    checkpoint.  Shows Cosine (mean±std), MSE, and R² side-by-side.
    """
    labels = list(variants_metrics.keys())
    cosine_means = [variants_metrics[v]["fe_dec_cosine_mean"] for v in labels]
    cosine_stds  = [variants_metrics[v]["fe_dec_cosine_std"]  for v in labels]
    mse_vals     = [variants_metrics[v]["fe_dec_mse"]         for v in labels]
    r2_vals      = [variants_metrics[v]["fe_dec_r2"]          for v in labels]

    x = np.arange(len(labels))
    colors = ["#3498DB", "#E74C3C", "#27AE60"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Decoder Variants — {run_name}", fontsize=12, fontweight="bold")

    panels = [
        (axes[0], cosine_means, cosine_stds, "Cosine Similarity (↑)", True),
        (axes[1], mse_vals,     None,         "MSE (↓)",              False),
        (axes[2], r2_vals,      None,         "R² (↑)",               True),
    ]
    for ax, vals, errs, ylabel, fmt3 in panels:
        bars = ax.bar(
            x, vals,
            yerr=errs, capsize=5 if errs else 0,
            color=[colors[i % len(colors)] for i in range(len(labels))],
            edgecolor="white", alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            fmt = f"{val:.3f}" if fmt3 else f"{val:.5f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.003 if fmt3 else 1e-6),
                    fmt, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] Saved variant comparison: {output_path}")


def plot_cross_checkpoint_variants(
    all_variants_metrics: Dict[str, Dict[str, Dict]],
    output_path: str,
) -> None:
    """
    Grouped bar chart: rows = metric (Cosine / R²), columns = decoder variant,
    colour = checkpoint.  Lets you compare models across all decoder depths at once.

    all_variants_metrics: {run_name → {variant_label → metrics_dict}}
    """
    run_names  = list(all_variants_metrics.keys())
    if not run_names:
        return
    variant_labels = list(next(iter(all_variants_metrics.values())).keys())

    palette = ["#3498DB", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12",
               "#1ABC9C", "#E67E22", "#95A5A6"]

    n_variants = len(variant_labels)
    fig, axes = plt.subplots(2, n_variants,
                             figsize=(5 * n_variants, 8),
                             constrained_layout=True)
    if n_variants == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("Decoder Variants × Checkpoints", fontsize=13, fontweight="bold")

    metrics_to_plot = [
        ("fe_dec_cosine_mean", "fe_dec_cosine_std", "Cosine Similarity (↑)"),
        ("fe_dec_r2",          None,                "R² (↑)"),
    ]

    x = np.arange(len(run_names))
    bar_w = 0.65

    for row, (key, err_key, ylabel) in enumerate(metrics_to_plot):
        for col, vlabel in enumerate(variant_labels):
            ax = axes[row, col]
            vals = [all_variants_metrics[rn].get(vlabel, {}).get(key, 0.0)
                    for rn in run_names]
            errs = ([all_variants_metrics[rn].get(vlabel, {}).get(err_key, 0.0)
                     for rn in run_names] if err_key else None)
            bars = ax.bar(
                x, vals, yerr=errs, capsize=4,
                width=bar_w,
                color=[palette[i % len(palette)] for i in range(len(run_names))],
                edgecolor="white", alpha=0.85,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(run_names, rotation=30, ha="right", fontsize=8)
            ax.set_title(vlabel if row == 0 else "", fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved cross-checkpoint variant plot: {output_path}")


# ── Per-checkpoint evaluation ─────────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint_path: str,
    train_inputs: np.ndarray,
    eval_inputs: np.ndarray,
    run_name: str,
    output_dir: Path,
    device_str: str,
    epochs: int,
    lr: float,
    batch_size: int,
    fe_batch_size: int,
    smoothness_weight: float = 0.1,
    decoder_variants: Optional[List[List[int]]] = None,
    include_embedding_decoder: bool = False,
) -> Optional[Dict]:
    """
    Run the full FE-decoder evaluation for one checkpoint.

    Trains one decoder per entry in ``decoder_variants``:
      - []        → Linear(512 → 245)                                 Exp 1
      - [512]     → Linear(512→512) → ReLU → Linear(512→245)         Exp 2
      - [512,256] → Linear(512→512) → ReLU→Linear(512→256)→ReLU→245 Exp 3
    """
    if decoder_variants is None:
        decoder_variants = [[]]   # default: single linear decoder

    print(f"\n{'='*60}")
    print(f"[+] Checkpoint: {run_name}")
    print(f"    Path: {checkpoint_path}")

    run_dir = output_dir / run_name
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_inputs = np.concatenate([train_inputs, eval_inputs], axis=0)
    n_train = len(train_inputs)

    print(f"[+] Extracting FE outputs for {len(all_inputs)} samples ...")
    all_fe = extract_fe_outputs_from_inputs(
        all_inputs, checkpoint_path, device_str=device_str, batch_size=fe_batch_size
    )
    if all_fe is None:
        print(f"    [!] FE extraction failed for {run_name} — skipping.")
        return None

    print(f"    FE output shape: {all_fe.shape}")

    np.save(run_dir / "fe_outputs_train.npy", all_fe[:n_train])
    np.save(run_dir / "fe_outputs_eval.npy", all_fe[n_train:])
    np.save(run_dir / "inputs_train.npy", train_inputs)
    np.save(run_dir / "inputs_eval.npy", eval_inputs)

    train_mask = np.zeros(len(all_inputs), dtype=bool)
    train_mask[:n_train] = True

    # ── Loop over decoder variants (Exp 1 / 2 / 3) ───────────────────────────
    variants_metrics: Dict[str, Dict] = {}  # label → clean metrics dict
    first_metrics = None                    # used for downstream plots
    first_reconstructed = None

    for hidden in decoder_variants:
        vlabel = variant_label(hidden)
        n_params = _decoder_param_count(hidden, all_fe.shape[1], train_inputs.shape[1])
        arch_str = _decoder_arch_str(hidden, all_fe.shape[1], train_inputs.shape[1])
        print(f"\n[+] Decoder [{vlabel}] {arch_str}  ({n_params:,} params, "
              f"epochs={epochs}, lr={lr}, smoothness={smoothness_weight})")

        metrics, reconstructed_eval = compute_fe_decoder_metrics(
            inputs=all_inputs,
            fe_outputs=all_fe,
            train_mask=train_mask,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            smoothness_weight=smoothness_weight,
            decoder_layers=hidden,
        )

        per_cosine = metrics.pop("_per_cosine")
        per_mse    = metrics.pop("_per_mse")
        per_r2     = metrics.pop("_per_r2")
        metrics.pop("_per_mae", None)

        print(f"    [{vlabel}] cosine={metrics['fe_dec_cosine_mean']:.4f}  "
              f"mse={metrics['fe_dec_mse']:.5f}  r2={metrics['fe_dec_r2']:.4f}")

        # Save per-variant metrics JSON
        metrics_json = {k: v for k, v in metrics.items()
                        if isinstance(v, (int, float, str, bool))}
        metrics_json.update({"run_name": run_name, "decoder_variant": vlabel,
                              "decoder_arch": arch_str, "n_params": n_params,
                              "checkpoint_path": checkpoint_path})
        variant_dir = run_dir / vlabel
        variant_dir.mkdir(exist_ok=True)
        with open(variant_dir / "metrics.json", "w") as f:
            json.dump(metrics_json, f, indent=2)

        # Save reconstructed eval + per-sample arrays (needed for combined plots)
        np.save(variant_dir / "reconstructed_eval.npy", reconstructed_eval)
        np.save(variant_dir / "per_cosine.npy", per_cosine)
        np.save(variant_dir / "per_mse.npy",    per_mse)

        # Per-variant reconstruction samples plot
        plot_fe_decoder_reconstruction_samples(
            originals=eval_inputs,
            reconstructed=reconstructed_eval,
            per_cosine=per_cosine,
            per_mse=per_mse,
            save_path=str(plots_dir / f"reconstruction_samples_{vlabel}.png"),
            title=f"FE Decoder [{vlabel}] — {run_name}",
            n_panels=8,
        )
        plot_fe_decoder_score_distribution(
            per_cosine=per_cosine,
            per_mse=per_mse,
            save_path=str(plots_dir / f"score_distributions_{vlabel}.png"),
            title=f"FE Decoder [{vlabel}] — Score Distributions",
            run_name=run_name,
        )

        variants_metrics[vlabel] = metrics
        # Keep the first variant's data for backward-compat downstream plots
        if first_metrics is None:
            first_metrics = dict(metrics)
            first_metrics["_per_cosine"]    = per_cosine
            first_metrics["_per_mse"]       = per_mse
            first_metrics["_per_r2"]        = per_r2
            first_metrics["_eval_inputs"]   = eval_inputs
            first_metrics["_reconstructed"] = reconstructed_eval

    # Per-checkpoint variant comparison bar chart
    if len(decoder_variants) > 1:
        plot_decoder_variants_comparison(
            variants_metrics=variants_metrics,
            run_name=run_name,
            output_path=str(plots_dir / "decoder_variants_comparison.png"),
        )

    print(f"    Saved plots to: {plots_dir}/")

    # ── Optional: train a comparison decoder on transformer embeddings ────────
    if include_embedding_decoder:
        print(f"[+] Extracting transformer embeddings ({run_name}) ...")
        all_emb = extract_embeddings_from_inputs(
            all_inputs, checkpoint_path, device_str=device_str, batch_size=fe_batch_size
        )
        if all_emb is not None:
            np.save(run_dir / "embeddings_train.npy", all_emb[:n_train])
            np.save(run_dir / "embeddings_eval.npy",  all_emb[n_train:])
            print(f"    Embedding shape: {all_emb.shape}")

            # Train transformer decoder for EACH variant and generate triple plots
            tr_variants_recon: Dict[str, np.ndarray]     = {}
            tr_variants_cosine: Dict[str, np.ndarray]    = {}
            tr_variants_mse: Dict[str, np.ndarray]       = {}
            tr_variants_r2: Dict[str, float]             = {}
            tr_variants_per_r2: Dict[str, np.ndarray]   = {}   # per-sample R² arrays
            tr_variants_metrics: Dict[str, Dict]         = {}
            fe_variants_recon: Dict[str, np.ndarray]     = {}
            fe_variants_cosine: Dict[str, np.ndarray]    = {}
            fe_variants_mse: Dict[str, np.ndarray]       = {}

            for hidden in decoder_variants:
                vlabel = variant_label(hidden)
                arch_str = _decoder_arch_str(hidden, all_emb.shape[1], train_inputs.shape[1])
                print(f"[+] Training transformer decoder [{vlabel}] {arch_str} ...")
                tr_metrics, tr_recon_eval = compute_fe_decoder_metrics(
                    inputs=all_inputs,
                    fe_outputs=all_emb,
                    train_mask=train_mask,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    smoothness_weight=smoothness_weight,
                    decoder_layers=hidden,
                )
                tr_per_cosine = tr_metrics.pop("_per_cosine")
                tr_per_mse    = tr_metrics.pop("_per_mse")
                tr_per_r2     = tr_metrics.pop("_per_r2")
                tr_metrics.pop("_per_mae", None)

                print(f"    [Trans {vlabel}] cosine={tr_metrics['fe_dec_cosine_mean']:.4f}  "
                      f"r2={tr_metrics['fe_dec_r2']:.4f}")

                tr_variants_recon[vlabel]   = tr_recon_eval
                tr_variants_cosine[vlabel]  = tr_per_cosine
                tr_variants_mse[vlabel]     = tr_per_mse
                tr_variants_r2[vlabel]      = float(tr_metrics.get("fe_dec_r2", 0.0))
                tr_variants_per_r2[vlabel]  = tr_per_r2
                tr_variants_metrics[vlabel] = {k: v for k, v in tr_metrics.items()
                                               if isinstance(v, (int, float, str, bool))}

                # Save per-variant transformer metrics
                tr_json = {k: v for k, v in tr_metrics.items()
                           if isinstance(v, (int, float, str, bool))}
                tr_json.update({"run_name": run_name, "decoder_source": "transformer",
                                "decoder_variant": vlabel})
                variant_dir = run_dir / vlabel
                variant_dir.mkdir(exist_ok=True)
                with open(variant_dir / "metrics_transformer.json", "w") as f:
                    json.dump(tr_json, f, indent=2)
                np.save(variant_dir / "tr_reconstructed_eval.npy", tr_recon_eval)

                # Load saved FE per-sample arrays for this variant
                fe_variants_recon[vlabel]  = np.load(run_dir / vlabel / "reconstructed_eval.npy")
                fe_variants_cosine[vlabel] = np.load(run_dir / vlabel / "per_cosine.npy")
                fe_variants_mse[vlabel]    = np.load(run_dir / vlabel / "per_mse.npy")

                # Per-variant triple plot
                tr_plots_dir = plots_dir / "transformer"
                tr_plots_dir.mkdir(exist_ok=True)

                plot_reconstruction_triple(
                    originals=eval_inputs,
                    fe_reconstructed=fe_variants_recon[vlabel],
                    tr_reconstructed=tr_recon_eval,
                    per_cosine_fe=fe_variants_cosine[vlabel],
                    per_cosine_tr=tr_per_cosine,
                    per_mse_fe=fe_variants_mse[vlabel],
                    per_mse_tr=tr_per_mse,
                    save_path=str(plots_dir / f"reconstruction_triple_{vlabel}.png"),
                    title=f"FE vs Transformer — {vlabel}\n{run_name}",
                    n_panels=8,
                )
                print(f"    [+] Saved reconstruction_triple_{vlabel}.png")

            # Combined all-variants grid
            plot_all_decoder_variants(
                originals=eval_inputs,
                fe_variants=fe_variants_recon,
                tr_variants=tr_variants_recon,
                fe_cosines=fe_variants_cosine,
                fe_mses=fe_variants_mse,
                tr_cosines=tr_variants_cosine,
                tr_mses=tr_variants_mse,
                save_path=str(plots_dir / "reconstruction_all_variants.png"),
                title=f"All Decoder Variants — FE vs Transformer\n{run_name}",
                n_panels=8,
            )
            print(f"    [+] Saved reconstruction_all_variants.png")

            # Per-checkpoint FE vs Transformer bar chart grouped by architecture
            plot_fe_vs_transformer_by_architecture(
                fe_variants_metrics=variants_metrics,
                tr_variants_metrics=tr_variants_metrics,
                run_name=run_name,
                save_path=str(plots_dir / "fe_vs_transformer_by_architecture.png"),
            )

            # Attach tr_variants_metrics for cross-model plot in main()
            first_metrics["_tr_variants_metrics"] = tr_variants_metrics

            # Score-distribution plots for transformer
            for hidden in decoder_variants:
                vlabel = variant_label(hidden)
                plot_fe_decoder_score_distribution(
                    per_cosine=tr_variants_cosine[vlabel],
                    per_mse=tr_variants_mse[vlabel],
                    save_path=str(plots_dir / "transformer" / f"score_distributions_{vlabel}.png"),
                    title=f"Transformer Decoder [{vlabel}] — Score Distributions",
                    run_name=run_name,
                )

            # Use first variant for FE vs transformer bar chart and D4 scatter
            first_vlabel = variant_label(decoder_variants[0])
            first_tr_json_path = run_dir / first_vlabel / "metrics_transformer.json"
            with open(first_tr_json_path) as f:
                _first_tr = json.load(f)
            first_metrics["_tr_metrics"]      = _first_tr
            first_metrics["_tr_reconstructed"] = tr_variants_recon[first_vlabel]
            first_metrics["_tr_per_cosine"]    = tr_variants_cosine[first_vlabel]
            first_metrics["_tr_per_mse"]       = tr_variants_mse[first_vlabel]
            first_metrics["_tr_per_r2"]        = tr_variants_per_r2[first_vlabel]
            print(f"    Saved all variant plots to: {plots_dir}/")

    # Attach variants_metrics so main() can build cross-checkpoint comparison
    first_metrics["_variants_metrics"] = variants_metrics
    return first_metrics


# ── Decoder architecture helpers ──────────────────────────────────────────────

def _decoder_arch_str(hidden: List[int], fe_dim: int, out_dim: int) -> str:
    dims = [fe_dim] + hidden + [out_dim]
    parts = []
    for i in range(len(dims) - 1):
        parts.append(f"Linear({dims[i]}→{dims[i+1]})")
        if i < len(dims) - 2:
            parts.append("ReLU")
    return " → ".join(parts)


def _decoder_param_count(hidden: List[int], fe_dim: int, out_dim: int) -> int:
    dims = [fe_dim] + hidden + [out_dim]
    return sum((dims[i] + 1) * dims[i + 1] for i in range(len(dims) - 1))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build checkpoint list ────────────────────────────────────────────────
    checkpoints: List[Tuple[str, str]] = []   # [(run_name, path), ...]

    if args.checkpoint:
        for cp in args.checkpoint:
            p = Path(cp)
            if p.is_dir():
                # User passed a directory as --checkpoint: look for best/last .pt
                for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
                    if (p / name).exists():
                        checkpoints.append((p.name, str(p / name)))
                        break
                else:
                    pts = sorted(p.glob("*.pt"))
                    if pts:
                        checkpoints.append((p.name, str(pts[0])))
            else:
                # Regular .pt file — derive run name from parent dir or stem
                parent = p.parent.name
                if parent and parent not in ("checkpoints", ".", ""):
                    run_name = parent
                else:
                    run_name = _run_name_from_stem(p.stem)
                checkpoints.append((run_name, str(cp)))

    if args.checkpoint_dir:
        discovered = discover_checkpoints(
            args.checkpoint_dir,
            best_only=args.best_only,
            latest_only=getattr(args, "latest_only", False),
            run_names=args.run_names,
        )
        checkpoints.extend(discovered)

    if not checkpoints:
        print("[!] No checkpoints found. Use --checkpoint or --checkpoint_dir.")
        sys.exit(1)

    # ── Load data once (shared across checkpoints for fair comparison) ───────
    train_inputs, eval_inputs = load_and_split_inputs(
        eval_data_dir=args.eval_data_dir,
        inputs_npy=getattr(args, "inputs_npy", None),
        max_eval_samples=args.max_eval_samples,
        n_train=getattr(args, "n_train", None),
        target_length=getattr(args, "target_length", 245),
        seed=args.seed,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    all_metrics = []
    run_names = []

    # Parse decoder variants once
    decoder_variants: List[List[int]] = [
        parse_decoder_variant(s) for s in args.decoder_variants
    ]

    for run_name, ckpt_path in checkpoints:
        metrics = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            train_inputs=train_inputs,
            eval_inputs=eval_inputs,
            run_name=run_name,
            output_dir=output_dir,
            device_str=args.device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            fe_batch_size=args.fe_batch_size,
            smoothness_weight=args.smoothness_weight,
            decoder_variants=decoder_variants,
            include_embedding_decoder=args.include_embedding_decoder,
        )
        if metrics is not None:
            all_metrics.append(metrics)
            run_names.append(run_name)

    if not all_metrics:
        print("[!] No successful evaluations.")
        sys.exit(1)

    # ── Comparison bar chart (first decoder variant, FE only) ────────────────
    plot_comparison_bar_chart(
        all_metrics=[{k: v for k, v in m.items() if not k.startswith("_")}
                     for m in all_metrics],
        run_names=run_names,
        output_path=str(output_dir / "comparison_bar_chart.png"),
    )

    # ── Cross-checkpoint decoder variants comparison ──────────────────────────
    if len(decoder_variants) > 1:
        all_variants_metrics = {
            rn: m["_variants_metrics"]
            for rn, m in zip(run_names, all_metrics)
            if "_variants_metrics" in m
        }
        plot_cross_checkpoint_variants(
            all_variants_metrics=all_variants_metrics,
            output_path=str(output_dir / "decoder_variants_cross_checkpoint.png"),
        )

    originals_list     = [m["_eval_inputs"]   for m in all_metrics]
    reconstructed_list = [m["_reconstructed"] for m in all_metrics]

    # ── FE vs Transformer comparison (when transformer decoder ran) ───────────
    has_transformer = all("_tr_metrics" in m for m in all_metrics)

    if has_transformer:
        tr_metrics_list     = [m["_tr_metrics"]     for m in all_metrics]
        tr_reconstructed_list = [m["_tr_reconstructed"] for m in all_metrics]

        plot_fe_vs_transformer_comparison_bar_chart(
            fe_metrics_list=[{k: v for k, v in m.items() if not k.startswith("_")}
                             for m in all_metrics],
            tr_metrics_list=tr_metrics_list,
            run_names=run_names,
            output_path=str(output_dir / "fe_vs_transformer_bar_chart.png"),
        )

        # Cross-checkpoint architecture × FE/Transformer grouped bar chart
        has_arch_metrics = all("_variants_metrics" in m and "_tr_variants_metrics" in m
                               for m in all_metrics)
        if has_arch_metrics:
            all_fe_variants = {rn: m["_variants_metrics"]    for rn, m in zip(run_names, all_metrics)}
            all_tr_variants = {rn: m["_tr_variants_metrics"] for rn, m in zip(run_names, all_metrics)}
            plot_fe_vs_transformer_by_architecture_multi_model(
                all_fe_variants=all_fe_variants,
                all_tr_variants=all_tr_variants,
                save_path=str(output_dir / "fe_vs_transformer_architecture_multi_model.png"),
            )

    # ── Multi-model diagnostic plots (D1–D4) ─────────────────────────────────
    # When transformer decoder ran, interleave FE and transformer into one set
    # of diagnostic plots: [run1_FE, run1_Tr, run2_FE, run2_Tr, ...]
    if has_transformer:
        diag_originals = []
        diag_recon     = []
        diag_names     = []
        for i, rn in enumerate(run_names):
            diag_originals += [originals_list[i],         originals_list[i]]
            diag_recon     += [reconstructed_list[i],     tr_reconstructed_list[i]]
            diag_names     += [f"{rn} [FE]",              f"{rn} [Transf]"]
    else:
        diag_originals = originals_list
        diag_recon     = reconstructed_list
        diag_names     = run_names

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    print("\n[+] Generating diagnostic plots ...")

    plot_per_bin_error_heatmap(
        diag_originals, diag_recon, diag_names,
        save_path=str(diag_dir / "D1_per_bin_error_heatmap.png"),
    )
    print("    [+] D1 saved: D1_per_bin_error_heatmap.png")

    plot_pca_component_r2(
        diag_originals[0], diag_recon, diag_names,
        save_path=str(diag_dir / "D2_pca_component_r2.png"),
    )
    print("    [+] D2 saved: D2_pca_component_r2.png")

    plot_residual_scatter(
        diag_originals, diag_recon, diag_names,
        save_path=str(diag_dir / "D3_residual_scatter.png"),
    )
    print("    [+] D3 saved: D3_residual_scatter.png")

    # D4: FE per-sample R² vs transformer per-sample R²
    if has_transformer:
        plot_fe_vs_transformer_r2(
            fe_per_r2_list=[m["_per_r2"]    for m in all_metrics],
            transformer_per_r2_list=[m["_tr_per_r2"] for m in all_metrics],
            run_names=run_names,
            save_path=str(diag_dir / "D4_fe_vs_transformer_r2.png"),
        )
        print("    [+] D4 saved: D4_fe_vs_transformer_r2.png")
    elif args.include_embedding_decoder:
        print("    [!] D4 skipped: transformer extraction failed for one or more models.")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'Run':<35} {'Cosine':>8} {'MSE':>10} {'R²':>8}")
    print("-" * 65)
    for rn, m in zip(run_names, all_metrics):
        print(f"{rn:<35} {m['fe_dec_cosine_mean']:>8.4f} "
              f"{m['fe_dec_mse']:>10.6f} {m['fe_dec_r2']:>8.4f}")
    print(f"\n[+] All results saved to: {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    parser = argparse.ArgumentParser(
        description="FE Decoder Reconstruction Evaluation  (eval_method: decode_fe)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Checkpoint source (mirrors evaluation_runner.py) ──────────────────
    ckpt_grp = parser.add_mutually_exclusive_group(required=True)
    ckpt_grp.add_argument(
        "--checkpoint", nargs="+",
        metavar="PATH",
        help="One or more .pt checkpoint files (or dirs containing checkpoint_best.pt).",
    )
    ckpt_grp.add_argument(
        "--checkpoint_dir",
        metavar="DIR",
        help=(
            "Directory to auto-discover checkpoints. "
            "Supports flat .pt files or run subdirs with checkpoint_best.pt."
        ),
    )

    # ── Checkpoint filtering (mirrors evaluation_runner.py) ───────────────
    parser.add_argument(
        "--best_only", action="store_true",
        help="Prefer checkpoint_best.pt when scanning subdirs (default behaviour).",
    )
    parser.add_argument(
        "--latest_only", action="store_true",
        help="Prefer checkpoint_last.pt when scanning subdirs.",
    )
    parser.add_argument(
        "--run_names", nargs="+", metavar="NAME",
        help="Only evaluate runs whose name matches one of these strings.",
    )

    # ── Eval method flag (accepted for CLI compatibility; only decode_fe runs) ──
    parser.add_argument(
        "--eval_methods", nargs="+", default=["decode_fe"],
        help="Eval methods to run (only 'decode_fe' is implemented here).",
    )

    # ── Data source (mirrors evaluation_runner.py) ─────────────────────────
    data_grp = parser.add_mutually_exclusive_group(required=True)
    data_grp.add_argument(
        "--eval_data_dir", metavar="DIR",
        help="Directory with WAV files used for both decoder training and evaluation.",
    )
    data_grp.add_argument(
        "--inputs_npy", metavar="NPY",
        help="Pre-saved inputs .npy (shape [N, D]); skips WAV loading.",
    )

    # ── Sample counts (mirrors evaluation_runner.py) ───────────────────────
    parser.add_argument(
        "--max_eval_samples", type=int, default=2000,
        help=(
            "Total samples to load. Split 50/50 into train/eval by default. "
            "Set to 2000 for the recommended 1k-train / 1k-eval setup."
        ),
    )
    parser.add_argument(
        "--n_train", type=int, default=None,
        help=(
            "Override train split size. "
            "Default: half of max_eval_samples."
        ),
    )
    parser.add_argument(
        "--target_length", type=int, default=245,
        help="Expected spectrogram length when loading WAV files.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data shuffle and decoder initialisation.",
    )

    # ── Decoder training ───────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=200,
                        help="Linear decoder training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size for decoder training.")
    parser.add_argument("--fe_batch_size", type=int, default=64,
                        help="Batch size for FE forward pass.")
    parser.add_argument("--smoothness_weight", type=float, default=0.1,
                        help="Weight for second-order smoothness penalty on decoder output.")
    parser.add_argument(
        "--decoder_variants", nargs="+", default=["0", "512", "512:256"],
        metavar="SPEC",
        help=(
            "Decoder architectures to compare.  Each spec is a colon-separated list of "
            "hidden-layer sizes, or '0' for a single linear layer.\n"
            "  '0'       → Linear(512→245)                          Exp 1 (default)\n"
            "  '512'     → Linear(512→512)→ReLU→Linear(512→245)    Exp 2\n"
            "  '512:256' → …→ReLU→Linear(512→256)→ReLU→Linear→245 Exp 3\n"
            "Default: ['0', '512', '512:256'] runs all 3 experiments and compares them."
        ),
    )
    parser.add_argument(
        "--decoder_hidden", type=int, default=None,
        help=(
            "[Legacy] Single hidden-layer size.  Equivalent to "
            "--decoder_variants 0 N.  Ignored when --decoder_variants is set explicitly."
        ),
    )
    parser.add_argument(
        "--include_embedding_decoder", action="store_true",
        help=(
            "Also train a decoder on transformer embeddings (768-dim mean-pooled) "
            "and generate D4 comparison scatter of FE vs transformer R² per sample."
        ),
    )

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_dir", default="code/eval_results/fe_decoder",
        help="Root output directory (a subdir is created per run).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for FE extraction.",
    )

    args = parser.parse_args()

    # Legacy --decoder_hidden: override decoder_variants if --decoder_variants was not
    # explicitly set (i.e., still at its default).
    if args.decoder_hidden is not None:
        args.decoder_variants = ["0", str(args.decoder_hidden)]

    # Warn about unknown eval_methods (only decode_fe is implemented)
    for m in args.eval_methods:
        if m != "decode_fe":
            print(f"[!] eval_method '{m}' is not implemented in this script "
                  f"(only 'decode_fe').")

    main(args)
