"""
FE Decoder Reconstruction Evaluation
=====================================

Trains a lightweight linear decoder on top of **frozen** CNN feature-extractor
outputs to reconstruct the original input spectrogram, then evaluates how well
the reconstruction succeeds.  Better reconstruction = more input information is
preserved at the CNN stage, before the transformer processes it.

Pipeline
--------
  input [B, 245]
    → feature_extractor  → [B, 512, 47]
    → transpose          → [B, 47, 512]
    → layer_norm         → [B, 47, 512]   ← FE output (before post_extract_proj)
    → mean over time     → [B, 512]       ← pooled FE representation (used for decoding)
    → Linear(512, 245)   → [B, 245]       ← decoder (trained; FE frozen)

NOTE on dimensionality: flattening [B, 47, 512] → [B, 24064] gives a 5.9M-parameter
decoder which is 5900× overparameterised for 1k training samples and fails to learn.
Mean-pooling first yields a 512-dim representation → 125k parameters, tractable with 1k
samples (≈8 samples/parameter).

CLI mirrors evaluation_runner.py so the script can be dropped into the same
workflow.  The eval method name is ``decode_fe``.

Usage examples
--------------
# Single checkpoint — same flags as evaluation_runner.py
python code/eval_fe_decoder.py \\
    --checkpoint /path/to/checkpoint_best.pt \\
    --eval_methods decode_fe \\
    --eval_data_dir fairseq/data/nova_data/single_channel_10k \\
    --max_eval_samples 2000

# Auto-discover all checkpoints in a directory (best-only)
python code/eval_fe_decoder.py \\
    --checkpoint_dir /mnt5/noy/SpectralFM/checkpoints/runai/2026-03-10-compare-single-to-multi \\
    --eval_methods decode_fe \\
    --eval_data_dir fairseq/data/nova_data/single_channel_10k \\
    --max_eval_samples 2000

# Filter specific run names within a directory
python code/eval_fe_decoder.py \\
    --checkpoint_dir /mnt5/noy/SpectralFM/checkpoints/runai \\
    --run_names 2026-01-07_21-50-07 2026-02-25_13-46-46 \\
    --eval_methods decode_fe \\
    --eval_data_dir fairseq/data/nova_data/single_channel_10k \\
    --max_eval_samples 2000

# Use pre-saved .npy inputs (skip live WAV loading)
python code/eval_fe_decoder.py \\
    --checkpoint_dir /path/to/runai_dir \\
    --inputs_npy code/eval_results/20260317_213209/data/inputs_2026-01-07_21-50-07_structured_similarity.npy \\
    --max_eval_samples 100

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
from typing import Dict, List, Optional, Tuple

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
    decoder_hidden: int = 0,
    include_embedding_decoder: bool = False,
) -> Optional[Dict]:
    """Run the full FE-decoder evaluation for one checkpoint."""
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

    arch = f"Linear({all_fe.shape[1]}→{decoder_hidden}→ReLU→245)" if decoder_hidden > 0 else f"Linear({all_fe.shape[1]}→245)"
    print(f"[+] Training FE decoder [{arch}] (epochs={epochs}, lr={lr}, smoothness={smoothness_weight}) ...")
    metrics, reconstructed_eval = compute_fe_decoder_metrics(
        inputs=all_inputs,
        fe_outputs=all_fe,
        train_mask=train_mask,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        smoothness_weight=smoothness_weight,
        decoder_hidden=decoder_hidden,
    )

    per_cosine = metrics.pop("_per_cosine")
    per_mse    = metrics.pop("_per_mse")
    per_r2     = metrics.pop("_per_r2")
    metrics.pop("_per_mae", None)

    print(f"\n    Results for {run_name}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"      {k}: {v:.5f}")
        else:
            print(f"      {k}: {v}")

    metrics_json = {k: v for k, v in metrics.items()
                    if isinstance(v, (int, float, str, bool))}
    metrics_json["run_name"] = run_name
    metrics_json["checkpoint_path"] = checkpoint_path
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"    Saved metrics: {run_dir / 'metrics.json'}")

    np.save(run_dir / "reconstructed_eval.npy", reconstructed_eval)

    plot_fe_decoder_reconstruction_samples(
        originals=eval_inputs,
        reconstructed=reconstructed_eval,
        per_cosine=per_cosine,
        per_mse=per_mse,
        save_path=str(plots_dir / "reconstruction_samples.png"),
        title=f"FE Decoder — Reconstruction Samples\n{run_name}",
        n_panels=8,
    )
    plot_fe_decoder_score_distribution(
        per_cosine=per_cosine,
        per_mse=per_mse,
        save_path=str(plots_dir / "score_distributions.png"),
        title="FE Decoder — Score Distributions",
        run_name=run_name,
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
            tr_arch = f"Linear({all_emb.shape[1]}→{decoder_hidden}→ReLU→245)" if decoder_hidden > 0 else f"Linear({all_emb.shape[1]}→245)"
            print(f"[+] Training transformer decoder [{tr_arch}] (epochs={epochs}, lr={lr}) ...")
            tr_metrics, tr_recon_eval = compute_fe_decoder_metrics(
                inputs=all_inputs,
                fe_outputs=all_emb,
                train_mask=train_mask,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                smoothness_weight=smoothness_weight,
                decoder_hidden=decoder_hidden,
            )

            tr_per_cosine = tr_metrics.pop("_per_cosine")
            tr_per_mse    = tr_metrics.pop("_per_mse")
            tr_per_r2     = tr_metrics.pop("_per_r2")
            tr_metrics.pop("_per_mae", None)

            print(f"\n    Transformer decoder results for {run_name}:")
            for k, v in tr_metrics.items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.5f}")

            # Save transformer metrics JSON
            tr_json = {k: v for k, v in tr_metrics.items()
                       if isinstance(v, (int, float, str, bool))}
            tr_json["run_name"] = run_name
            tr_json["decoder_source"] = "transformer_embeddings"
            with open(run_dir / "metrics_transformer.json", "w") as f:
                json.dump(tr_json, f, indent=2)

            np.save(run_dir / "tr_reconstructed_eval.npy", tr_recon_eval)

            # Per-run transformer plots
            tr_plots_dir = plots_dir / "transformer"
            tr_plots_dir.mkdir(exist_ok=True)
            plot_fe_decoder_reconstruction_samples(
                originals=eval_inputs,
                reconstructed=tr_recon_eval,
                per_cosine=tr_per_cosine,
                per_mse=tr_per_mse,
                save_path=str(tr_plots_dir / "reconstruction_samples.png"),
                title=f"Transformer Decoder — Reconstruction Samples\n{run_name}",
                n_panels=8,
            )
            plot_fe_decoder_score_distribution(
                per_cosine=tr_per_cosine,
                per_mse=tr_per_mse,
                save_path=str(tr_plots_dir / "score_distributions.png"),
                title="Transformer Decoder — Score Distributions",
                run_name=run_name,
            )

            # Triple comparison plot (original / FE recon / transformer recon)
            plot_reconstruction_triple(
                originals=eval_inputs,
                fe_reconstructed=reconstructed_eval,
                tr_reconstructed=tr_recon_eval,
                per_cosine_fe=per_cosine,
                per_cosine_tr=tr_per_cosine,
                per_mse_fe=per_mse,
                per_mse_tr=tr_per_mse,
                save_path=str(plots_dir / "reconstruction_triple.png"),
                title=f"FE vs Transformer Decoder\n{run_name}",
            )
            print(f"    Saved transformer plots to: {tr_plots_dir}/")

            # Store for multi-model plots in main()
            metrics["_tr_metrics"]     = {k: v for k, v in tr_metrics.items()
                                          if isinstance(v, (int, float, str, bool))}
            metrics["_tr_reconstructed"] = tr_recon_eval
            metrics["_tr_per_cosine"]  = tr_per_cosine
            metrics["_tr_per_mse"]     = tr_per_mse
            metrics["_tr_per_r2"]      = tr_per_r2

    # Return raw arrays so main() can build multi-model diagnostic plots
    metrics["_per_cosine"]    = per_cosine
    metrics["_per_mse"]       = per_mse
    metrics["_per_r2"]        = per_r2
    metrics["_eval_inputs"]   = eval_inputs
    metrics["_reconstructed"] = reconstructed_eval
    return metrics


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
            decoder_hidden=args.decoder_hidden,
            include_embedding_decoder=args.include_embedding_decoder,
        )
        if metrics is not None:
            all_metrics.append(metrics)
            run_names.append(run_name)

    if not all_metrics:
        print("[!] No successful evaluations.")
        sys.exit(1)

    # ── Comparison bar chart (FE decoder only) ───────────────────────────────
    plot_comparison_bar_chart(
        all_metrics=[{k: v for k, v in m.items() if not k.startswith("_")}
                     for m in all_metrics],
        run_names=run_names,
        output_path=str(output_dir / "comparison_bar_chart.png"),
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
        "--decoder_hidden", type=int, default=0,
        help=(
            "If > 0, use a 2-layer MLP decoder: Linear(dim → hidden) → ReLU → Linear(hidden → 245). "
            "Default 0 = single linear layer. "
            "Recommended: 256 or 512 for a richer decoder that better tests representation quality."
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

    # Warn about unknown eval_methods (only decode_fe is implemented)
    for m in args.eval_methods:
        if m != "decode_fe":
            print(f"[!] eval_method '{m}' is not implemented in this script "
                  f"(only 'decode_fe').")

    main(args)
