#!/usr/bin/env python3
"""
Linear regression probe: predict parameter_0 from raw inputs vs transformer embeddings.

Uses ``compute_linear_probing_metrics`` (ridge CV) from eval_metrics.py.
Metrics are prefixed ``label_reg_input_*`` and ``label_reg_emb_*``; improvement is
``label_reg_improvement_r2`` = emb R² − input R².

Example:
  python code/eval_label_regression.py \\
    --checkpoint_dir /path/to/checkpoints \\
    --labeled_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FAIRSEQ = os.path.join(_THIS_DIR, "..", "fairseq")
if _FAIRSEQ not in sys.path:
    sys.path.insert(0, _FAIRSEQ)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eval_fe_decoder import (  # noqa: E402
    discover_checkpoints,
    extract_embeddings_from_inputs,
    _resolve_output_labels,
)
from eval_metrics import compute_linear_probing_metrics  # noqa: E402


def _rename_probe_metrics(m: Dict[str, Any], kind: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in m.items():
        if isinstance(v, str):
            out[f"label_reg_{kind}_error"] = v
            continue
        if not isinstance(v, (int, float, np.floating)):
            continue
        tail = k.replace("downstream_ridge_probe_", "")
        out[f"label_reg_{kind}_{tail}"] = float(v)
    return out


def load_labeled_spectrograms(
    labeled_data_dir: str,
    max_samples: int,
    target_length: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    labels_path = os.path.join(labeled_data_dir, "labels.tsv")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(labels_path)
    pairs: List[Tuple[str, float]] = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], float(parts[1])))
    rng = np.random.default_rng(seed)
    if len(pairs) > max_samples:
        idx = rng.choice(len(pairs), max_samples, replace=False)
        pairs = [pairs[i] for i in idx]

    import glob

    wav_root = labeled_data_dir
    for sub in ("wav", "wavs"):
        cand = os.path.join(labeled_data_dir, sub)
        if glob.glob(os.path.join(cand, "*.wav")):
            wav_root = cand
            break

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError("soundfile is required") from e

    inputs: List[np.ndarray] = []
    ys: List[float] = []
    for fname, y in pairs:
        fp = os.path.join(wav_root, fname)
        if not os.path.isfile(fp):
            fp = os.path.join(labeled_data_dir, fname)
        if not os.path.isfile(fp):
            continue
        data, _ = sf.read(fp, dtype="float32")
        data = np.asarray(data).flatten()
        if len(data) >= target_length:
            row = data[:target_length].astype(np.float32, copy=False)
        else:
            row = np.zeros(target_length, dtype=np.float32)
            row[: len(data)] = data.astype(np.float32, copy=False)
        inputs.append(row)
        ys.append(y)
    if not inputs:
        raise RuntimeError(f"No labeled wavs loaded under {labeled_data_dir}")
    return np.stack(inputs, axis=0), np.array(ys, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="Label regression (parameter_0 linear probe)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", nargs="+", help="Checkpoint .pt file(s)")
    src.add_argument("--checkpoint_dir", help="Directory with checkpoints (flat .pt or subdirs)")
    p.add_argument(
        "--labeled_data_dir",
        default="/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data",
        help="Dataset with labels.tsv and wavs/",
    )
    p.add_argument("--max_samples", type=int, default=2000, help="Labeled samples (subsampled with seed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_length", type=int, default=245)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--output_dir",
        default=None,
        help="Save label_reg_results.json here (default: cwd)",
    )
    p.add_argument("--device", default=None)
    args = p.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints: List[Tuple[str, str]] = []
    if args.checkpoint:
        for cp in args.checkpoint:
            checkpoints.append((os.path.basename(cp), cp))
    else:
        checkpoints.extend(
            discover_checkpoints(args.checkpoint_dir, best_only=False, latest_only=False)
        )
    if not checkpoints:
        print("[!] No checkpoints")
        sys.exit(1)
    checkpoints = _resolve_output_labels(checkpoints)

    print(f"[+] Loading up to {args.max_samples} labeled samples from {args.labeled_data_dir} ...")
    X, y = load_labeled_spectrograms(
        args.labeled_data_dir,
        max_samples=args.max_samples,
        target_length=args.target_length,
        seed=args.seed,
    )
    print(f"    Loaded {len(y)} samples, shape X={X.shape}")

    out_root = args.output_dir or os.getcwd()
    os.makedirs(out_root, exist_ok=True)

    all_results = []
    print(f"\n{'Run':<50} {'in_R²':>8} {'emb_R²':>8} {'ΔR²':>8}")
    print("-" * 78)
    for label, ckpt_path in checkpoints:
        emb = extract_embeddings_from_inputs(
            X, ckpt_path, device_str=device, batch_size=args.batch_size
        )
        if emb is None:
            print(f"[!] Skip {label}: embedding extraction failed")
            continue
        m_in = compute_linear_probing_metrics(X, y, probe_type="ridge", task="regression")
        m_emb = compute_linear_probing_metrics(emb, y, probe_type="ridge", task="regression")
        rin = _rename_probe_metrics(m_in, "input")
        rem = _rename_probe_metrics(m_emb, "emb")
        r2_in = rin.get("label_reg_input_r2", float("nan"))
        r2_emb = rem.get("label_reg_emb_r2", float("nan"))
        merged = {**rin, **rem}
        merged["label_reg_improvement_r2"] = float(r2_emb - r2_in)
        merged["checkpoint"] = ckpt_path
        merged["run_label"] = label
        merged["n_samples"] = int(len(y))
        all_results.append(merged)
        print(f"{label:<50} {r2_in:8.4f} {r2_emb:8.4f} {merged['label_reg_improvement_r2']:8.4f}")

        sub = os.path.join(out_root, label)
        os.makedirs(sub, exist_ok=True)
        with open(os.path.join(sub, "label_regression.json"), "w") as f:
            json.dump(merged, f, indent=2)

    summary_path = os.path.join(out_root, "label_reg_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Wrote {summary_path}")

    if all_results:
        _plot_label_regression(all_results, out_root)


def _plot_label_regression(results: List[Dict[str, Any]], out_root: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["run_label"] for r in results]
    r2_in = [r.get("label_reg_input_r2", 0.0) for r in results]
    r2_emb = [r.get("label_reg_emb_r2", 0.0) for r in results]
    delta = [r.get("label_reg_improvement_r2", 0.0) for r in results]
    n = len(labels)
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 3), 5))
    fig.suptitle("Label Regression — Ridge Probe (parameter_0)", fontsize=13, fontweight="bold")

    # Panel 1: side-by-side R²
    ax = axes[0]
    b1 = ax.bar(x - w / 2, r2_in, w, color="#90CAF9", edgecolor="white", label="Input R²")
    b2 = ax.bar(x + w / 2, r2_emb, w, color="#1565C0", edgecolor="white", label="Embedding R²")
    for bar, v in zip(b1, r2_in):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8, color="#555")
    for bar, v in zip(b2, r2_emb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8, color="#1565C0", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("R²")
    ax.set_title("Input vs Embedding R²")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    # Panel 2: ΔR²
    ax2 = axes[1]
    colors = ["#2E7D32" if d > 0 else "#C62828" for d in delta]
    bars = ax2.bar(x, delta, 0.5, color=colors, edgecolor="white")
    for bar, v in zip(bars, delta):
        y_off = 0.001 if v >= 0 else -0.003
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_off,
                 f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("ΔR² (embedding − input)")
    ax2.set_title("Improvement over Raw Input")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plots_dir = os.path.join(out_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, "label_regression_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved plot: {save_path}")


if __name__ == "__main__":
    main()
