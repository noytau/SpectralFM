#!/usr/bin/env python3
"""Reconstruction analysis for a fairseq Hydra checkpoint (data2vec_audio + recon_trans).

Loads a checkpoint, evaluates the transformer-branch reconstruction decoder on
either a TSV-driven split (train.tsv / valid.tsv) or a raw wav directory, and
produces best/worst sample plots (sorted by per-sample MSE).

Outputs (per split):
  - best_<k>_<split>.png       — k samples with the LOWEST L2 (overlay target+pred)
  - worst_<k>_<split>.png      — k samples with the HIGHEST L2 (overlay target+pred)
  - mse_histogram_<split>.png  — histogram of per-sample MSE
  - summary_<split>.txt        — aggregate stats (mean/std/min/max/median MSE, target var)
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (
    os.path.join(_REPO, "fairseq"),
    os.path.join(_REPO, "fairseq", "examples"),
    os.path.dirname(__file__),
):
    if p not in sys.path:
        sys.path.insert(0, p)


def _read_tsv_manifest(tsv_path: str) -> Tuple[str, List[str]]:
    """Return (audio_root, relative wav names) from a fairseq TSV manifest."""
    with open(tsv_path) as f:
        root = f.readline().strip()
        names = [line.strip().split("\t")[0] for line in f if line.strip()]
    if not names:
        raise FileNotFoundError(f"empty manifest: {tsv_path}")
    return root, names


def _load_wav_at(path: str, recon_len: int) -> torch.Tensor:
    import soundfile as sf

    x, _ = sf.read(path, dtype="float32")
    if x.ndim == 2:
        x = x[:, 0]
    if x.shape[0] < recon_len:
        raise ValueError(f"{path}: length {x.shape[0]} < recon_len={recon_len}")
    return torch.from_numpy(x[:recon_len])


def _load_from_tsv_at_indices(
    tsv_path: str,
    indices: List[int],
    recon_len: int,
) -> Tuple[torch.Tensor, List[str]]:
    """Load only manifest rows ``indices`` (fast for large TSVs)."""
    root, names = _read_tsv_manifest(tsv_path)
    wavs: List[torch.Tensor] = []
    used: List[str] = []
    for i in indices:
        n = names[i]
        path = os.path.join(root, n)
        if not os.path.exists(path):
            continue
        wavs.append(_load_wav_at(path, recon_len))
        used.append(n)
    if not wavs:
        raise FileNotFoundError(f"no usable wavs at indices {indices[:5]}... in {tsv_path}")
    return torch.stack(wavs), used


def _load_from_tsv(tsv_path: str, recon_len: int) -> Tuple[torch.Tensor, List[str]]:
    """Load every wav listed in a fairseq TSV manifest, truncated to recon_len."""
    _, names = _read_tsv_manifest(tsv_path)
    return _load_from_tsv_at_indices(tsv_path, list(range(len(names))), recon_len)


def _load_from_dir(data_dir: str, recon_len: int, n: int) -> Tuple[torch.Tensor, List[str]]:
    import glob
    import soundfile as sf

    paths = sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:n]
    if not paths:
        raise FileNotFoundError(f"no wav files in {data_dir}")
    wavs, names = [], []
    for p in paths:
        x, _ = sf.read(p, dtype="float32")
        if x.ndim == 2:
            x = x[:, 0]
        if x.shape[0] < recon_len:
            continue
        wavs.append(torch.from_numpy(x[:recon_len]))
        names.append(os.path.basename(p))
    return torch.stack(wavs), names


@torch.no_grad()
def _predict(model, source: torch.Tensor) -> torch.Tensor:
    """Run encoder + trans_recon_decoder in eval mode (mask=False, deterministic)."""
    enc = model(source, padding_mask=None, mask=False, features_only=True)
    x = enc["x"]
    if getattr(model.trans_recon_decoder, "needs_full_sequence", False):
        pred = model.trans_recon_decoder(x)
    else:
        tp = model._mean_pool(x, None)
        pred = model.trans_recon_decoder(tp)
    return pred


def _plot_grid(
    target: torch.Tensor,
    pred: torch.Tensor,
    per_l2: torch.Tensor,
    names: List[str],
    indices: List[int],
    title: str,
    save_path: str,
    cols: int = 2,
) -> None:
    n = len(indices)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 2.0 * rows), squeeze=False)
    for ax_idx, sample_idx in enumerate(indices):
        ax = axes[ax_idx // cols, ax_idx % cols]
        t = target[sample_idx].cpu().numpy()
        p = pred[sample_idx].detach().cpu().numpy()
        ax.plot(t, color="#1f77b4", lw=1.2, label="target")
        ax.plot(p, color="#d62728", lw=1.0, alpha=0.85, label="recon")
        ax.set_title(f"{names[sample_idx][:32]}   L2={float(per_l2[sample_idx]):.4f}", fontsize=8)
        ax.tick_params(labelsize=7)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=7)
        ax.set_ylim(-1.0, 2.0)
    for empty in range(n, rows * cols):
        axes[empty // cols, empty % cols].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram(per_l2: torch.Tensor, target_var: float, split_label: str, save_path: str) -> None:
    arr = per_l2.cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(arr, bins=30, color="#2c7fb8", edgecolor="white", alpha=0.85)
    ax.axvline(arr.mean(), color="#d62728", ls="--", lw=1.3, label=f"mean = {arr.mean():.4f}")
    ax.axvline(target_var, color="gray", ls=":", lw=1.3, label=f"predict-mean baseline = {target_var:.4f}")
    ax.set_xlabel("per-sample MSE")
    ax.set_ylabel("count")
    ax.set_title(f"{split_label}: per-sample MSE distribution (n={len(arr)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fairseq checkpoint_*.pt path")
    ap.add_argument(
        "--tsv",
        action="append",
        default=None,
        help="path to a fairseq TSV manifest. Can repeat for multiple splits "
             "(e.g. --tsv .../train.tsv --tsv .../valid.tsv). Overrides --data_dir.",
    )
    ap.add_argument(
        "--wav_list",
        default=None,
        help="text file with one absolute wav path per line (panel mode; overrides --tsv/--data_dir).",
    )
    ap.add_argument(
        "--sample_indices",
        type=int,
        nargs="*",
        default=None,
        help="with a single --tsv: load only these manifest row indices (0-based).",
    )
    ap.add_argument(
        "--data_dir",
        default="fairseq/data/nova_data/single_channel_100/wav",
        help="fallback: load up to --n wavs from this directory.",
    )
    ap.add_argument("--n", type=int, default=200, help="cap on samples per split")
    ap.add_argument("--k_best", type=int, default=6, help="# of best samples to plot")
    ap.add_argument("--k_worst", type=int, default=6, help="# of worst samples to plot")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--out_dir", default="code/eval_results/hydra_recon_analysis")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--label", default="", help="extra label appended to filenames / titles")
    args = ap.parse_args()

    from fairseq import checkpoint_utils
    from data2vec.models.data2vec_audio import Data2VecAudioModel

    device = torch.device(args.device)
    print(f"Loading checkpoint: {args.ckpt}")
    state = checkpoint_utils.load_checkpoint_to_cpu(args.ckpt, {})
    cfg = state["cfg"]["model"]
    model = Data2VecAudioModel(cfg)
    model.load_state_dict(state["model"], strict=False)
    model.eval().to(device)

    recon_len = int(getattr(cfg, "recon_output_dim", 245))

    splits: List[Tuple[str, str]] = []
    panel_paths: Optional[List[str]] = None
    if args.wav_list:
        with open(args.wav_list) as f:
            panel_paths = [ln.strip() for ln in f if ln.strip()]
        if not panel_paths:
            raise FileNotFoundError(f"empty wav list: {args.wav_list}")
        splits.append(("panel", args.wav_list))
    elif args.tsv:
        for t in args.tsv:
            split_name = os.path.splitext(os.path.basename(t))[0]
            splits.append((split_name, t))
    else:
        splits.append(("dir", args.data_dir))

    os.makedirs(args.out_dir, exist_ok=True)

    if not (hasattr(model, "trans_recon_decoder") and float(getattr(cfg, "lambda_recon_trans", 0)) > 0):
        print("[!] lambda_recon_trans=0 or no trans_recon_decoder — nothing to plot.")
        return

    for split_name, source_path in splits:
        if panel_paths is not None:
            wavs, names = [], []
            for p in panel_paths:
                if not os.path.isfile(p):
                    print(f"  skip missing: {p}")
                    continue
                wavs.append(_load_wav_at(p, recon_len))
                names.append(os.path.basename(p))
            if not wavs:
                raise FileNotFoundError(f"no wavs loaded from {args.wav_list}")
            source = torch.stack(wavs)
        elif args.tsv:
            if args.sample_indices is not None:
                if len(args.tsv) != 1:
                    raise ValueError("--sample_indices requires exactly one --tsv")
                source, names = _load_from_tsv_at_indices(
                    source_path, list(args.sample_indices), recon_len,
                )
            else:
                source, names = _load_from_tsv(source_path, recon_len)
        else:
            source, names = _load_from_dir(source_path, recon_len, args.n)
        source = source.to(device)
        if source.shape[0] > args.n:
            source = source[: args.n]
            names = names[: args.n]
        target = source[:, :recon_len].float()
        print(f"[{split_name}] loaded {source.shape[0]} samples from {source_path}")

        pred = _predict(model, source).float()
        if pred.shape[-1] != recon_len:
            pred = pred[..., :recon_len]
        per_l2 = F.mse_loss(pred, target, reduction="none").mean(dim=1)
        order_worst_to_best = torch.argsort(per_l2, descending=True)
        worst_idx = order_worst_to_best[: args.k_worst].cpu().tolist()
        best_idx = order_worst_to_best[-args.k_best:].cpu().tolist()[::-1]

        target_var = float(target.var().item())
        agg_mse = float(per_l2.mean().item())
        median_mse = float(per_l2.median().item())
        min_mse = float(per_l2.min().item())
        max_mse = float(per_l2.max().item())

        print(
            f"[{split_name}] MSE  mean={agg_mse:.4f}  median={median_mse:.4f}  "
            f"min={min_mse:.4f}  max={max_mse:.4f}  target_var={target_var:.4f}"
        )

        suffix = f"_{args.label}" if args.label else ""
        ckpt_tag = os.path.splitext(os.path.basename(args.ckpt))[0]
        display_name = args.label if args.label else split_name
        title_prefix = f"{ckpt_tag} — {display_name}"

        _plot_grid(
            target, pred, per_l2, names, best_idx,
            f"{title_prefix} — BEST {args.k_best} (mean MSE on split = {agg_mse:.4f})",
            os.path.join(args.out_dir, f"best_{args.k_best}_{split_name}{suffix}.png"),
            cols=args.cols,
        )
        _plot_grid(
            target, pred, per_l2, names, worst_idx,
            f"{title_prefix} — WORST {args.k_worst} (mean MSE on split = {agg_mse:.4f})",
            os.path.join(args.out_dir, f"worst_{args.k_worst}_{split_name}{suffix}.png"),
            cols=args.cols,
        )
        _plot_histogram(
            per_l2, target_var, f"{title_prefix} — n={source.shape[0]}",
            os.path.join(args.out_dir, f"mse_histogram_{split_name}{suffix}.png"),
        )

        with open(os.path.join(args.out_dir, f"summary_{split_name}{suffix}.txt"), "w") as f:
            f.write(
                f"checkpoint: {args.ckpt}\nsplit: {split_name}\nn_samples: {source.shape[0]}\n"
                f"recon_output_dim: {recon_len}\n"
                f"target_var: {target_var:.6f}\n"
                f"mse_mean:   {agg_mse:.6f}\n"
                f"mse_median: {median_mse:.6f}\n"
                f"mse_min:    {min_mse:.6f}\n"
                f"mse_max:    {max_mse:.6f}\n"
                f"best_indices: {best_idx}\n"
                f"worst_indices: {worst_idx}\n"
            )
        print(f"[{split_name}] wrote best_/worst_/mse_histogram/summary files in {args.out_dir}/")


if __name__ == "__main__":
    main()
