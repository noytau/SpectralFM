# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Cosine similarity heatmaps during data2vec_audio training (rank 0 only).
#
# What runs when triggered (epoch_cosim_enable=True, subset file present):
#   1) Load fixed train indices from epoch_cosim_subset_path (.npy/.txt/.json); cap at epoch_cosim_max_samples.
#   2) For each micro-batch of indices, collate from task.dataset("train"), forward in eval mode via
#      model.extract_cosim_epoch_features → pooled input / FE / transformer embedding vectors (numpy).
#   3) Concatenate batches; compute sklearn cosine_similarity matrices for the three representations.
#   4) Save a 3-panel PNG under checkpoint.save_dir / epoch_cosim_output_subdir:
#        step_{updates:07d}_cosim_triple.png  (when step_label is set — every N updates), or
#        epoch_{epoch:03d}_cosim_triple.png   (when epoch_cosim_interval_updates=0, end-of-epoch only).
#   5) Log mean/std of upper triangle to WandB under epoch_cosim/* with step=trainer.get_num_updates().
#
# Scheduling: fairseq_cli/train.py calls AudioPretrainingTask.maybe_run_epoch_cosim_on_interval after each
# successful train_step when epoch_cosim_interval_updates > 0 and num_updates % N == 0. If interval is 0,
# only task.end_epoch invokes this (once per epoch).

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_subset_indices(path: str, max_samples: int, task=None) -> List[int]:
    """Load 0-based train indices from JSON, .npy, or .txt."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"epoch_cosim subset not found: {path}")

    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        entries = payload.get("entries", payload if isinstance(payload, list) else [])
        if task is not None and hasattr(task, "cfg") and hasattr(task.cfg, "data"):
            data_root = os.path.normpath(os.path.expanduser(task.cfg.data))
            base = os.path.basename(data_root.rstrip(os.sep))
            indices = []
            for e in entries:
                ds = e.get("dataset")
                ep = os.path.normpath(os.path.expanduser(e.get("path", "")))
                if ds == base or ep == data_root:
                    indices.append(int(e["index"]))
            if not indices:
                raise ValueError(
                    f"epoch_cosim: JSON has no entries matching task.data={task.cfg.data!r}"
                )
        else:
            indices = [int(e["index"]) for e in entries]
    elif path.endswith(".npy"):
        arr = np.load(path)
        indices = np.atleast_1d(arr).astype(np.int64).flatten().tolist()
        indices = [int(x) for x in indices]
    else:
        arr = np.loadtxt(path, dtype=np.int64)
        indices = [int(x) for x in np.atleast_1d(arr).astype(np.int64).flatten().tolist()]

    if max_samples > 0:
        indices = indices[:max_samples]
    return indices


def upper_triangle_mean_std(sim: np.ndarray) -> Tuple[float, float]:
    n = sim.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    iu = np.triu_indices(n, k=1)
    vals = sim[iu]
    return float(vals.mean()), float(vals.std())


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity

    if vectors.shape[0] < 2:
        return np.ones((vectors.shape[0], vectors.shape[0]), dtype=np.float64)
    return cosine_similarity(vectors.astype(np.float64, copy=False))


def plot_three_cosim_panels(
    sim_input: np.ndarray,
    sim_fe: np.ndarray,
    sim_emb: np.ndarray,
    out_path: str,
    mean_std: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
) -> None:
    import matplotlib.pyplot as plt

    ms_in, ms_fe, ms_emb = mean_std
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = [
        f"Input\nMean={ms_in[0]:.3f} Std={ms_in[1]:.3f}",
        f"FE (layer_norm pooled)\nMean={ms_fe[0]:.3f} Std={ms_fe[1]:.3f}",
        f"Transformer emb\nMean={ms_emb[0]:.3f} Std={ms_emb[1]:.3f}",
    ]
    mats = [sim_input, sim_fe, sim_emb]
    for ax, sim, title in zip(axes, mats, titles):
        if sim is not None and sim.shape[0] > 1:
            im = ax.imshow(sim, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("idx")
        ax.set_ylabel("idx")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def collect_cosim_arrays(trainer, task, indices: List[int], micro_batch: int):
    import torch

    dataset = task.dataset("train")
    collater = dataset.collater
    model = _unwrap_model(trainer.get_model())
    if not hasattr(model, "extract_cosim_epoch_features"):
        raise RuntimeError("Model has no extract_cosim_epoch_features (expected data2vec_audio).")

    device = trainer.device
    was_training = model.training
    model.eval()

    in_list, fe_list, emb_list = [], [], []

    try:
        with torch.no_grad():
            for start in range(0, len(indices), micro_batch):
                chunk = indices[start : start + micro_batch]
                samples = [dataset[i] for i in chunk]
                sample = collater(samples)
                if not sample or "net_input" not in sample:
                    continue
                net = sample["net_input"]
                source = net["source"].to(device)
                padding_mask = net.get("padding_mask")
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device)

                inp, fe, emb = model.extract_cosim_epoch_features(source, padding_mask)
                in_list.append(inp.float().cpu().numpy())
                fe_list.append(fe.float().cpu().numpy())
                emb_list.append(emb.float().cpu().numpy())
    finally:
        if was_training:
            model.train()

    if not in_list:
        raise RuntimeError("epoch_cosim: no samples collected (empty batches?)")

    inputs = np.concatenate(in_list, axis=0)
    fes = np.concatenate(fe_list, axis=0)
    embs = np.concatenate(emb_list, axis=0)
    return inputs, fes, embs


def maybe_run_epoch_cosim(
    epoch: int, trainer, task, *, step_label: Optional[int] = None
) -> None:
    from fairseq import distributed_utils

    if not distributed_utils.is_master(trainer.cfg.distributed_training):
        return

    model = _unwrap_model(trainer.get_model())
    cfg = getattr(model, "cfg", None)
    if cfg is None or not getattr(cfg, "epoch_cosim_enable", False):
        return
    subset_path = getattr(cfg, "epoch_cosim_subset_path", None)
    if not subset_path:
        logger.warning("epoch_cosim_enable=True but epoch_cosim_subset_path is unset; skipping.")
        return

    max_n = int(getattr(cfg, "epoch_cosim_max_samples", 100))
    micro = int(getattr(cfg, "epoch_cosim_micro_batch", 8))
    subdir = getattr(cfg, "epoch_cosim_output_subdir", "cosim_epoch")

    try:
        indices = load_subset_indices(subset_path, max_n, task)
    except Exception as e:
        logger.error("epoch_cosim: failed to load subset: %s", e)
        return

    if len(indices) < 2:
        logger.warning("epoch_cosim: need at least 2 indices, got %d", len(indices))
        return

    save_dir = trainer.cfg.checkpoint.save_dir
    out_dir = os.path.join(save_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    try:
        inputs, fes, embs = collect_cosim_arrays(trainer, task, indices, micro)
    except Exception as e:
        logger.exception("epoch_cosim: collection failed: %s", e)
        return

    sim_in = cosine_similarity_matrix(inputs)
    sim_fe = cosine_similarity_matrix(fes)
    sim_emb = cosine_similarity_matrix(embs)

    ms_in = upper_triangle_mean_std(sim_in)
    ms_fe = upper_triangle_mean_std(sim_fe)
    ms_emb = upper_triangle_mean_std(sim_emb)

    if step_label is not None:
        png_path = os.path.join(out_dir, f"step_{step_label:07d}_cosim_triple.png")
    else:
        png_path = os.path.join(out_dir, f"epoch_{epoch:03d}_cosim_triple.png")
    try:
        plot_three_cosim_panels(sim_in, sim_fe, sim_emb, png_path, (ms_in, ms_fe, ms_emb))
    except Exception as e:
        logger.exception("epoch_cosim: plot failed: %s", e)

    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "epoch_cosim/input_mean": ms_in[0],
                    "epoch_cosim/input_std": ms_in[1],
                    "epoch_cosim/fe_mean": ms_fe[0],
                    "epoch_cosim/fe_std": ms_fe[1],
                    "epoch_cosim/embedding_mean": ms_emb[0],
                    "epoch_cosim/embedding_std": ms_emb[1],
                    "epoch_cosim/triple_heatmap": wandb.Image(png_path),
                },
                step=trainer.get_num_updates(),
            )
    except Exception as e:
        logger.debug("epoch_cosim: wandb log skipped: %s", e)

    logger.info(
        "epoch_cosim epoch=%d updates=%d step_label=%s saved=%s in_mean=%.4f emb_mean=%.4f",
        epoch,
        trainer.get_num_updates(),
        step_label,
        png_path,
        ms_in[0],
        ms_emb[0],
    )
