"""
Multi-stage, multi-layer, multi-pooling representation extraction.

Extraction stores a MOMENT BANK — ten per-channel statistics over the
token/time axis, per stage, per component. Every (stage, pooling) readout is
then a cheap CPU subset-concatenation of that bank, so the whole
layer x pooling grid is explorable from a single GPU pass.

BACKBONE-GENERAL BY DESIGN. Two stage families, so a new backbone drops in
with zero code changes:

  - `layer0`..`layerN` — every entry of `model(...).hidden_states`. This is
    the only extraction HuggingFace guarantees on any `output_hidden_states`
    model, so it works unmodified for any Transformer encoder, not just the
    one this repo was built against.
  - `fe` / `extract_features` — the conv-feature-extractor taps this repo's
    specific data2vec-audio backbone happens to expose
    (`model.feature_extractor`, `model.feature_projection.layer_norm`).
    Extracted via `hasattr` and silently skipped for a backbone that has
    neither submodule, rather than assumed.

Display names follow the same split: `layerN` always renders as
"Transformer layer N" (backbone-agnostic). The FE/Projector names below are
this specific backbone's own block vocabulary (see ARCHITECTURE.md) and only
ever apply to stages that were actually extracted.
"""
from __future__ import annotations

import ast
import os

import numpy as np

POOL_STATS = ("mean", "std", "max", "min", "first", "last",
              "seg0", "seg1", "seg2", "seg3")

POOLINGS = {
    "mean": ("mean",),
    "mean_std": ("mean", "std"),
    "mean_max_min": ("mean", "max", "min"),
    "segment4": ("seg0", "seg1", "seg2", "seg3"),
    "first_last": ("first", "last"),
}

# This backbone's own extra taps, before the Transformer stack: `fe` is the
# raw conv output, `extract_features` is the same output post-LayerNorm (via
# `feature_projection.layer_norm`, not yet through its Linear). Named after
# the attributes they read, so `extract_bank` can hasattr-guard them without
# a backbone-specific branch.
FE_STAGE_ATTRS = {"fe": "feature_extractor", "extract_features": "feature_projection"}

# `layer0` is what the Transformer stack receives as input for THIS backbone
# (FE post-LN, projected 512->768, plus positional embedding) -- the
# Projector's contribution, before any Transformer block runs. Shown here
# only for context; nothing downstream depends on this being layer0
# specifically vs. any other hidden_states index.
KNOWN_STAGE_DISPLAY_NAMES = {
    "fe": "FE (pre-LN)",
    "extract_features": "FE (post-LN)",
    "layer0": "Projector",
}


def stage_display_name(stage: str) -> str:
    if stage in KNOWN_STAGE_DISPLAY_NAMES:
        return KNOWN_STAGE_DISPLAY_NAMES[stage]
    if stage.startswith("layer") and stage[len("layer"):].isdigit():
        return f"Transformer layer {stage[len('layer'):]}"
    return stage


def build_readout(bank: np.ndarray, comp_idx: list, pooling: str) -> np.ndarray:
    """
    bank: [N, K, len(POOL_STATS), D] for ONE stage.
    Returns [N, len(comp_idx) * len(POOLINGS[pooling]) * D], float32, laid
    out component-major then stat then channel.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; known: {sorted(POOLINGS)}")
    stat_idx = [POOL_STATS.index(s) for s in POOLINGS[pooling]]
    sel = bank[:, comp_idx, :, :][:, :, stat_idx, :]  # [N, k, n_stats, D]
    n = sel.shape[0]
    return np.ascontiguousarray(sel.reshape(n, -1), dtype=np.float32)


def _bank_stats_from_seq(seq):
    """seq: [B, T, D] time-major. Returns [B, 10, D] in POOL_STATS order."""
    import torch
    T = seq.shape[1]
    edges = np.linspace(0, T, 5).astype(int)
    parts = {
        "mean": seq.mean(dim=1), "std": seq.std(dim=1),
        "max": seq.amax(dim=1), "min": seq.amin(dim=1),
        "first": seq[:, 0, :], "last": seq[:, -1, :],
    }
    for s in range(4):
        lo, hi = int(edges[s]), int(max(edges[s] + 1, edges[s + 1]))
        parts[f"seg{s}"] = seq[:, lo:hi, :].mean(dim=1)
    return torch.stack([parts[s] for s in POOL_STATS], dim=1)


def _backbone_name(model) -> str:
    """Auto-derived, no per-backbone config: the class name is enough to
    label a run in a cross-backbone comparison, and works for any model."""
    return type(model).__name__


def extract_bank(model, signals_z: np.ndarray, device: str = "cuda",
                  batch_size: int = 64) -> dict:
    """
    signals_z: [M, 245] float32, ALREADY z-scored (features.normalize_like_fairseq).
    Returns {stage: [M, 10, D]} float32. `layer0..layerN` come from
    `output_hidden_states=True`, present on any HF encoder -- this is the
    only extraction a new backbone needs to support to work here at all.
    `fe` / `extract_features` are extracted only if the model exposes the
    named submodules (see FE_STAGE_ATTRS); silently absent otherwise.
    """
    import torch

    model.eval()
    model.to(device)
    t = torch.from_numpy(np.asarray(signals_z, dtype=np.float32))
    has_fe = all(hasattr(model, a) for a in FE_STAGE_ATTRS.values())
    out = None  # stage list is only known after the first forward pass

    with torch.no_grad():
        for i in range(0, len(t), batch_size):
            batch = t[i:i + batch_size].to(device)
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)

            fe_t = None
            if has_fe:
                fe_out = model.feature_extractor(batch)  # [B, 512, T]
                fe_t = fe_out.transpose(1, 2)  # [B, T, 512]

            hs = model(input_values=batch, output_hidden_states=True).hidden_states
            if out is None:
                stages = (["fe", "extract_features"] if has_fe else []) + \
                    [f"layer{li}" for li in range(len(hs))]
                out = {s: [] for s in stages}

            if has_fe:
                out["fe"].append(_bank_stats_from_seq(fe_t).cpu().numpy())
                ef = model.feature_projection.layer_norm(fe_t)  # post-LN
                out["extract_features"].append(_bank_stats_from_seq(ef).cpu().numpy())

            if len(hs) != len(out) - (2 if has_fe else 0):
                raise RuntimeError(
                    f"hidden_states length changed mid-run: {len(hs)} vs "
                    f"{len(out) - (2 if has_fe else 0)} on the first batch")
            for li, h in enumerate(hs):
                out[f"layer{li}"].append(_bank_stats_from_seq(h).cpu().numpy())

    return {s: np.concatenate(v, axis=0).astype(np.float32) for s, v in out.items()}


def build_bank_cache(checkpoint_path: str, labeled_data_dir: str, out_dir: str,
                      comps: tuple = tuple(range(12)), max_samples: int = 5000,
                      device: str = "cuda", batch_size: int = 64,
                      seed: int = 42) -> str:
    """One GPU pass -> <out_dir>/bank.npz. Idempotent."""
    import time

    from ..checkpoint_loader import CheckpointLoader
    from ..data_loader import load_labeled_data
    from . import features as feat

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bank.npz")
    if os.path.exists(path):
        print(f"[label_probe] bank cache hit: {path}", flush=True)
        return path

    raw, y = load_labeled_data(labeled_data_dir, max_samples=max_samples,
                                seed=seed, comps=comps)
    n, k, L = raw.shape
    print(f"[label_probe] loaded {raw.shape}, extracting bank on {device}", flush=True)

    flat_raw = raw.reshape(n * k, L).astype(np.float32)
    flat_z = feat.normalize_like_fairseq(flat_raw)

    model = CheckpointLoader.from_file(checkpoint_path)
    backbone = _backbone_name(model)
    t0 = time.time()
    bank = extract_bank(model, flat_z, device=device, batch_size=batch_size)
    print(f"[label_probe] extracted in {time.time() - t0:.1f}s "
          f"(backbone={backbone}, stages={sorted(bank)})", flush=True)

    payload = {f"bank__{s}": arr.reshape(n, k, len(POOL_STATS), arr.shape[-1])
               for s, arr in bank.items()}
    payload["input_raw"] = flat_raw.reshape(n, k, L)
    payload["input_z"] = flat_z.reshape(n, k, L)
    payload["y"] = y
    payload["_meta"] = np.array([repr({
        "checkpoint": checkpoint_path, "backbone": backbone, "comps": comps,
        "n": int(n), "max_samples": max_samples, "seed": seed,
        "pool_stats": POOL_STATS, "stages": sorted(bank)})])

    # Atomic write: a killed run (OOM, preemption) must never leave a
    # truncated bank.npz behind, since the idempotency check above is a
    # plain os.path.exists.
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            np.savez(f, **payload)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f"[label_probe] wrote {path} ({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)
    return path


def load_bank_cache(path: str):
    """Returns (bank {stage: [N,K,10,D]}, input_raw, input_z, y, meta)."""
    data = np.load(path, allow_pickle=False)
    bank = {k[len("bank__"):]: data[k] for k in data.files if k.startswith("bank__")}
    meta = ast.literal_eval(str(data["_meta"][0]))
    return bank, data["input_raw"], data["input_z"], data["y"], meta
