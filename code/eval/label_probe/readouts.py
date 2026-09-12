"""
Multi-stage, multi-layer, multi-pooling representation extraction.

Extraction stores a MOMENT BANK — ten per-channel statistics over the
token/time axis, per stage, per component. Every (stage, pooling) readout is
then a cheap CPU subset-concatenation of that bank, so the whole
layer x pooling grid is explorable from a single GPU pass.

Stages, in the model's own block vocabulary (FE / Projector / Transformer,
per ARCHITECTURE.md): `fe` is the FE (conv feature extractor) output,
pre-LayerNorm; `extract_features` is the FE output post-LayerNorm (this is
literally "conv FE output (post-LayerNorm)" in ARCHITECTURE.md's FE-decoder
row — still 512-d, the LayerNorm's own submodule of `feature_projection`,
not yet through its Linear). `layer0` is what the Transformer actually
receives as input: FE post-LN, projected 512->768 by the Projector's Linear,
plus positional conv embedding and the encoder's own pre-block LayerNorm —
displayed as "Projector" since it is the Projector's contribution to the
pipeline, immediately before any Transformer block runs. `layer1`..`layer12`
are Transformer block 1..12 outputs — the standard "final-layer, mean-pool"
convention used elsewhere in this codebase's eval package is exactly
(stage="layer12", pooling="mean"), displayed as "Transformer layer 12".
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

N_TRANSFORMER_LAYERS = 13  # HF hidden_states = embeddings output + 12 blocks
BANK_STAGES = ("fe", "extract_features") + tuple(
    f"layer{i}" for i in range(N_TRANSFORMER_LAYERS))

# Internal bank keys ("fe", "extract_features", "layer0".."layer12") stay as
# they are -- they're cache keys, matched against an on-disk bank.npz, and
# renaming them would invalidate every cached extraction. This maps a key to
# the model's own block vocabulary for anything user-facing (labels, prints,
# reports): FE / Projector / Transformer, never "readout" or a bare "layerN".
STAGE_DISPLAY_NAMES = {
    "fe": "FE (pre-LN)",
    "extract_features": "FE (post-LN)",
    "layer0": "Projector",
    **{f"layer{i}": f"Transformer layer {i}" for i in range(1, N_TRANSFORMER_LAYERS)},
}


def stage_display_name(stage: str) -> str:
    return STAGE_DISPLAY_NAMES.get(stage, stage)


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


def _bank_stats_from_seq(seq) -> "torch.Tensor":
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


def extract_bank(model, signals_z: np.ndarray, device: str = "cuda",
                  batch_size: int = 64) -> dict:
    """
    signals_z: [M, 245] float32, ALREADY z-scored (features.normalize_like_fairseq).
    Returns {stage: [M, 10, D]} float32 for every stage in BANK_STAGES.
    One forward pass per batch with output_hidden_states=True yields all 13
    transformer taps; the two FE stages come from the same pass's submodules.
    """
    import torch

    model.eval()
    model.to(device)
    t = torch.from_numpy(np.asarray(signals_z, dtype=np.float32))
    out = {s: [] for s in BANK_STAGES}

    with torch.no_grad():
        for i in range(0, len(t), batch_size):
            batch = t[i:i + batch_size].to(device)
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)

            fe_out = model.feature_extractor(batch)  # [B, 512, T]
            fe_t = fe_out.transpose(1, 2)  # [B, T, 512]
            out["fe"].append(_bank_stats_from_seq(fe_t).cpu().numpy())

            ef = model.feature_projection.layer_norm(fe_t)  # post-LN
            out["extract_features"].append(_bank_stats_from_seq(ef).cpu().numpy())

            hs = model(input_values=batch, output_hidden_states=True).hidden_states
            if len(hs) != N_TRANSFORMER_LAYERS:
                raise RuntimeError(
                    f"expected {N_TRANSFORMER_LAYERS} hidden states, got {len(hs)}")
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
    t0 = time.time()
    bank = extract_bank(model, flat_z, device=device, batch_size=batch_size)
    print(f"[label_probe] extracted in {time.time() - t0:.1f}s", flush=True)

    payload = {f"bank__{s}": arr.reshape(n, k, len(POOL_STATS), arr.shape[-1])
               for s, arr in bank.items()}
    payload["input_raw"] = flat_raw.reshape(n, k, L)
    payload["input_z"] = flat_z.reshape(n, k, L)
    payload["y"] = y
    payload["_meta"] = np.array([repr({
        "checkpoint": checkpoint_path, "comps": comps, "n": int(n),
        "max_samples": max_samples, "seed": seed,
        "pool_stats": POOL_STATS, "stages": BANK_STAGES})])

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
