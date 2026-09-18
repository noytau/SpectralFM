"""
Readout construction for step 6 -- the layer x pooling axis (H-A, H-B).

Design note: rather than re-running the GPU for every pooling, extraction
stores a MOMENT BANK -- ten per-channel statistics over the 47 tokens, per
stage, per component. Every pooling in POOLINGS is a subset-concatenation of
that bank, so the entire pooling axis is explorable on CPU. See
docs/superpowers/specs/2026-09-09-few-shot-readout-geometry-design.md sec. 4.

taps.py is deliberately untouched: steps 1-5 lineage depends on it being
byte-identical, and its mean-pooled last-layer output is one cell of this
grid (BANK_STAGES[-1], POOLINGS['mean']).
"""
from __future__ import annotations

import ast

import numpy as np

# Order matters: it is the layout of axis 2 of every bank array, and it is
# baked into the on-disk cache. Never reorder -- append only.
POOL_STATS = ("mean", "std", "max", "min", "first", "last",
              "seg0", "seg1", "seg2", "seg3")

POOLINGS = {
    "mean": ("mean",),
    "mean_std": ("mean", "std"),
    "mean_max_min": ("mean", "max", "min"),
    "segment4": ("seg0", "seg1", "seg2", "seg3"),
    "first_last": ("first", "last"),
    "mean_std_max_min": ("mean", "std", "max", "min"),
}

N_TRANSFORMER_LAYERS = 13   # HF hidden_states = embeddings output + 12 blocks
BANK_STAGES = ("fe", "extract_features") + tuple(
    f"layer{i}" for i in range(N_TRANSFORMER_LAYERS))


def build_readout(bank: np.ndarray, comp_idx: list, pooling: str) -> np.ndarray:
    """
    bank: [N, K, len(POOL_STATS), D] for ONE stage.
    Returns [N, len(comp_idx) * len(POOLINGS[pooling]) * D], float32,
    laid out component-major then stat then channel -- the same wide layout
    features.make_wide produces, so the non-exchangeability guard still holds.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; known: {sorted(POOLINGS)}")
    stat_idx = [POOL_STATS.index(s) for s in POOLINGS[pooling]]
    sel = bank[:, comp_idx, :, :][:, :, stat_idx, :]      # [N, k, n_stats, D]
    n = sel.shape[0]
    return np.ascontiguousarray(sel.reshape(n, -1), dtype=np.float32)


def raw_moments(signal: np.ndarray) -> np.ndarray:
    """
    signal: [N, K, L] raw 1D spectra. Returns [N, K, len(POOL_STATS)].

    The raw-input analogue of a pooled embedding. Raw input has no channel
    axis to pool over, so this is not a true symmetric counterpart to
    POOLINGS -- it is offered as an extra candidate in the raw family so the
    raw baseline is as strong as we can make it. See spec sec. 6.
    """
    n, k, L = signal.shape
    x = signal.astype(np.float32)
    edges = np.linspace(0, L, 5).astype(int)
    stats = {
        "mean": x.mean(axis=-1), "std": x.std(axis=-1),
        "max": x.max(axis=-1), "min": x.min(axis=-1),
        "first": x[..., 0], "last": x[..., -1],
    }
    for s in range(4):
        lo, hi = edges[s], max(edges[s] + 1, edges[s + 1])
        stats[f"seg{s}"] = x[..., lo:hi].mean(axis=-1)
    return np.stack([stats[s] for s in POOL_STATS], axis=-1).astype(np.float32)


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
    signals_z: [M, 245] float32, ALREADY z-scored (the distribution the
    backbone was trained on -- features.normalize_like_fairseq).

    Returns {stage: [M, 10, D]} float16 for every stage in BANK_STAGES.
    fp16 halves a ~3 GB cache and is far below the noise floor of an R2
    measured to ~0.001; the arrays are cast back to float32 on assembly.

    One forward pass per batch with output_hidden_states=True yields all 13
    transformer taps; the two FE stages come from the same pass's
    sub-modules, matching taps.extract_stage_reps exactly (fe is
    channel-major pre-LayerNorm; extract_features is post-LayerNorm).
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

            fe_out = model.feature_extractor(batch)          # [B, 512, T]
            fe_t = fe_out.transpose(1, 2)                    # [B, T, 512]
            out["fe"].append(_bank_stats_from_seq(fe_t).cpu().numpy())

            ef = model.feature_projection.layer_norm(fe_t)   # post-LN
            out["extract_features"].append(_bank_stats_from_seq(ef).cpu().numpy())

            hs = model(input_values=batch, output_hidden_states=True).hidden_states
            if len(hs) != N_TRANSFORMER_LAYERS:
                raise RuntimeError(
                    f"expected {N_TRANSFORMER_LAYERS} hidden states, got {len(hs)}")
            for li, h in enumerate(hs):
                out[f"layer{li}"].append(_bank_stats_from_seq(h).cpu().numpy())

    return {s: np.concatenate(v, axis=0).astype(np.float16)
            for s, v in out.items()}


def build_bank_cache(checkpoint_path: str, labeled_data_dir: str, out_dir: str,
                     comps: tuple = (0, 1, 2), max_samples: int = 5000,
                     device: str = "cuda", batch_size: int = 64,
                     seed: int = 42) -> str:
    """
    One GPU pass -> <out_dir>/bank.npz. Idempotent: returns immediately if the
    file already exists (it is ~3 GB and takes minutes to rebuild).
    """
    import os
    import time

    from ..checkpoint_loader import CheckpointLoader
    from ..data_loader import load_labeled_data
    from . import features as feat

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bank.npz")
    if os.path.exists(path):
        print(f"[step6] bank cache hit: {path}", flush=True)
        return path

    raw, y = load_labeled_data(labeled_data_dir, max_samples=max_samples,
                               seed=seed, comps=comps)
    n, k, L = raw.shape
    print(f"[step6] loaded {raw.shape}, extracting bank on {device}", flush=True)

    flat_raw = raw.reshape(n * k, L).astype(np.float32)
    flat_z = feat.normalize_like_fairseq(flat_raw)

    model = CheckpointLoader.from_file(checkpoint_path)
    t0 = time.time()
    bank = extract_bank(model, flat_z, device=device, batch_size=batch_size)
    print(f"[step6] extracted in {time.time() - t0:.1f}s", flush=True)

    payload = {f"bank__{s}": arr.reshape(n, k, len(POOL_STATS), arr.shape[-1])
               for s, arr in bank.items()}
    payload["input_raw"] = flat_raw.reshape(n, k, L)
    payload["input_z"] = flat_z.reshape(n, k, L)
    payload["y"] = y
    payload["_meta"] = np.array([repr({
        "checkpoint": checkpoint_path, "comps": comps, "n": int(n),
        "max_samples": max_samples, "seed": seed,
        "pool_stats": POOL_STATS, "stages": BANK_STAGES})])
    # Write to a temp file and atomically rename into place. A killed run
    # (OOM, preemption, Ctrl-C -- all plausible for a multi-GB, minutes-long
    # GPU job on a shared box) must never leave a truncated bank.npz behind:
    # the idempotency check above is a plain os.path.exists, so a corrupt
    # partial file would be silently treated as a cache hit by the next run.
    # np.savez appends ".npz" to a bare string path that lacks that suffix,
    # which would silently turn "bank.npz.tmp" into "bank.npz.tmp.npz" --
    # pass an open file handle instead, which numpy writes to as-is.
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            np.savez(f, **payload)   # uncompressed: fp16 barely compresses and
                                      # savez_compressed on 3 GB takes minutes
        os.replace(tmp_path, path)   # atomic within a filesystem
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f"[step6] wrote {path} "
          f"({os.path.getsize(path) / 1e9:.2f} GB)", flush=True)
    return path


def load_bank_cache(path: str):
    """Returns (bank {stage: [N,K,10,D]}, input_raw, input_z, y, meta)."""
    data = np.load(path, allow_pickle=False)
    bank = {k[len("bank__"):]: data[k] for k in data.files if k.startswith("bank__")}
    meta = ast.literal_eval(str(data["_meta"][0]))
    return bank, data["input_raw"], data["input_z"], data["y"], meta
