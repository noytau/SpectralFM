"""
.npz cache of extracted stage representations, keyed by checkpoint + n +
comps + pooling, so steps 2-4 (regressor swap, DS features, feature-vector
sweeps) are CPU-only re-runs of step 1's expensive GPU extraction.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np


def cache_key(checkpoint_path: str, n_samples: int, comps: tuple, pooling: str,
              seed: int) -> str:
    raw = f"{os.path.basename(checkpoint_path)}|{n_samples}|{comps}|{pooling}|{seed}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"reps_{key}.npz")


def save_reps(cache_dir: str, key: str, reps_by_comp: dict, y: np.ndarray,
              meta: dict) -> str:
    """reps_by_comp: {stage: [N, K, D]} (K = n_comps loaded). Saved as one
    npz with stage-prefixed keys plus labels and a meta blob."""
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, key)
    payload = {f"stage__{s}": arr for s, arr in reps_by_comp.items()}
    payload["y"] = y
    payload["_meta"] = np.array([repr(meta)])
    np.savez_compressed(path, **payload)
    return path


def load_reps(cache_dir: str, key: str):
    path = cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    reps_by_comp = {k[len("stage__"):]: data[k] for k in data.files
                    if k.startswith("stage__")}
    y = data["y"]
    meta = eval(str(data["_meta"][0]))  # noqa: S307 -- our own repr(dict), trusted
    return reps_by_comp, y, meta
