"""
Structured Similarity Evaluation
----------------------------------
Evaluates the model on the canonical 100-sample panel:
    0– 29  single_channel_all  (3 stacks  × 10)
   30– 59  multi_channel       (3 comps   × 10)
   60– 79  sampled_data        (2 comps   × 10)
   80– 99  labeled_data        (2 comps   × 10)

Panel is seeded (seed=42) and deterministic — same as the epoch-cosim training callback.
"""
from __future__ import annotations
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ── Path helpers ───────────────────────────────────────────────────────────────

_PATH_REMAPS = [
    ("/storage/noy/", "/mnt5/noy/"), ("/storage/", "/mnt5/"),
    ("/mnt5/noy/", "/storage/noy/"), ("/mnt5/", "/storage/"),
]

def _remap_root(root: str) -> str:
    # Try both directions (/storage <-> /mnt5) when the recorded root doesn't
    # resolve locally, keeping whichever candidate actually does. Needed
    # because some manifests (e.g. the Geoffrey-authored special reference
    # datasets under nova_data/) still carry a /mnt5/... root even after
    # their wav files are copied to a /storage/...-native host like gpu55/56.
    if os.path.isdir(root):
        return root
    for src, dst in _PATH_REMAPS:
        if root.startswith(src):
            candidate = dst + root[len(src):]
            if os.path.isdir(candidate):
                return candidate
    return root


# ── TSV / WAV loading ─────────────────────────────────────────────────────────

def _read_tsv(dataset_dir: str, prefer_manifest: Optional[str] = None) -> List[Tuple[int, str]]:
    order = ("train.tsv", "valid.tsv") if prefer_manifest == "train" else ("valid.tsv", "train.tsv")
    for name in order:
        p = os.path.join(dataset_dir, name)
        if os.path.exists(p):
            rows = []
            with open(p) as f:
                lines = f.readlines()
            for i, line in enumerate(lines[1:]):
                line = line.strip()
                if line:
                    rows.append((i, line.split("\t")[0]))
            return rows
    raise FileNotFoundError(f"No valid.tsv or train.tsv in {dataset_dir}")


def _load_wav(dataset_dir: str, index: int, prefer_manifest: Optional[str] = None) -> Optional[np.ndarray]:
    order = ("train.tsv", "valid.tsv") if prefer_manifest == "train" else ("valid.tsv", "train.tsv")
    tsv_path = None
    for name in order:
        p = os.path.join(dataset_dir, name)
        if os.path.exists(p):
            tsv_path = p
            break
    if tsv_path is None:
        return None
    with open(tsv_path) as f:
        lines = f.readlines()
    if len(lines) < 2 or index < 0 or index > len(lines) - 2:
        return None
    root = _remap_root(lines[0].strip())
    rel  = lines[index + 1].strip().split("\t")[0]
    wav_path = os.path.join(root, rel)
    try:
        import soundfile as sf
        data, _ = sf.read(wav_path, dtype="float32")
        return data.flatten()
    except Exception:
        pass
    try:
        import torchaudio
        w, _ = torchaudio.load(wav_path)
        return w.mean(0).numpy()
    except Exception:
        return None


def _normalize_like_fairseq(arr: np.ndarray) -> np.ndarray:
    mean = arr.mean(axis=1, keepdims=True)
    std  = arr.std(axis=1,  keepdims=True) + 1e-8
    return ((arr - mean) / std).astype(np.float32)


# ── Component grouping ─────────────────────────────────────────────────────────

_COMP_RE  = re.compile(r"_comp(\d+)_")
_COMP_RE2 = re.compile(r"comp(\d+)")

def _comp_id(fname: str) -> Optional[int]:
    m = _COMP_RE.search(fname) or _COMP_RE2.search(fname)
    return int(m.group(1)) if m else None

def _group_by_comp(rows: List[Tuple[int, str]]) -> Dict[int, List[Tuple[int, str]]]:
    groups: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for idx, fname in rows:
        c = _comp_id(fname)
        if c is not None:
            groups[c].append((idx, fname))
    return dict(groups)


# ── Panel builder ──────────────────────────────────────────────────────────────

def build_panel(
    nova_data_dir: str,
    *,
    seed: int = 42,
    prefer_manifest: Optional[str] = None,
    target_length: int = 245,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the canonical 100-sample structured similarity panel.
    Composition (matches eval_utils.build_structured_similarity_subset):
        0– 29  single_channel_all  3 stacks  × 10
       30– 59  multi_channel       3 comps   × 10
       60– 79  sampled_data        2 comps   × 10
       80– 99  labeled_data        2 comps   × 10

    Returns:
        inputs  — np.ndarray [100, target_length], per-sample layer-norm normalised
        groups  — list[str] of length 100 with dataset label per sample
    """
    rng = np.random.default_rng(seed)
    entries: List[Tuple[str, str, int]] = []  # (dataset_name, dataset_path, tsv_index)
    group_labels: List[str] = []

    # ── single_channel_all: 3 stacks × 10 ────────────────────────────────────
    sc_path = os.path.join(nova_data_dir, "single_channel_all")
    sc_rows = _read_tsv(sc_path, prefer_manifest)
    total_stacks = len(sc_rows) // 10
    chosen_stacks = sorted(rng.choice(total_stacks, 3, replace=False).tolist())
    for s in chosen_stacks:
        for i in range(s * 10, s * 10 + 10):
            entries.append(("single_channel_all", sc_path, sc_rows[i][0]))
            group_labels.append("single_channel_all")

    # ── component-based datasets ───────────────────────────────────────────────
    for ds_name, n_groups in [("multi_channel", 3), ("sampled_data", 2), ("labeled_data", 2)]:
        ds_path  = os.path.join(nova_data_dir, ds_name)
        ds_rows  = _read_tsv(ds_path, prefer_manifest)
        comp_map = _group_by_comp(ds_rows)
        eligible = sorted(c for c, r in comp_map.items() if len(r) >= 10)
        if len(eligible) < n_groups:
            raise ValueError(f"{ds_name}: only {len(eligible)} components with ≥10 samples; need {n_groups}.")
        chosen_comps = sorted(rng.choice(eligible, n_groups, replace=False).tolist())
        for comp in chosen_comps:
            comp_rows = comp_map[comp]
            positions = sorted(rng.choice(len(comp_rows), 10, replace=False).tolist())
            for pos in positions:
                idx, _ = comp_rows[pos]
                entries.append((ds_name, ds_path, idx))
                group_labels.append(ds_name)

    # ── Load wavs ─────────────────────────────────────────────────────────────
    inputs: List[np.ndarray] = []
    for ds_name, ds_path, idx in entries:
        data = _load_wav(ds_path, idx, prefer_manifest)
        if data is None:
            raise RuntimeError(f"Failed to load index {idx} from {ds_name} ({ds_path})")
        if len(data) >= target_length:
            row = data[:target_length].astype(np.float32)
        else:
            row = np.zeros(target_length, dtype=np.float32)
            row[:len(data)] = data
        inputs.append(row)

    arr = np.stack(inputs)
    return _normalize_like_fairseq(arr), group_labels


# ── Representation extraction ──────────────────────────────────────────────────

def _extract_all_representations(
    model, inputs: np.ndarray, device: str, batch_size: int = 32
) -> dict:
    """
    Extract all 4 representations in a single forward pass per batch:

      fe:         raw CNN output, mean-pooled over time  → [N, 512]
                  model.feature_extractor(x)
      projection: conv → layer_norm → linear 512→768, mean-pooled → [N, 768]
                  (this is what the transformer receives as input)
      embedding:  full transformer output, mean-pooled   → [N, 768]
                  model.last_hidden_state

    Returns dict with keys fe, projection, embedding (all np.ndarray or None on error).
    """
    model.eval()
    model.to(device)
    fe_list, proj_list, emb_list = [], [], []
    t = torch.from_numpy(inputs)
    try:
        with torch.no_grad():
            for i in range(0, len(t), batch_size):
                batch = t[i:i + batch_size].to(device)              # [B, 245]

                # 1. FE: raw conv output
                x_fe = model.feature_extractor(batch)               # [B, 512, T']
                fe_list.append(x_fe.mean(dim=-1).cpu().numpy())     # [B, 512]

                # 2. Projection: conv → layer_norm → linear (512→768)
                x = x_fe.transpose(1, 2)                            # [B, T', 512]
                x = model.feature_projection.layer_norm(x)
                x = model.feature_projection.projection(x)          # [B, T', 768]
                proj_list.append(x.mean(dim=1).cpu().numpy())       # [B, 768]

                # 3. Embedding: full transformer
                out = model(input_values=batch)
                emb_list.append(
                    out.last_hidden_state.mean(dim=1).cpu().numpy() # [B, 768]
                )
    except Exception as e:
        print(f"[StructuredSim] Extraction failed: {e}")
        return {"fe": None, "projection": None, "embedding": None}

    return {
        "fe":         np.concatenate(fe_list,   axis=0),
        "projection": np.concatenate(proj_list, axis=0),
        "embedding":  np.concatenate(emb_list,  axis=0),
    }


# ── Main run ───────────────────────────────────────────────────────────────────

def run(
    model,
    nova_data_dir: str,
    device: str = "cpu",
    batch_size: int = 32,
    seed: int = 42,
    prefer_manifest: Optional[str] = None,
) -> dict:
    """
    Returns dict with representations and cosine similarity matrices for all 4 stages:
      inputs         — [100, 245]   raw z-score-normalised input signals
      fe_outputs     — [100, 512]   raw CNN output (before layer_norm/projection)
      proj_outputs   — [100, 768]   after layer_norm + linear projection (transformer input)
      embeddings     — [100, 768]   transformer output
      sim_matrix_*   — [100, 100]   cosine similarity matrices (vmin=-1, vmax=1)
      groups         — list[str]    dataset label per sample
    """
    from sklearn.metrics.pairwise import cosine_similarity as _cosim

    inputs, groups = build_panel(nova_data_dir, seed=seed, prefer_manifest=prefer_manifest)
    reps = _extract_all_representations(model, inputs, device=device, batch_size=batch_size)

    def _sim(m):
        return _cosim(m).astype(np.float32) if m is not None else None

    def _sim_centered(m):
        """Cosine after removing the mean vector (anisotropy correction).
        Trained models share one dominant common direction (positive GELU
        activations, mean-pooled) that pushes all raw cosines to ~0.9+;
        centering reveals the per-sample structure underneath."""
        if m is None:
            return None
        return _cosim(m - m.mean(axis=0, keepdims=True)).astype(np.float32)

    return {
        "inputs":          inputs,
        "fe_outputs":      reps["fe"],
        "proj_outputs":    reps["projection"],
        "embeddings":      reps["embedding"],
        "groups":          groups,
        "sim_matrix_inp":  _sim(inputs),
        "sim_matrix_fe":   _sim(reps["fe"]),
        "sim_matrix_proj": _sim(reps["projection"]),
        "sim_matrix_emb":  _sim(reps["embedding"]),
        "sim_matrix_fe_centered":   _sim_centered(reps["fe"]),
        "sim_matrix_proj_centered": _sim_centered(reps["projection"]),
        "sim_matrix_emb_centered":  _sim_centered(reps["embedding"]),
    }
