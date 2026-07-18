"""
Data loading for SpectralFM evaluation.
Supports: directory of .wav files, single CSV/parquet file.
No fairseq dependency.
"""
import os
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader


# ── Raw data loading ──────────────────────────────────────────────────────────

def load_wav_dir(directory: str, sample_rate: int = 16000, stack_size: int = 10) -> pd.DataFrame:
    """
    Load all .wav files in a directory.
    SpectralFM wavs are 1D spectrograms (245 samples at 16kHz) — load raw samples directly.
    Returns a DataFrame with 'data' (list of floats) and 'stack_idx' columns.
    Stack = consecutive group of `stack_size` files from the same observation.
    """
    all_rows = []

    wav_files = sorted(f for f in os.listdir(directory) if f.endswith(".wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in: {directory}")

    for i, fname in enumerate(wav_files):
        path = os.path.join(directory, fname)
        try:
            waveform, sr = torchaudio.load(path)
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            signal = waveform.squeeze(0).tolist()  # [245]
            all_rows.append({
                "data": signal,
                "stack_idx": i // stack_size,
                "filename": fname,
            })
        except Exception as e:
            print(f"[DataLoader] Skipping {fname}: {e}")

    if not all_rows:
        raise RuntimeError("No valid .wav files could be loaded.")

    result = pd.DataFrame(all_rows)
    print(f"[DataLoader] Loaded {len(wav_files)} files → {len(result)} rows from {directory}")
    return result


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV or parquet file. Expects numeric columns + optional 'stack_idx'."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "stack_idx" not in df.columns:
        df["stack_idx"] = df.index // 10
    print(f"[DataLoader] Loaded CSV: {path} → {df.shape}")
    return df


def load_data(source: str, sample_rate: int = 16000, stack_size: int = 10) -> pd.DataFrame:
    """Auto-detect source type and load data."""
    if os.path.isdir(source):
        return load_wav_dir(source, sample_rate=sample_rate, stack_size=stack_size)
    elif os.path.isfile(source):
        return load_csv(source)
    raise FileNotFoundError(f"Source not found: {source}")


_MANIFEST_REMAPS = [("/storage/noy/", "/mnt5/noy/"), ("/storage/", "/mnt5/")]


def load_manifest_subset(
    dataset_dir: str,
    split: str = "valid",
    n: int = 500,
    seed: int = 42,
    stack_size: int = 10,
) -> pd.DataFrame:
    """
    Load a deterministic n-sample subset from a fairseq manifest (train.tsv/valid.tsv).

    Samples WHOLE stacks (contiguous blocks of `stack_size` manifest rows) so the
    stack structure needed by the stack-query/clustering evals survives the draw:
    n // stack_size blocks are chosen with a seeded RNG, each contributing its
    `stack_size` consecutive rows. TSV roots written for RunAI (/storage/...) are
    remapped to the Geoffrey mount (/mnt5/...).

    Returns a DataFrame with 'data', 'stack_idx', 'filename' columns.
    """
    tsv = os.path.join(dataset_dir, f"{split}.tsv")
    if not os.path.isfile(tsv):
        raise FileNotFoundError(tsv)
    with open(tsv) as f:
        root = f.readline().strip()
        rows = [ln.strip().split("\t")[0] for ln in f if ln.strip()]
    for src, dst in _MANIFEST_REMAPS:
        if root.startswith(src):
            root = dst + root[len(src):]
            break

    n_blocks_total = len(rows) // stack_size
    n_blocks = max(1, min(n // stack_size, n_blocks_total))
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(n_blocks_total, n_blocks, replace=False))

    all_rows = []
    for stack_idx, b in enumerate(chosen):
        for fname in rows[b * stack_size : (b + 1) * stack_size]:
            path = os.path.join(root, fname)
            try:
                waveform, _ = torchaudio.load(path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                all_rows.append({
                    "data": waveform.squeeze(0).tolist(),
                    "stack_idx": stack_idx,
                    "filename": fname,
                })
            except Exception as e:
                print(f"[DataLoader] Skipping {fname}: {e}")

    if not all_rows:
        raise RuntimeError(f"No wavs loaded from {tsv}")
    df = pd.DataFrame(all_rows)
    print(f"[DataLoader] {os.path.basename(dataset_dir)}/{split}: "
          f"{len(df)} samples ({n_blocks} stacks, seed={seed})")
    return df


_LABELED_PATTERN = re.compile(r"dataset(\d+)_comp(\d+)_spec_(\d+)\.wav")


def load_labeled_data(
    labeled_data_dir: str,
    max_samples: int = 2000,
    target_length: int = 245,
    seed: int = 42,
    comps: tuple = (0,),
) -> tuple:
    """
    Load labeled spectrograms for the parameter_0 regression probe.
    Ported from eval_label_regression.py::load_labeled_spectrograms, with
    multi-channel support from label_reg_evaluation.py::_build_merged.

    Expects `labels.tsv` (filename \\t parameter_0) in labeled_data_dir and wavs
    under `wav/`, `wavs/`, or the directory itself. Wavs follow
    `dataset<D>_comp<C>_spec_<S>.wav`; components of the same (dataset, spec)
    share one label.

    comps: which components to stack per spectrum, e.g. (0,), (0, 1), (0, 1, 2).
    Only spectra that have ALL requested components are kept.

    Returns (inputs [N, len(comps), target_length] float32, labels [N] float64).
    """
    import glob

    labels_path = os.path.join(labeled_data_dir, "labels.tsv")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(labels_path)

    # (dataset, spec) → {comp: filename}, label
    spec_files: dict = {}
    spec_label: dict = {}
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            m = _LABELED_PATTERN.match(parts[0])
            if m is None:
                continue
            ds, comp, spec = int(m.group(1)), int(m.group(2)), int(m.group(3))
            spec_files.setdefault((ds, spec), {})[comp] = parts[0]
            spec_label[(ds, spec)] = float(parts[1])

    keys = sorted(k for k, files in spec_files.items()
                  if all(c in files for c in comps))
    if not keys:
        raise RuntimeError(f"No spectra with components {comps} under {labeled_data_dir}")
    rng = np.random.default_rng(seed)
    if len(keys) > max_samples:
        idx = rng.choice(len(keys), max_samples, replace=False)
        idx.sort()
        keys = [keys[i] for i in idx]

    wav_root = labeled_data_dir
    for sub in ("wav", "wavs"):
        cand = os.path.join(labeled_data_dir, sub)
        if glob.glob(os.path.join(cand, "*.wav")):
            wav_root = cand
            break

    try:
        import soundfile as sf
        _read = lambda fp: sf.read(fp, dtype="float32")[0]
    except ImportError:
        _read = lambda fp: torchaudio.load(fp)[0].mean(0).numpy()

    def _load_one(fname):
        fp = os.path.join(wav_root, fname)
        if not os.path.isfile(fp):
            fp = os.path.join(labeled_data_dir, fname)
        if not os.path.isfile(fp):
            return None
        data = np.asarray(_read(fp)).flatten()
        row = np.zeros(target_length, dtype=np.float32)
        row[: min(len(data), target_length)] = data[:target_length]
        return row

    inputs, ys = [], []
    for key in keys:
        rows = [_load_one(spec_files[key][c]) for c in comps]
        if any(r is None for r in rows):
            continue
        inputs.append(np.stack(rows, axis=0))          # [n_comps, L]
        ys.append(spec_label[key])
    if not inputs:
        raise RuntimeError(f"No labeled wavs loaded under {labeled_data_dir}")
    print(f"[DataLoader] Loaded {len(inputs)} labeled spectra "
          f"(comps={comps}) from {labeled_data_dir}")
    return np.stack(inputs, axis=0), np.array(ys, dtype=np.float64)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def normalize_to_audio_range(series: pd.Series) -> pd.Series:
    """Normalize values to [-1, 1]."""
    return series.apply(lambda row: [2 * float(v) - 1 for v in row])


def apply_masking(original: torch.Tensor, mask_ratio: float = 0.15, masking_type: str = "random") -> torch.Tensor:
    """
    Apply masking to a 1D tensor.
    Supported types: random, grid, span_start, span_end, span, low_energy, high_energy.
    """
    masked = original.clone()

    if masking_type in ("random", None):
        indices = torch.randperm(masked.shape[0])[: int(mask_ratio * masked.shape[0])]
        masked[indices] = 0.0

    elif masking_type == "grid":
        step = int(1 / mask_ratio)
        masked[::step] = 0.0

    elif masking_type == "span_start":
        n = int(mask_ratio * len(masked))
        masked[:n] = 0.0

    elif masking_type == "span_end":
        n = int(mask_ratio * len(masked))
        masked[-n:] = 0.0

    elif masking_type == "span":
        total = int(mask_ratio * len(masked))
        used: set = set()
        while len(used) < total:
            span = random.randint(10, max(10, total))
            span = min(span, total - len(used))
            start = random.randint(0, len(masked) - span)
            if any(i in used for i in range(start, start + span)):
                continue
            for i in range(start, start + span):
                masked[i] = 0.0
                used.add(i)

    elif masking_type == "low_energy":
        threshold = torch.quantile(original.abs(), mask_ratio)
        masked[original.abs() < threshold] = 0.0

    elif masking_type == "high_energy":
        threshold = torch.quantile(original.abs(), mask_ratio)
        masked[original.abs() > threshold] = 0.0

    else:
        raise ValueError(f"Unknown masking_type: {masking_type!r}")

    return masked


def _collate(batch):
    return {
        "data": torch.stack([b["data"] for b in batch]),
        "masked_data": torch.stack([b["masked_data"] for b in batch]),
    }


def build_dataloader(
    df: pd.DataFrame,
    mask_ratio: float = 0.15,
    masking_type: str = "random",
    batch_size: int = 16,
    data_col: str = "data",
    normalize: bool = True,
) -> tuple:
    """
    Build a DataLoader from a DataFrame.
    Each row in data_col must be a list/array of floats.
    normalize=True applies per-sample layer_norm to match fairseq training (normalize: true).
    Returns (dataloader, df_with_masked_data_column).
    """
    df = df.copy()
    records = []
    masked_col = []
    norm_col = []

    for i, row in df.iterrows():
        raw = row[data_col] if not isinstance(row[data_col], list) else row[data_col]
        original = torch.tensor(raw, dtype=torch.float32)
        if normalize:
            original = F.layer_norm(original, original.shape)
        masked = apply_masking(original, mask_ratio=mask_ratio, masking_type=masking_type)
        masked_col.append(masked.tolist())
        norm_col.append(original.tolist())
        records.append({"data": original, "masked_data": masked})

    df["masked_data"] = masked_col
    df[data_col] = norm_col  # Update df to hold normalized values too
    loader = DataLoader(records, batch_size=batch_size, collate_fn=_collate)
    return loader, df


# ── Test split helpers ────────────────────────────────────────────────────────

def split_stack_holdout(df: pd.DataFrame, n_holdout: int = 5) -> pd.DataFrame:
    """Hold out the first n_holdout stacks for evaluation."""
    stacks = df["stack_idx"].unique()
    heldout = stacks[:n_holdout]
    return df[df["stack_idx"].isin(heldout)].reset_index(drop=True)


def split_partial_stack(df: pd.DataFrame, holdout_ratio: float = 0.3) -> pd.DataFrame:
    """From each stack, hold out a fraction of samples."""
    parts = []
    for _, group in df.groupby("stack_idx"):
        n = max(1, int(len(group) * holdout_ratio))
        parts.append(group.sample(n=n))
    return pd.concat(parts).reset_index(drop=True)
