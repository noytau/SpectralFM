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

# ── Component metadata (single- vs multi-component datasets) ───────────────────

_COMP_PATTERN = re.compile(r"dataset(\d+)_comp(\d+)_spec_(\d+)\.wav")

# Verified duplicates: in both multi_channel and labeled_data, comp20 is byte-identical
# to comp14 and comp21 to comp15 (np.allclose on the raw wavs; per-component std matches
# exactly at 0.0779 and 0.0546). sampled_data has no identical pairs. Left in place, they
# double-count ~2/12 (multi_channel) and ~2/14 (labeled_data) of the mass and inflate
# every components-per-spectrum count by 2.
_DUPLICATE_COMPS = {
    "multi_channel": {20: 14, 21: 15},
    "labeled_data":  {20: 14, 21: 15},
}

_COMP_META_COLUMNS = ["filename", "dataset_id", "comp", "spec",
                      "n_comps", "n_comps_in_split", "component_group"]


def duplicate_comps_for(dataset_dir: str) -> dict:
    """Redundant-component map for a dataset dir, keyed by its basename ({} if none)."""
    return _DUPLICATE_COMPS.get(os.path.basename(os.path.normpath(dataset_dir)), {})


def _scan_manifest_components(tsv: str, drop: set) -> list:
    """Yield (filename, dataset_id, comp, spec) for every comp-named row of a manifest."""
    rows = []
    with open(tsv) as f:
        f.readline()                                     # root path line
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname = line.split("\t")[0]
            m = _COMP_PATTERN.match(os.path.basename(fname))
            if m is None:
                continue
            comp = int(m.group(2))
            if comp in drop:
                continue
            rows.append((fname, int(m.group(1)), comp, int(m.group(3))))
    return rows


def parse_component_metadata(
    dataset_dir: str,
    split: str = "valid",
    drop_duplicate_comps: bool = True,
    count_splits: tuple = ("train", "valid"),
) -> pd.DataFrame:
    """
    Scan a manifest and describe each file's place in its spectrum.

    Every wav in nova_data is a single 245-bin component. In the multi-component
    subsets (multi_channel, sampled_data, labeled_data) the physical sample is the SET
    of components sharing a spec index, named `dataset<D>_comp<C>_spec_<S>.wav`. The
    single-channel subsets have no `comp` token at all — that absence IS the
    single-vs-multi signal, and this function returns an empty frame for them.

    Two component counts, because they answer different questions and disagree sharply:
      n_comps           components of this spectrum across `count_splits` (the whole
                        dataset). This is the physical component count.
      n_comps_in_split  components of this spectrum present in `split` alone.
    They diverge badly on multi_channel, whose valid split holds a scattered sample:
    93k of its 96k spectra have a single component in valid.tsv while the dataset
    actually provides 8–12 per spectrum. Use n_comps for "how many components does
    this sample have"; n_comps_in_split only describes the draw.

    `n_comps` cannot come from a loaded subset: `load_manifest_subset` draws contiguous
    10-row blocks and the multi-component manifests are shuffled, so a subset never
    holds a whole spectrum. Scanning multi_channel's full train+valid (3.4M rows) costs
    ~7 s; pass count_splits=(split,) to skip it when only the draw matters.

    Returns a DataFrame with the columns in _COMP_META_COLUMNS; `component_group` is
    'multi' for every row of a multi-component manifest.
    """
    tsv = os.path.join(dataset_dir, f"{split}.tsv")
    if not os.path.isfile(tsv):
        raise FileNotFoundError(tsv)

    name = os.path.basename(os.path.normpath(dataset_dir))
    dupes = duplicate_comps_for(dataset_dir) if drop_duplicate_comps else {}
    drop = set(dupes)

    rows = _scan_manifest_components(tsv, drop)
    if not rows:
        print(f"[DataLoader] {name}/{split}: single-component "
              f"(no comp field in filenames)")
        return pd.DataFrame(columns=_COMP_META_COLUMNS)

    df = pd.DataFrame(rows, columns=["filename", "dataset_id", "comp", "spec"])
    df["n_comps_in_split"] = df.groupby(["dataset_id", "spec"])["comp"].transform("size")

    # Dataset-wide component count, from every split that exists on disk.
    universe = []
    for s in count_splits:
        path = os.path.join(dataset_dir, f"{s}.tsv")
        if os.path.isfile(path):
            universe.extend(_scan_manifest_components(path, drop))
    if universe:
        u = pd.DataFrame(universe, columns=["filename", "dataset_id", "comp", "spec"])
        totals = (u.drop_duplicates(["dataset_id", "spec", "comp"])
                   .groupby(["dataset_id", "spec"]).size().rename("n_comps"))
        df = df.merge(totals, on=["dataset_id", "spec"], how="left")
        df["n_comps"] = df["n_comps"].fillna(df["n_comps_in_split"]).astype(int)
    else:
        df["n_comps"] = df["n_comps_in_split"]

    df["component_group"] = "multi"
    df = df[_COMP_META_COLUMNS]

    msg = (f"[DataLoader] {name}/{split}: multi-component — {len(df)} rows, "
           f"{df.groupby(['dataset_id', 'spec']).ngroups} spectra, "
           f"comps {sorted(int(c) for c in df['comp'].unique())}, "
           f"n_comps per spectrum {sorted(int(v) for v in df['n_comps'].unique())}")
    if dupes:
        pairs = ", ".join(f"comp{d}≡comp{k}" for d, k in dupes.items())
        msg += f"; dropped redundant {pairs}"
    print(msg)
    return df
