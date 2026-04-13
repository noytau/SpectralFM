"""
Utility functions for SpectralFM evaluation.

Contains:
- DatasetSplitter: seen/unseen/ID/OOD/cross-component split logic
- Component ID extraction from WAV filenames
- Label loading helpers
- Attention extraction utilities
- build_structured_similarity_subset: offline builder for the structured eval subset

Separated from evaluation_runner.py for maintainability.
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict


# ------------------------------------------------------------------ #
#  Component ID Extraction                                           #
# ------------------------------------------------------------------ #

# Pattern: dataset0002_comp3_spec_12345.wav -> comp=3
_COMP_PATTERN = re.compile(r"_comp(\d+)_")
# Alternative: just compN anywhere
_COMP_PATTERN_ALT = re.compile(r"comp(\d+)")
# Pattern: dataset0002_comp3_spec_12345.wav -> spec_index=12345
_SPEC_PATTERN = re.compile(r"_spec_(\d+)")


def extract_component_id(filename: str) -> Optional[int]:
    """
    Extract component ID from a WAV filename.

    Convention: dataset0002_comp3_spec_12345.wav -> 3

    Args:
        filename: WAV filename (with or without path)

    Returns:
        Integer component ID, or None if not found
    """
    basename = os.path.basename(filename)
    match = _COMP_PATTERN.search(basename)
    if match:
        return int(match.group(1))
    match = _COMP_PATTERN_ALT.search(basename)
    if match:
        return int(match.group(1))
    return None


def extract_component_ids_from_dataset(
    dataset,
    selected_indices: Optional[List[int]] = None,
    n_samples: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract component IDs for all samples in a fairseq dataset.

    Args:
        dataset: fairseq FileAudioDataset
        selected_indices: specific indices to use
        n_samples: number of samples (if selected_indices not given)

    Returns:
        (component_ids: [N,] int array, filenames: list of strings)
        component_ids[i] = -1 if extraction failed
    """
    ids = []
    filenames = []

    if selected_indices is not None:
        indices = selected_indices
    elif n_samples is not None:
        indices = list(range(min(n_samples, len(dataset))))
    else:
        indices = list(range(len(dataset)))

    for idx in indices:
        try:
            # FileAudioDataset stores filenames in .fnames
            if hasattr(dataset, 'fnames'):
                fn = dataset.fnames[idx]
            elif hasattr(dataset, 'fname_list'):
                fn = dataset.fname_list[idx]
            else:
                fn = str(idx)
        except (IndexError, AttributeError):
            fn = str(idx)

        filenames.append(fn)
        comp_id = extract_component_id(fn)
        ids.append(comp_id if comp_id is not None else -1)

    return np.array(ids), filenames


def extract_component_ids_from_directory(
    data_dir: str,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract component IDs from WAV filenames in a directory.

    Args:
        data_dir: directory containing WAV files
        max_files: maximum number of files to scan

    Returns:
        (component_ids, filenames)
    """
    import glob
    wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    if max_files:
        wav_files = wav_files[:max_files]

    ids = []
    filenames = []
    for fp in wav_files:
        fn = os.path.basename(fp)
        filenames.append(fn)
        comp_id = extract_component_id(fn)
        ids.append(comp_id if comp_id is not None else -1)

    return np.array(ids), filenames


# ------------------------------------------------------------------ #
#  Dataset Splitter (Phase 0B)                                       #
# ------------------------------------------------------------------ #

class DatasetSplitter:
    """
    Split datasets by component overlap for evaluation.

    Modes:
        seen:              Evaluate on data from the SAME dataset used for training
        unseen:            Hold-out validation split from training dataset
        in_distribution:   Different dataset with overlapping component types
        out_of_distribution: Dataset with novel component types
        cross_component:   Within dataset, hold out specific component IDs

    Ported from code/testing.py into evaluation_runner.py ecosystem.
    """

    @staticmethod
    def split_seen_unseen(
        component_ids: np.ndarray,
        test_fraction: float = 0.2,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random train/test split (seen data = same distribution).

        Returns:
            (train_indices, test_indices)
        """
        rng = np.random.RandomState(seed)
        n = len(component_ids)
        indices = np.arange(n)
        rng.shuffle(indices)
        split = int(n * (1 - test_fraction))
        return indices[:split], indices[split:]

    @staticmethod
    def split_cross_component(
        component_ids: np.ndarray,
        holdout_components: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hold out specific component IDs for testing.

        Args:
            component_ids: [N,] component IDs
            holdout_components: list of component IDs to hold out

        Returns:
            (train_indices, test_indices)
        """
        holdout_set = set(holdout_components)
        train_idx = np.where(~np.isin(component_ids, list(holdout_set)))[0]
        test_idx = np.where(np.isin(component_ids, list(holdout_set)))[0]
        return train_idx, test_idx

    @staticmethod
    def get_overlapping_components(
        comp_ids_a: np.ndarray,
        comp_ids_b: np.ndarray,
    ) -> Tuple[set, set, set]:
        """
        Find shared and unique components between two datasets.

        Returns:
            (shared, only_in_a, only_in_b)
        """
        set_a = set(comp_ids_a[comp_ids_a >= 0])
        set_b = set(comp_ids_b[comp_ids_b >= 0])
        return set_a & set_b, set_a - set_b, set_b - set_a

    @staticmethod
    def split_by_overlap(
        component_ids: np.ndarray,
        reference_components: set,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split into in-distribution (overlapping) and OOD (non-overlapping).

        Args:
            component_ids: [N,] component IDs
            reference_components: set of component IDs from reference/training dataset

        Returns:
            (in_distribution_indices, ood_indices)
        """
        in_dist = np.where(np.isin(component_ids, list(reference_components)))[0]
        ood = np.where(~np.isin(component_ids, list(reference_components)))[0]
        return in_dist, ood


# ------------------------------------------------------------------ #
#  Label Loading                                                     #
# ------------------------------------------------------------------ #

def load_labels_tsv(dataset_dir: str) -> Dict[str, float]:
    """
    Load labels.tsv from a dataset directory.

    Format: filename<TAB>parameter_0_value (no header)

    Returns:
        Dict mapping filename -> parameter_0 value
    """
    labels_path = os.path.join(dataset_dir, "labels.tsv")
    if not os.path.exists(labels_path):
        return {}

    label_map = {}
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                try:
                    label_map[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return label_map


# ------------------------------------------------------------------ #
#  Attention Extraction (Phase 6D)                                   #
# ------------------------------------------------------------------ #

def extract_attention_maps(
    model,
    sample_input,
    device: str = "cpu",
) -> List[np.ndarray]:
    """
    Extract attention weights from all transformer layers.

    Args:
        model: fairseq data2vec-audio model
        sample_input: [1, L] input tensor

    Returns:
        List of attention weight matrices [n_heads, L, L], one per layer
    """
    import torch

    model.eval()
    attention_maps = []

    # Register hooks on attention layers
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is typically (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    attention_maps.append(attn_weights.detach().cpu().numpy())
        return hook_fn

    # Try to find attention layers
    try:
        encoder = model.encoder if hasattr(model, 'encoder') else model
        if hasattr(encoder, 'layers'):
            for i, layer in enumerate(encoder.layers):
                if hasattr(layer, 'self_attn'):
                    h = layer.self_attn.register_forward_hook(make_hook(i))
                    hooks.append(h)
        elif hasattr(encoder, 'transformer'):
            if hasattr(encoder.transformer, 'layers'):
                for i, layer in enumerate(encoder.transformer.layers):
                    if hasattr(layer, 'self_attn'):
                        h = layer.self_attn.register_forward_hook(make_hook(i))
                        hooks.append(h)
    except Exception:
        pass

    # Forward pass
    tensor_input = torch.tensor(sample_input, dtype=torch.float32).to(device)
    if tensor_input.dim() == 1:
        tensor_input = tensor_input.unsqueeze(0)

    with torch.no_grad():
        try:
            model(tensor_input)
        except Exception:
            pass

    # Remove hooks
    for h in hooks:
        h.remove()

    return attention_maps


# ------------------------------------------------------------------ #
#  Light Mode Data Loading (Phase 0C)                                #
# ------------------------------------------------------------------ #

def load_wav_files_torchaudio(
    data_dir: str,
    max_samples: int = 100,
    target_length: int = 245,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load WAV files directly using torchaudio (no fairseq dependency).

    This is the "light mode" data loading path.

    Args:
        data_dir: directory containing WAV files
        max_samples: maximum number of files to load
        target_length: expected spectrogram length

    Returns:
        (inputs: [N, target_length], filenames: list)
    """
    import glob

    try:
        import torchaudio
    except ImportError:
        raise ImportError("torchaudio is required for light mode. Install with: pip install torchaudio")

    # Most nova_data datasets store WAVs in a `wav/` or `wavs/` subdirectory.
    # Fall back to those automatically when the root dir has no *.wav files.
    wav_root = data_dir
    if not glob.glob(os.path.join(data_dir, "*.wav")):
        for subdir in ("wav", "wavs"):
            candidate = os.path.join(data_dir, subdir)
            if glob.glob(os.path.join(candidate, "*.wav")):
                wav_root = candidate
                break

    wav_files = sorted(glob.glob(os.path.join(wav_root, "*.wav")))[:max_samples]
    if not wav_files:
        raise FileNotFoundError(
            f"No WAV files found in {data_dir} (also tried wav/ and wavs/ subdirs)"
        )

    inputs = []
    filenames = []

    for fp in wav_files:
        try:
            waveform, sr = torchaudio.load(fp)
            # Convert to numpy and flatten
            data = waveform.numpy().flatten()
            # Pad or truncate to target length
            if len(data) >= target_length:
                data = data[:target_length]
            else:
                padded = np.zeros(target_length)
                padded[:len(data)] = data
                data = padded
            inputs.append(data)
            filenames.append(os.path.basename(fp))
        except Exception:
            continue

    if not inputs:
        raise RuntimeError(f"Failed to load any WAV files from {data_dir}")

    return np.array(inputs, dtype=np.float32), filenames


# ------------------------------------------------------------------ #
#  Structured Similarity Subset (offline builder)                    #
# ------------------------------------------------------------------ #

def _read_tsv_indices(
    dataset_dir: str,
    prefer_manifest: Optional[str] = None,
) -> List[Tuple[int, str]]:
    """
    Read a fairseq-style TSV manifest and return (line_index, filename) pairs.

    Line index matches enumerate(lines[1:]) (data lines after root). For training
    alignment use ``prefer_manifest='train'`` so indices match ``train.tsv``.
    """
    if prefer_manifest == "train":
        order = ("train.tsv", "valid.tsv")
    elif prefer_manifest == "valid":
        order = ("valid.tsv", "train.tsv")
    else:
        order = ("valid.tsv", "train.tsv")

    tsv_path = None
    for tsv_name in order:
        p = os.path.join(dataset_dir, tsv_name)
        if os.path.exists(p):
            tsv_path = p
            break
    if tsv_path is None:
        raise FileNotFoundError(f"No valid.tsv or train.tsv found in {dataset_dir}")

    rows: List[Tuple[int, str]] = []
    with open(tsv_path, "r") as f:
        lines = f.readlines()
    for dataset_idx, line in enumerate(lines[1:]):
        line = line.strip()
        if not line:
            continue
        filename = line.split("\t")[0]
        rows.append((dataset_idx, filename))
    return rows


def fairseq_manifest_line_to_train_dataset_index(
    train_tsv_path: str,
    min_sample_size: int = 0,
) -> Dict[int, int]:
    """Map manifest data-line index → FileAudioDataset index (train.tsv)."""
    mapping: Dict[int, int] = {}
    ds_i = 0
    with open(train_tsv_path, "r") as f:
        f.readline()
        for line_i, line in enumerate(f):
            items = line.strip().split("\t")
            if len(items) != 2:
                continue
            sz = int(items[1])
            if min_sample_size is not None and sz < min_sample_size:
                continue
            mapping[line_i] = ds_i
            ds_i += 1
    return mapping


def _group_by_component(rows: List[Tuple[int, str]]) -> Dict[int, List[Tuple[int, str]]]:
    """
    Group (dataset_index, filename) pairs by component ID parsed from the filename.

    Returns {comp_id: [(dataset_index, filename), ...]}
    Components whose ID cannot be extracted are silently dropped.
    """
    groups: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for idx, fname in rows:
        comp_id = extract_component_id(fname)
        if comp_id is not None:
            groups[comp_id].append((idx, fname))
    return dict(groups)


def _group_by_spec_index(rows: List[Tuple[int, str]]) -> Dict[int, List[Tuple[int, int, str]]]:
    """
    Group (dataset_index, filename) pairs by spec_index parsed from the filename.

    Returns {spec_index: [(comp_id, dataset_index, filename), ...]}
    Rows whose spec_index or comp_id cannot be extracted are silently dropped.
    """
    groups: Dict[int, List[Tuple[int, int, str]]] = defaultdict(list)
    for idx, fname in rows:
        spec_m = _SPEC_PATTERN.search(fname)
        comp_id = extract_component_id(fname)
        if spec_m is not None and comp_id is not None:
            spec_idx = int(spec_m.group(1))
            groups[spec_idx].append((comp_id, idx, fname))
    return dict(groups)


def _load_wav_data_for_index(
    dataset_dir: str,
    index: int,
    prefer_manifest: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Load waveform for manifest line index (see :func:`_read_tsv_indices`)."""
    try:
        import soundfile as sf
    except ImportError:
        try:
            import scipy.io.wavfile as _wavfile
            sf = None
        except ImportError:
            return None

    if prefer_manifest == "train":
        order = ("train.tsv", "valid.tsv")
    elif prefer_manifest == "valid":
        order = ("valid.tsv", "train.tsv")
    else:
        order = ("valid.tsv", "train.tsv")

    tsv_path = None
    for tsv_name in order:
        p = os.path.join(dataset_dir, tsv_name)
        if os.path.exists(p):
            tsv_path = p
            break
    if tsv_path is None:
        return None

    with open(tsv_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return None

    root_dir = lines[0].strip()
    if index < 0 or index > len(lines) - 2:
        return None
    raw = lines[index + 1].strip()
    if not raw:
        return None
    rel_path = raw.split("\t")[0]
    wav_path = os.path.join(root_dir, rel_path)

    try:
        if sf is not None:
            data, _ = sf.read(wav_path, dtype="float32")
        else:
            import scipy.io.wavfile as _wavfile
            _, data = _wavfile.read(wav_path)
            data = data.astype(np.float32)
        return data.flatten()
    except Exception:
        return None


def build_structured_similarity_subset(
    nova_data_dir: str,
    seed: int = 42,
    *,
    prefer_manifest: Optional[str] = None,
) -> List[Dict]:
    """
    Build the structured similarity subset definition (run once offline).

    Reads only TSV manifests — does NOT load any audio.

    Composition (in evaluation order):
        3 stacks  × 10 samples  from single_channel_all  (stack = dataset_index // 10)
        3 comps   × 10 samples  from multi_channel
        2 comps   × 10 samples  from sampled_data
        2 comps   × 10 samples  from labeled_data

    Returns:
        Ordered list of entry dicts:
        {
            "dataset":    dataset directory name (str),
            "path":       absolute dataset path (str),
            "index":      0-based dataset index for load_eval_dataset_fairseq (int),
            "group":      human-readable group label, e.g. "stack_7" or "comp_3" (str),
            "group_type": "stack" | "component" (str),
        }
    """
    rng = np.random.default_rng(seed)
    entries: List[Dict] = []

    # ---- single_channel_all: 3 stacks × 10 --------------------------------
    sc_name = "single_channel_all"
    sc_path = os.path.join(nova_data_dir, sc_name)
    sc_rows = _read_tsv_indices(sc_path, prefer_manifest=prefer_manifest)
    total_stacks = len(sc_rows) // 10
    if total_stacks < 3:
        raise ValueError(
            f"single_channel_all has only {len(sc_rows)} samples "
            f"({total_stacks} full stacks); need at least 3."
        )
    chosen_stacks = sorted(rng.choice(total_stacks, 3, replace=False).tolist())
    for s in chosen_stacks:
        for i in range(s * 10, s * 10 + 10):
            idx, fname = sc_rows[i]
            entries.append({
                "dataset": sc_name,
                "path": sc_path,
                "index": idx,
                "group": f"stack_{s}",
                "group_type": "stack",
            })

    # ---- comp-based datasets -----------------------------------------------
    for ds_name, n_groups in [
        ("multi_channel", 3),
        ("sampled_data", 2),
        ("labeled_data", 2),
    ]:
        ds_path = os.path.join(nova_data_dir, ds_name)
        ds_rows = _read_tsv_indices(ds_path, prefer_manifest=prefer_manifest)
        comp_to_rows = _group_by_component(ds_rows)
        eligible = sorted(c for c, rows in comp_to_rows.items() if len(rows) >= 10)
        if len(eligible) < n_groups:
            raise ValueError(
                f"{ds_name}: only {len(eligible)} components have ≥10 samples; "
                f"need {n_groups}."
            )
        chosen = sorted(
            rng.choice(eligible, n_groups, replace=False).tolist()
        )
        for comp in chosen:
            rows = comp_to_rows[comp]
            picked_positions = sorted(
                rng.choice(len(rows), 10, replace=False).tolist()
            )
            for pos in picked_positions:
                idx, fname = rows[pos]
                entries.append({
                    "dataset": ds_name,
                    "path": ds_path,
                    "index": idx,
                    "group": f"comp_{comp}",
                    "group_type": "component",
                })

    return entries


def structured_subset_epoch_cosim_train_indices(
    nova_data_dir: str,
    task_data_dir: str,
    min_sample_size: int,
    seed: int = 42,
    prefer_manifest: str = "train",
) -> List[int]:
    """Structured subset filtered to ``task_data_dir``, remapped to train dataset indices."""
    entries = build_structured_similarity_subset(
        nova_data_dir, seed=seed, prefer_manifest=prefer_manifest
    )
    data_root = os.path.normpath(os.path.expanduser(task_data_dir))
    base = os.path.basename(data_root.rstrip(os.sep))
    train_tsv = os.path.join(data_root, "train.tsv")
    if not os.path.isfile(train_tsv):
        raise FileNotFoundError(f"task train manifest not found: {train_tsv}")

    line_map = fairseq_manifest_line_to_train_dataset_index(
        train_tsv, min_sample_size=min_sample_size
    )

    out: List[int] = []
    for e in entries:
        ep = os.path.normpath(os.path.expanduser(e.get("path", "")))
        ds = e.get("dataset")
        if ds != base and ep != data_root:
            continue
        line_idx = int(e["index"])
        if line_idx not in line_map:
            raise ValueError(
                f"Structured index {line_idx} ({e.get('group')}) not in FileAudioDataset(train)."
            )
        out.append(line_map[line_idx])
    return out


# ------------------------------------------------------------------ #
#  Failure Analysis (Phase 6F)                                       #
# ------------------------------------------------------------------ #

def analyze_failures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    component_ids: Optional[np.ndarray] = None,
    percentile: float = 5.0,
) -> Dict[str, Any]:
    """
    Analyze worst-performing samples.

    Args:
        y_true: true labels
        y_pred: predicted labels
        component_ids: optional component IDs for breakdown
        percentile: bottom percentile to consider as failures

    Returns:
        Dict with failure analysis results
    """
    errors = np.abs(y_true - y_pred)
    threshold = np.percentile(errors, 100 - percentile)
    worst_mask = errors >= threshold
    worst_indices = np.where(worst_mask)[0]

    results: Dict[str, Any] = {
        "n_failures": int(len(worst_indices)),
        "failure_mean_error": float(np.mean(errors[worst_mask])),
        "failure_max_error": float(np.max(errors[worst_mask])),
        "overall_mean_error": float(np.mean(errors)),
        "overall_median_error": float(np.median(errors)),
        "failure_threshold": float(threshold),
    }

    # Per-component breakdown
    if component_ids is not None:
        per_component = {}
        for comp in np.unique(component_ids):
            comp_mask = component_ids == comp
            if comp_mask.sum() == 0:
                continue
            comp_errors = errors[comp_mask]
            per_component[int(comp)] = {
                "mean_error": float(np.mean(comp_errors)),
                "n_samples": int(comp_mask.sum()),
                "n_failures": int(np.sum(worst_mask & comp_mask)),
            }
        results["per_component"] = per_component

    return results


# ------------------------------------------------------------------ #
#  Large Analysis Subset Builders                                    #
# ------------------------------------------------------------------ #

def build_stack_analysis(
    nova_data_dir: str,
    dataset_name: str = "single_channel_all",
    n_stacks: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a stack-based analysis subset for a single-channel dataset.

    Randomly selects `n_stacks` distinct stacks (stack_id = dataset_index // 10)
    and takes all 10 samples from each, giving n_stacks * 10 total entries.

    Args:
        nova_data_dir: parent directory containing all nova datasets
        dataset_name:  subdirectory name, e.g. "single_channel_all"
        n_stacks:      number of stacks to sample (default 100)
        seed:          RNG seed for reproducibility

    Returns:
        Ordered list of entry dicts compatible with structured_similarity_file format.
    """
    rng = np.random.default_rng(seed)
    ds_path = os.path.join(nova_data_dir, dataset_name)
    rows = _read_tsv_indices(ds_path)

    total_stacks = len(rows) // 10
    if total_stacks < n_stacks:
        raise ValueError(
            f"{dataset_name} has only {len(rows)} samples ({total_stacks} full stacks); "
            f"need at least {n_stacks}."
        )

    chosen_stacks = sorted(rng.choice(total_stacks, n_stacks, replace=False).tolist())

    entries: List[Dict] = []
    for s in chosen_stacks:
        for i in range(s * 10, s * 10 + 10):
            idx, fname = rows[i]
            entries.append({
                "dataset": dataset_name,
                "path": ds_path,
                "index": idx,
                "group": f"stack_{s}",
                "group_type": "stack",
                "filename": os.path.basename(fname),
            })
    return entries


def build_per_component_analysis(
    nova_data_dir: str,
    dataset_name: str,
    n_components: int = 10,
    n_samples_per_comp: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a per-component analysis subset for a multi-channel dataset.

    Randomly selects `n_components` components with enough samples and picks
    `n_samples_per_comp` samples from each, giving n_components * n_samples_per_comp
    total entries ordered by component.

    Args:
        nova_data_dir:      parent directory containing all nova datasets
        dataset_name:       subdirectory name, e.g. "multi_channel"
        n_components:       number of components to sample (default 10)
        n_samples_per_comp: samples per component (default 10)
        seed:               RNG seed for reproducibility

    Returns:
        Ordered list of entry dicts compatible with structured_similarity_file format.
    """
    rng = np.random.default_rng(seed)
    ds_path = os.path.join(nova_data_dir, dataset_name)
    rows = _read_tsv_indices(ds_path)
    comp_to_rows = _group_by_component(rows)

    eligible = sorted(c for c, r in comp_to_rows.items() if len(r) >= n_samples_per_comp)
    if len(eligible) < n_components:
        raise ValueError(
            f"{dataset_name}: only {len(eligible)} components have ≥{n_samples_per_comp} "
            f"samples; need {n_components}."
        )

    chosen_comps = sorted(rng.choice(eligible, n_components, replace=False).tolist())

    entries: List[Dict] = []
    for comp in chosen_comps:
        comp_rows = comp_to_rows[comp]
        picked = sorted(rng.choice(len(comp_rows), n_samples_per_comp, replace=False).tolist())
        for pos in picked:
            idx, fname = comp_rows[pos]
            entries.append({
                "dataset": dataset_name,
                "path": ds_path,
                "index": idx,
                "group": f"comp_{comp}",
                "group_type": "component",
                "filename": os.path.basename(fname),
            })
    return entries


def build_per_spec_index_analysis(
    nova_data_dir: str,
    dataset_name: str,
    n_spec_indices: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """
    Build a per-spec-index (cross-component) analysis subset for a multi-channel dataset.

    For each chosen spec_index, gathers one entry per component, so group size equals
    the total number of components in the dataset. Only spec indices present in ALL
    components are eligible.

    Args:
        nova_data_dir:   parent directory containing all nova datasets
        dataset_name:    subdirectory name, e.g. "multi_channel"
        n_spec_indices:  number of spec indices to sample (default 10)
        seed:            RNG seed for reproducibility

    Returns:
        Ordered list of entry dicts compatible with structured_similarity_file format.
        Entries are sorted by spec_index first, then by comp_id within each group.
    """
    rng = np.random.default_rng(seed)
    ds_path = os.path.join(nova_data_dir, dataset_name)
    rows = _read_tsv_indices(ds_path)

    spec_to_entries = _group_by_spec_index(rows)
    total_comps = len({extract_component_id(fname) for _, fname in rows
                       if extract_component_id(fname) is not None})

    if total_comps == 0:
        raise ValueError(f"{dataset_name}: no component IDs found in filenames.")

    # Keep only spec indices where every component is represented
    eligible = sorted(
        spec_idx for spec_idx, items in spec_to_entries.items()
        if len(items) == total_comps
    )

    if len(eligible) < n_spec_indices:
        raise ValueError(
            f"{dataset_name}: only {len(eligible)} spec indices have all {total_comps} "
            f"components present; need {n_spec_indices}."
        )

    chosen_specs = sorted(rng.choice(eligible, n_spec_indices, replace=False).tolist())

    entries: List[Dict] = []
    for spec_idx in chosen_specs:
        # Sort by comp_id within each spec group so similar-component samples are adjacent
        items_sorted = sorted(spec_to_entries[spec_idx], key=lambda x: x[0])  # sort by comp_id
        for comp_id, dataset_idx, fname in items_sorted:
            entries.append({
                "dataset": dataset_name,
                "path": ds_path,
                "index": dataset_idx,
                "group": f"spec_{spec_idx}",
                "group_type": "spec_index",
                "filename": os.path.basename(fname),
                "comp_id": comp_id,
            })
    return entries
