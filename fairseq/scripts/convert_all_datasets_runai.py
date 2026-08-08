#!/usr/bin/env python3
"""
Convert ALL nova pkl datasets to per-component WAV files.
Designed for RunAI server (uses /storage paths).

Converts:
  - multi_channel:  dataset0002 (14 comps), dataset0005 (8 comps)
  - labeled_data:   dataset0022 (16 comps)  [has parameter_0 labels]
  - sampled_data:   dataset0024 (28 comps)

Each component → separate WAV file (245 samples each).
Filename: {dataset}_comp{N}_spec_{row_idx}.wav

After conversion, prints a comparison table of all datasets.

Usage:
    python convert_all_datasets_runai.py              # full conversion
    python convert_all_datasets_runai.py --dry_run    # preview only
    python convert_all_datasets_runai.py --table_only # just print comparison table
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION — auto-detect /storage (RunAI) vs /mnt5 (local)
# ============================================================
SAMPLE_RATE = 16000
VALIDATION_FRACTION = 0.05

# Auto-detect base path
BASE = '/storage/noy' if os.path.isdir('/storage/noy') else '/mnt5/noy'

# Source PKL directories
PKL_SOURCES = {
    'multi_channel': f'{BASE}/nova_samples/multi_channel',
    'labeled_data':  f'{BASE}/nova_samples/labeled_data_test/v3',
    'sampled_data':  f'{BASE}/nova_samples/sampled_data',
}

# Output directories (under fairseq/data/nova_data/)
OUTPUT_BASE = f'{BASE}/SpectralFM/fairseq/data/nova_data'
OUTPUT_DIRS = {
    'multi_channel': os.path.join(OUTPUT_BASE, 'multi_channel'),
    'labeled_data':  os.path.join(OUTPUT_BASE, 'labeled_data'),
    'sampled_data':  os.path.join(OUTPUT_BASE, 'sampled_data'),
}

# Files to skip (ref/label files — not converted to WAVs)
SKIP_SUFFIXES = ('_ref',)


# ============================================================
# CONVERSION FUNCTIONS
# ============================================================

def parse_components(df):
    """
    Parse DataFrame columns to extract component groups.
    Handles: "component_X:Y" and "X:Y" formats.
    Returns: {comp_id: [sorted column names]}
    """
    components = {}
    for col in df.columns:
        col_str = str(col)
        if ':' not in col_str:
            continue
        prefix, feature_idx = col_str.rsplit(':', 1)
        comp_id = prefix.replace('component_', '') if prefix.startswith('component_') else prefix
        if comp_id not in components:
            components[comp_id] = []
        components[comp_id].append((int(feature_idx), col_str))

    result = {}
    for comp_id, cols in components.items():
        cols.sort(key=lambda x: x[0])
        result[comp_id] = [col_name for _, col_name in cols]
    return result


def find_pickle_files(input_dir):
    """Find .pkl and extensionless pickle files, skipping ref files.
    
    Deduplicates: if both 'dataset0002' and 'dataset0002.pkl' exist,
    only the .pkl version is kept.
    """
    files = []
    if not os.path.isdir(input_dir):
        print(f"  ⚠️  Directory not found: {input_dir}")
        return files
    
    seen_bases = {}  # base_name → full_path (prefer .pkl)
    for entry in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, entry)
        if not os.path.isfile(full_path):
            continue
        base = os.path.splitext(entry)[0]
        if any(base.endswith(s) for s in SKIP_SUFFIXES):
            continue
        if entry.endswith('.pkl') or '.' not in entry:
            # Prefer .pkl over extensionless if both exist
            if base in seen_bases:
                if entry.endswith('.pkl'):
                    seen_bases[base] = full_path  # upgrade to .pkl
                # else: keep existing .pkl
            else:
                seen_bases[base] = full_path
    
    files = [seen_bases[b] for b in sorted(seen_bases.keys())]
    return files


def process_pickle_file(pkl_path, output_wav_dir, dry_run=False, discard_constant=True):
    """Convert one pkl file → per-component WAVs. Returns [(filename, length), ...]
    
    Args:
        discard_constant: If True, skip spectrograms where all 245 features have
            the same value (std < 1e-6). Components 26 and 29 in dataset0002/0022
            are entirely constant (value=1.0) and carry zero information.
    """
    pkl_name = os.path.splitext(os.path.basename(pkl_path))[0]
    file_metadata = []

    print(f"\n  --- {pkl_name} ---")
    try:
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)
    except Exception as e:
        print(f"    ❌ Error loading: {e}")
        return []

    components = parse_components(df)
    if not components:
        print(f"    ⚠️ No feature columns found. Skipping.")
        return []

    comp_ids = sorted(components.keys(), key=lambda x: int(x))
    n_features = len(next(iter(components.values())))

    print(f"    Shape: {df.shape}")
    print(f"    Components ({len(comp_ids)}): {comp_ids}")
    print(f"    Features/component: {n_features}")
    print(f"    WAVs before filtering: {len(df)} rows × {len(comp_ids)} comps = {len(df) * len(comp_ids)}")

    # Pre-extract numpy arrays per component
    comp_matrices = {}
    for comp_id in comp_ids:
        comp_matrices[comp_id] = df[components[comp_id]].values.astype(np.float32)
    original_indices = df.index.tolist()
    del df

    # Identify constant components/rows to discard
    skipped_total = 0
    skipped_components = []
    if discard_constant:
        for comp_id in comp_ids:
            matrix = comp_matrices[comp_id]
            row_stds = matrix.std(axis=1)
            n_constant = (row_stds < 1e-6).sum()
            if n_constant == len(matrix):
                const_val = matrix[0, 0]
                skipped_components.append(comp_id)
                skipped_total += n_constant
                print(f"    ⚡ Discarding comp {comp_id}: ALL {n_constant} rows constant (value={const_val:.4f})")
            elif n_constant > 0:
                skipped_total += n_constant
                print(f"    ⚡ Comp {comp_id}: {n_constant}/{len(matrix)} constant rows will be skipped")

        if skipped_total > 0:
            total_before = len(original_indices) * len(comp_ids)
            print(f"    → Discarding {skipped_total} constant spectrograms "
                  f"({skipped_total / total_before * 100:.1f}% of total)")

    expected_total = len(original_indices) * len(comp_ids) - skipped_total

    if dry_run:
        print(f"    [dry run] ~{expected_total} WAVs would be written")
        return [(f"{pkl_name}_compX_spec_0.wav", n_features)] * expected_total

    # Write WAVs
    written = 0
    with tqdm(total=expected_total, desc=f"    {pkl_name}", unit="wav", leave=False) as pbar:
        for comp_id in comp_ids:
            if comp_id in skipped_components:
                continue

            matrix = comp_matrices[comp_id]

            if discard_constant:
                row_stds = matrix.std(axis=1)

            for i, original_idx in enumerate(original_indices):
                if discard_constant and row_stds[i] < 1e-6:
                    continue

                fname = f"{pkl_name}_comp{comp_id}_spec_{original_idx}.wav"
                save_path = os.path.join(output_wav_dir, fname)
                try:
                    sf.write(save_path, matrix[i], SAMPLE_RATE, subtype='FLOAT')
                    file_metadata.append((fname, len(matrix[i])))
                    written += 1
                except Exception as e:
                    print(f"    Error: row {i}, comp {comp_id}: {e}")
                pbar.update(1)

    print(f"    ✅ Written: {written} WAVs (skipped {skipped_total} constant)")

    del comp_matrices
    return file_metadata


def write_manifest(metadata_list, split_name, output_wav_dir, manifest_dir):
    """Write Fairseq-compliant TSV manifest."""
    tsv_path = os.path.join(manifest_dir, f"{split_name}.tsv")
    with open(tsv_path, "w") as tsv:
        tsv.write(output_wav_dir + "\n")
        for fname, length in metadata_list:
            tsv.write(f"{fname}\t{length}\n")
    print(f"    {split_name}.tsv: {len(metadata_list):,} samples")


def convert_dataset(dataset_name, pkl_dir, output_dir, dry_run=False):
    """Convert all pkl files in a directory to per-component WAVs."""
    print(f"\n{'='*60}")
    print(f"  Converting: {dataset_name}")
    print(f"  Source: {pkl_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    pkl_files = find_pickle_files(pkl_dir)
    if not pkl_files:
        print(f"  No pickle files found in {pkl_dir}")
        return

    print(f"  Found {len(pkl_files)} file(s): {[os.path.basename(f) for f in pkl_files]}")

    wav_dir = os.path.join(output_dir, 'wavs')
    if not dry_run:
        os.makedirs(wav_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    all_metadata = []
    for pkl_file in pkl_files:
        metadata = process_pickle_file(pkl_file, wav_dir, dry_run=dry_run)
        all_metadata.extend(metadata)

    print(f"\n  Total WAVs: {len(all_metadata):,}")

    if dry_run or len(all_metadata) == 0:
        return

    # Train/valid split
    train_meta, valid_meta = train_test_split(
        all_metadata,
        test_size=VALIDATION_FRACTION,
        random_state=42,
        shuffle=True,
    )
    write_manifest(train_meta, "train", wav_dir, output_dir)
    write_manifest(valid_meta, "valid", wav_dir, output_dir)


# ============================================================
# COMPARISON TABLE
# ============================================================

def print_comparison_table():
    """Print comparison table of all converted datasets."""
    print("\n")
    print("=" * 100)
    print("  DATASET COMPARISON TABLE (per-component WAV format)")
    print("=" * 100)

    # Header
    header = f"{'Dataset':<14} {'Source PKL':<14} {'Rows':>10} {'Comps':>6} {'WAVs':>12} {'Train':>12} {'Valid':>8} {'Feat':>5} {'Range':>16} {'Mean':>7} {'Std':>7}"
    print(header)
    print("-" * 100)

    for ds_name, output_dir in OUTPUT_DIRS.items():
        wav_dir = os.path.join(output_dir, 'wavs')
        if not os.path.isdir(wav_dir):
            print(f"  {ds_name:<14} — not converted yet")
            continue

        wavs = os.listdir(wav_dir)
        if not wavs:
            print(f"  {ds_name:<14} — empty")
            continue

        # Count per source dataset
        datasets = Counter()
        comps_per_ds = {}
        for w in wavs:
            parts = w.split('_spec_')[0]
            ds = parts.split('_comp')[0]
            comp = parts.split('_comp')[1] if '_comp' in parts else '?'
            datasets[ds] += 1
            if ds not in comps_per_ds:
                comps_per_ds[ds] = set()
            comps_per_ds[ds].add(comp)

        # Train/valid counts
        train_path = os.path.join(output_dir, 'train.tsv')
        valid_path = os.path.join(output_dir, 'valid.tsv')
        train_n = sum(1 for _ in open(train_path)) - 1 if os.path.exists(train_path) else 0
        valid_n = sum(1 for _ in open(valid_path)) - 1 if os.path.exists(valid_path) else 0

        # Sample a few wavs for stats
        import random
        random.seed(42)
        sample_wavs = random.sample(wavs, min(200, len(wavs)))
        all_vals = []
        for sw in sample_wavs:
            data, _ = sf.read(os.path.join(wav_dir, sw), dtype='float32')
            all_vals.append(data)
        all_vals = np.concatenate(all_vals)
        vmin, vmax = all_vals.min(), all_vals.max()
        vmean, vstd = all_vals.mean(), all_vals.std()
        feat_len = len(sf.read(os.path.join(wav_dir, sample_wavs[0]), dtype='float32')[0])

        for ds in sorted(datasets):
            n_comps = len(comps_per_ds[ds])
            rows = datasets[ds] // n_comps
            print(f"  {ds_name:<12} {ds:<14} {rows:>10,} {n_comps:>6} {datasets[ds]:>12,} {'':>12} {'':>8} {feat_len:>5} [{vmin:>6.2f}, {vmax:>5.2f}] {vmean:>7.4f} {vstd:>7.4f}")

        # Print totals line
        print(f"  {'':12} {'TOTAL':<14} {'':>10} {'':>6} {len(wavs):>12,} {train_n:>12,} {valid_n:>8,}")
        print()

    # Labels info
    print("-" * 100)
    print("  LABELS:")
    ref_path = os.path.join(PKL_SOURCES['labeled_data'], 'dataset0022_ref')
    if os.path.exists(ref_path):
        with open(ref_path, 'rb') as f:
            ref = pickle.load(f)
        if 'parameter_0' in ref.columns:
            p = ref['parameter_0']
            print(f"    dataset0022_ref → parameter_0: {p.nunique()} unique values, "
                  f"range [{p.min():.4f}, {p.max():.4f}], mean {p.mean():.4f}")
    else:
        print(f"    No ref file found at {ref_path}")

    print("=" * 100)

    # Component overlap table
    print("\n  COMPONENT OVERLAP:")
    print(f"  {'Comp IDs':<12} {'dataset0002 (14)':<18} {'dataset0005 (8)':<18} {'dataset0022 (16)':<18} {'dataset0024 (28)':<18}")
    print(f"  {'-'*80}")

    ds_comps = {}
    for ds_name_inner, pkl_dir in PKL_SOURCES.items():
        for pkl_file in find_pickle_files(pkl_dir):
            name = os.path.splitext(os.path.basename(pkl_file))[0]
            with open(pkl_file, 'rb') as f:
                df = pickle.load(f)
            comps = parse_components(df)
            ds_comps[name] = set(int(c) for c in comps.keys())
            del df

    all_comp_ids = sorted(set().union(*ds_comps.values()))
    for cid in all_comp_ids:
        row = f"  {cid:<12}"
        for ds in ['dataset0002', 'dataset0005', 'dataset0022', 'dataset0024']:
            if ds in ds_comps and cid in ds_comps[ds]:
                row += f"  {'✓':^16}"
            else:
                row += f"  {'—':^16}"
        print(row)

    print()


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Convert all nova datasets to per-component WAVs (RunAI)')
    parser.add_argument('--dry_run', action='store_true', help='Preview without writing files')
    parser.add_argument('--table_only', action='store_true', help='Just print comparison table (skip conversion)')
    parser.add_argument('--datasets', nargs='+', choices=list(PKL_SOURCES.keys()),
                        default=list(PKL_SOURCES.keys()),
                        help='Which datasets to convert (default: all)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Nova Dataset → Per-Component WAV Converter (RunAI)      ║")
    print("╚════════════════════════════════════════════════════════════╝")

    if not args.table_only:
        for ds_name in args.datasets:
            convert_dataset(
                ds_name,
                PKL_SOURCES[ds_name],
                OUTPUT_DIRS[ds_name],
                dry_run=args.dry_run,
            )

        if args.dry_run:
            print("\n[Dry run] No files written.")
            return

    print_comparison_table()


if __name__ == "__main__":
    main()
