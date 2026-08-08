"""
Multi-component PKL → WAV converter.

Takes pkl files from nova_samples and converts them so that:
  - Each component becomes a SEPARATE wav file (245 features each)
  - Component number is embedded in the wav filename
  
Filename format: {dataset}_comp{comp_num}_spec_{row_index}.wav

Handles two column naming conventions:
  - "component_X:Y"  (e.g. dataset0002, dataset0024)
  - "X:Y"            (e.g. dataset0005)
"""

import os
import glob
import pickle
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# --- CONFIGURATION (defaults, overridable via CLI) ---
DEFAULT_INPUT_DIRS = [
    '/mnt5/noy/nova_samples/multi_channel',
    '/mnt5/noy/nova_samples/sampled_data',
    '/mnt5/noy/nova_samples/labeled_data_test/v3',
]
DEFAULT_OUTPUT_WAV_DIR = '/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data/wavs'
DEFAULT_MANIFEST_DIR = '/mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data'
DEFAULT_VALIDATION_FRACTION = 0.05
SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(description='Convert multi-component PKL files to per-component WAV files')
    parser.add_argument('--input_dirs', nargs='+', default=DEFAULT_INPUT_DIRS,
                        help='Directories containing .pkl files')
    parser.add_argument('--output_wav_dir', default=DEFAULT_OUTPUT_WAV_DIR,
                        help='Output directory for WAV files')
    parser.add_argument('--manifest_dir', default=DEFAULT_MANIFEST_DIR,
                        help='Output directory for TSV manifests')
    parser.add_argument('--validation_fraction', type=float, default=DEFAULT_VALIDATION_FRACTION,
                        help='Fraction of data for validation split')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be done without writing files')
    return parser.parse_args()


def setup_directories(output_wav_dir, manifest_dir):
    os.makedirs(output_wav_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)


def parse_components(df):
    """
    Parse DataFrame columns to extract component groups.
    
    Handles two formats:
      - "component_X:Y" → component id = X
      - "X:Y"           → component id = X
    
    Returns:
        dict: {component_id (str): [sorted list of column names]}
        e.g. {'0': ['component_0:0', 'component_0:1', ...], '1': [...]}
    """
    components = {}
    
    for col in df.columns:
        col_str = str(col)
        if ':' not in col_str:
            continue
        
        prefix, feature_idx = col_str.rsplit(':', 1)
        
        # Extract component number
        if prefix.startswith('component_'):
            comp_id = prefix.replace('component_', '')
        else:
            comp_id = prefix
        
        if comp_id not in components:
            components[comp_id] = []
        components[comp_id].append((int(feature_idx), col_str))
    
    # Sort columns within each component by feature index
    result = {}
    for comp_id, cols in components.items():
        cols.sort(key=lambda x: x[0])
        result[comp_id] = [col_name for _, col_name in cols]
    
    return result


def process_pickle_file(pkl_path, output_wav_dir, dry_run=False, discard_constant=True):
    """
    Loads a pkl file and converts each row × component into a separate WAV.
    
    Args:
        pkl_path: Path to the pickle file.
        output_wav_dir: Output directory for WAV files.
        dry_run: If True, prints what would be done without writing.
        discard_constant: If True, skip spectrograms where all 245 features
            have the same value (std < 1e-6). These carry zero information.
    
    Returns:
        list of (filename, length) tuples
    """
    pkl_name = os.path.splitext(os.path.basename(pkl_path))[0]
    file_metadata = []

    print(f"\n--- Processing: {pkl_name} ---")
    
    try:
        with open(pkl_path, 'rb') as f:
            df = pickle.load(f)
    except Exception as e:
        print(f"  ❌ Error loading {pkl_name}: {e}")
        return []

    # Parse components
    components = parse_components(df)
    if not components:
        print(f"  ⚠️ No feature columns found in {pkl_name}. Skipping.")
        return []

    comp_ids = sorted(components.keys(), key=lambda x: int(x))
    n_features = len(next(iter(components.values())))
    
    print(f"  Shape: {df.shape}")
    print(f"  Components: {len(comp_ids)} → {comp_ids}")
    print(f"  Features per component: {n_features}")
    print(f"  Rows: {len(df)}")
    print(f"  WAVs before filtering: {len(df)} rows × {len(comp_ids)} components = {len(df) * len(comp_ids)}")

    # Pre-extract numpy arrays per component for speed
    comp_matrices = {}
    for comp_id in comp_ids:
        cols = components[comp_id]
        comp_matrices[comp_id] = df[cols].values.astype(np.float32)

    original_indices = df.index.tolist()

    # Free the DataFrame early
    del df

    # Identify constant components/rows to discard
    skipped_total = 0
    skipped_components = []  # fully-constant components
    if discard_constant:
        for comp_id in comp_ids:
            matrix = comp_matrices[comp_id]
            row_stds = matrix.std(axis=1)
            n_constant = (row_stds < 1e-6).sum()
            if n_constant == len(matrix):
                # Entire component is constant — report and skip
                const_val = matrix[0, 0]
                skipped_components.append(comp_id)
                skipped_total += n_constant
                print(f"  ⚡ Discarding comp {comp_id}: ALL {n_constant} rows constant (value={const_val:.4f})")
            elif n_constant > 0:
                skipped_total += n_constant
                print(f"  ⚡ Comp {comp_id}: {n_constant}/{len(matrix)} constant rows will be skipped")
        
        if skipped_total > 0:
            total_before = len(original_indices) * len(comp_ids)
            print(f"  → Discarding {skipped_total} constant spectrograms "
                  f"({skipped_total/total_before*100:.1f}% of total)")

    expected_total = len(original_indices) * len(comp_ids) - skipped_total

    if dry_run:
        active_comps = [c for c in comp_ids if c not in skipped_components]
        for comp_id in active_comps:
            for original_idx in original_indices[:3]:
                fname = f"{pkl_name}_comp{comp_id}_spec_{original_idx}.wav"
                print(f"    [dry run] {fname}  ({n_features} samples)")
            print(f"    ... ({len(original_indices)} rows total for comp {comp_id})")
        print(f"  [dry run] ~{expected_total} WAVs would be written")
        return [(f"{pkl_name}_compX_spec_0.wav", n_features)] * expected_total

    # Write WAVs
    written = 0
    skipped_rows = 0
    with tqdm(total=expected_total, desc=f"  {pkl_name}", unit="wav", leave=False) as pbar:
        for comp_id in comp_ids:
            # Skip fully-constant components entirely
            if comp_id in skipped_components:
                continue

            matrix = comp_matrices[comp_id]
            
            # Pre-compute per-row std for partial-constant filtering
            if discard_constant:
                row_stds = matrix.std(axis=1)
            
            for i, original_idx in enumerate(original_indices):
                # Skip constant rows
                if discard_constant and row_stds[i] < 1e-6:
                    skipped_rows += 1
                    continue

                fname = f"{pkl_name}_comp{comp_id}_spec_{original_idx}.wav"
                save_path = os.path.join(output_wav_dir, fname)

                try:
                    data_row = matrix[i]
                    sf.write(save_path, data_row, SAMPLE_RATE, subtype='FLOAT')
                    file_metadata.append((fname, len(data_row)))
                    written += 1
                except Exception as e:
                    print(f"  Error: row {i}, comp {comp_id} in {pkl_name}: {e}")
                
                pbar.update(1)

    print(f"  ✅ Written: {written} WAVs (skipped {skipped_total} constant)")

    del comp_matrices
    return file_metadata


def write_manifest(metadata_list, split_name, output_wav_dir, manifest_dir):
    """Write Fairseq-compliant TSV manifest."""
    tsv_path = os.path.join(manifest_dir, f"{split_name}.tsv")
    print(f"Writing {split_name} manifest ({len(metadata_list)} samples) → {tsv_path}")
    
    with open(tsv_path, "w") as tsv:
        tsv.write(output_wav_dir + "\n")
        for fname, length in metadata_list:
            tsv.write(f"{fname}\t{length}\n")


def main():
    args = parse_args()

    print("=" * 60)
    print("Multi-Component PKL → WAV Converter")
    print("=" * 60)
    print(f"Input dirs:    {args.input_dirs}")
    print(f"Output WAVs:   {args.output_wav_dir}")
    print(f"Manifests:     {args.manifest_dir}")
    print(f"Valid frac:    {args.validation_fraction}")
    print(f"Dry run:       {args.dry_run}")
    print("=" * 60)

    if not args.dry_run:
        setup_directories(args.output_wav_dir, args.manifest_dir)

    # 1. Gather pickle files from all input directories
    #    Supports both .pkl files and extensionless pickle files (e.g. labeled_data_test)
    #    Deduplicates: if both 'datasetXXXX' and 'datasetXXXX.pkl' exist, only .pkl is used
    pkl_files = []
    SKIP_SUFFIXES = ('_ref',)  # skip ref/label files
    for input_dir in args.input_dirs:
        seen_bases = {}  # base_name → full_path (prefer .pkl)
        for entry in sorted(os.listdir(input_dir)):
            full_path = os.path.join(input_dir, entry)
            if not os.path.isfile(full_path):
                continue
            # Skip ref files
            base = os.path.splitext(entry)[0]
            if any(base.endswith(s) for s in SKIP_SUFFIXES):
                continue
            # Accept .pkl files or extensionless files
            if entry.endswith('.pkl') or '.' not in entry:
                if base in seen_bases:
                    if entry.endswith('.pkl'):
                        seen_bases[base] = full_path  # prefer .pkl
                else:
                    seen_bases[base] = full_path
        found = [seen_bases[b] for b in sorted(seen_bases.keys())]
        print(f"Found {len(found)} pkl files in {input_dir}")
        pkl_files.extend(found)
    
    if not pkl_files:
        print("CRITICAL: No .pkl files found in any input directory!")
        exit(1)

    print(f"\nTotal pkl files to process: {len(pkl_files)}")
    for f in pkl_files:
        print(f"  - {f}")

    # 2. Process all files
    all_metadata = []
    for pkl_file in pkl_files:
        metadata = process_pickle_file(pkl_file, args.output_wav_dir, dry_run=args.dry_run)
        all_metadata.extend(metadata)

    print(f"\n{'=' * 60}")
    print(f"Total WAV files: {len(all_metadata)}")

    if args.dry_run:
        print("[Dry run] No files written.")
        return

    # 3. Train/Valid split
    if len(all_metadata) > 0:
        pct_train = 100 - args.validation_fraction * 100
        pct_valid = args.validation_fraction * 100
        print(f"Splitting: {pct_train:.0f}% train, {pct_valid:.0f}% valid")

        train_meta, valid_meta = train_test_split(
            all_metadata,
            test_size=args.validation_fraction,
            random_state=42,
            shuffle=True,
        )

        # 4. Write manifests
        write_manifest(train_meta, "train", args.output_wav_dir, args.manifest_dir)
        write_manifest(valid_meta, "valid", args.output_wav_dir, args.manifest_dir)

        print(f"\n--- DONE ---")
        print(f"WAVs:      {args.output_wav_dir}")
        print(f"Manifests: {args.manifest_dir}")
        print(f"Train:     {len(train_meta)} samples")
        print(f"Valid:     {len(valid_meta)} samples")
    else:
        print("No samples were processed successfully.")


if __name__ == "__main__":
    main()
