#!/usr/bin/env python3
"""
Generate labels.tsv for the labeled_data dataset.

Reads dataset0022_ref (which contains parameter_0 per row), then maps every
WAV filename to its parameter_0 value. WAV filenames follow the pattern:
  dataset0022_comp{COMP}_spec_{ROW}.wav

Every component from the same row shares the same parameter_0 value.
The output labels.tsv is written alongside the WAV manifests so that the
evaluation runner can load it at eval time.

Usage:
    python generate_labels_tsv.py                          # default paths
    python generate_labels_tsv.py --ref_path /path/to/ref  # custom ref
    python generate_labels_tsv.py --dry_run                # preview only
"""

import os
import sys
import pickle
import argparse

# Auto-detect environment
BASE = '/storage/noy' if os.path.isdir('/storage/noy') else '/mnt5/noy'

DEFAULT_REF_PATH = os.path.join(BASE, 'nova_samples/labeled_data_test/v3/dataset0022_ref')
DEFAULT_WAV_DIR = os.path.join(BASE, 'SpectralFM/fairseq/data/nova_data/labeled_data/wavs')
DEFAULT_OUTPUT = os.path.join(BASE, 'SpectralFM/fairseq/data/nova_data/labeled_data/labels.tsv')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate labels.tsv for labeled_data')
    parser.add_argument('--ref_path', default=DEFAULT_REF_PATH,
                        help='Path to dataset0022_ref pickle file')
    parser.add_argument('--wav_dir', default=DEFAULT_WAV_DIR,
                        help='Directory containing WAV files (to verify filenames)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Output labels.tsv path')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be written without writing')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Ref file:  {args.ref_path}")
    print(f"WAV dir:   {args.wav_dir}")
    print(f"Output:    {args.output}")

    # 1. Load reference file
    if not os.path.exists(args.ref_path):
        print(f"ERROR: Ref file not found: {args.ref_path}")
        sys.exit(1)

    with open(args.ref_path, 'rb') as f:
        ref = pickle.load(f)

    if 'parameter_0' not in ref.columns:
        print(f"ERROR: 'parameter_0' column not found in ref. Columns: {list(ref.columns)}")
        sys.exit(1)

    print(f"Loaded ref: {len(ref)} rows, parameter_0 range "
          f"[{ref['parameter_0'].min():.4f}, {ref['parameter_0'].max():.4f}]")

    # 2. List WAV files to get actual component IDs
    if os.path.isdir(args.wav_dir):
        wav_files = sorted([f for f in os.listdir(args.wav_dir) if f.endswith('.wav')])
        print(f"Found {len(wav_files)} WAV files in {args.wav_dir}")
    else:
        wav_files = None
        print(f"WARNING: WAV dir not found ({args.wav_dir}), generating labels for all possible filenames")

    # 3. Build filename -> parameter_0 mapping
    # Parse component IDs from WAV filenames if available
    if wav_files:
        # Extract unique component IDs from existing WAVs
        comp_ids = set()
        for f in wav_files:
            # Format: dataset0022_comp{COMP}_spec_{ROW}.wav
            parts = f.replace('.wav', '').split('_comp')
            if len(parts) == 2:
                comp_and_row = parts[1].split('_spec_')
                if len(comp_and_row) == 2:
                    comp_ids.add(comp_and_row[0])
        comp_ids = sorted(comp_ids, key=lambda x: int(x))
        print(f"Component IDs found in WAVs: {comp_ids}")
    else:
        # Fallback: generate for a broad set of component IDs
        comp_ids = [str(i) for i in range(32)]

    # 4. Generate labels
    labels = []  # (filename, parameter_0)
    matched = 0
    unmatched = 0

    for row_idx in ref.index:
        p0 = ref.loc[row_idx, 'parameter_0']
        for comp_id in comp_ids:
            fname = f"dataset0022_comp{comp_id}_spec_{row_idx}.wav"
            labels.append((fname, p0))

    # If we have WAV files, verify coverage
    if wav_files:
        label_set = {fname for fname, _ in labels}
        for wf in wav_files:
            if wf in label_set:
                matched += 1
            else:
                unmatched += 1
        print(f"Coverage: {matched}/{len(wav_files)} WAVs matched ({unmatched} unmatched)")

    print(f"Total labels: {len(labels)}")

    # 5. Preview
    print(f"\nSample labels (first 5):")
    for fname, p0 in labels[:5]:
        print(f"  {fname}\t{p0:.6f}")
    print(f"  ...")
    for fname, p0 in labels[-3:]:
        print(f"  {fname}\t{p0:.6f}")

    if args.dry_run:
        print(f"\n[Dry run] Would write {len(labels)} labels to {args.output}")
        return

    # 6. Write labels.tsv
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for fname, p0 in labels:
            f.write(f"{fname}\t{p0:.6f}\n")

    print(f"\nWritten {len(labels)} labels to {args.output}")


if __name__ == "__main__":
    main()
