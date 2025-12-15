#!/usr/bin/env python3
import argparse
import os
import pathlib
import pickle

def get_num_samples_from_pkl(pkl_path):
    """
    Load the .pkl file (or just open metadata) and return the number of frames/samples.
    Adjust based on your format (maybe a dict with key 'features' etc).
    """
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    # Example: assume obj is a numpy array or list of frames
    try:
        return obj.shape[0]
    except AttributeError:
        return len(obj)

def write_manifest(feature_folder: pathlib.Path, output_manifest: pathlib.Path):
    """
    Walk through the folder, find .pkl files, compute num_samples and write manifest.
    """
    # gather all pkl files
    feature_files = sorted(feature_folder.rglob("*.pkl"))
    if len(feature_files) == 0:
        raise RuntimeError(f"No .pkl files found under {feature_folder}")

    # compute common root (folder) path
    root = os.path.commonpath([str(f.parent) for f in feature_files])

    with open(output_manifest, "w", encoding="utf-8") as outf:
        outf.write(root + "\n")
        for fpath in feature_files:
            num_samples = get_num_samples_from_pkl(fpath)
            rel = os.path.relpath(str(fpath), root)
            outf.write(f"{rel}\t{num_samples}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Create manifest from feature .pkl files for fairseq"
    )
    parser.add_argument(
        "feature_folder", type=pathlib.Path,
        help="root folder containing .pkl feature files"
    )
    parser.add_argument(
        "output_manifest", type=pathlib.Path,
        help="path to write manifest .tsv"
    )
    args = parser.parse_args()

    write_manifest(args.feature_folder, args.output_manifest)

if __name__ == "__main__":
    main()
