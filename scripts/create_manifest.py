#!/usr/bin/env python3
"""
Create train.tsv and valid.tsv manifest files for fairseq audio pretraining.

Usage:
    python create_manifest.py /path/to/wavs_dir [--train-ratio 0.9] [--sample-length 245] [--seed 42]
    
Example:
    python create_manifest.py /mnt5/noy/fairseq/data/single_channel_10k/wavs --train-ratio 0.9
"""

import argparse
import random
from pathlib import Path


def get_wav_length(wav_path: Path, default_length: int = 245) -> int:
    """Get the number of samples in a wav file. Falls back to default if can't read."""
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return info.frames
    except Exception:
        return default_length


def create_manifest(
    wavs_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.9,
    sample_length: int = None,
    seed: int = 42,
    detect_length: bool = False,
):
    """Create train.tsv and valid.tsv manifest files."""
    random.seed(seed)
    
    # Get all wav files
    wav_files = sorted(list(wavs_dir.glob("*.wav")))
    if not wav_files:
        raise ValueError(f"No wav files found in {wavs_dir}")
    
    print(f"Found {len(wav_files)} wav files in {wavs_dir}")
    
    # Detect sample length from first file if not provided
    if sample_length is None:
        if detect_length:
            sample_length = get_wav_length(wav_files[0])
            print(f"Detected sample length: {sample_length}")
        else:
            sample_length = 245  # Default for SpectralFM
            print(f"Using default sample length: {sample_length}")
    
    # Get file names and lengths
    if detect_length:
        print("Detecting lengths for all files (this may take a while)...")
        file_data = [(f.name, get_wav_length(f, sample_length)) for f in wav_files]
    else:
        file_data = [(f.name, sample_length) for f in wav_files]
    
    # Shuffle and split
    random.shuffle(file_data)
    split_idx = int(len(file_data) * train_ratio)
    train_data = file_data[:split_idx]
    valid_data = file_data[split_idx:]
    
    print(f"Train: {len(train_data)} samples")
    print(f"Valid: {len(valid_data)} samples")
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write train.tsv
    train_path = output_dir / "train.tsv"
    with open(train_path, "w") as f:
        f.write(f"{wavs_dir}\n")
        for name, length in train_data:
            f.write(f"{name}\t{length}\n")
    print(f"Written: {train_path}")
    
    # Write valid.tsv
    valid_path = output_dir / "valid.tsv"
    with open(valid_path, "w") as f:
        f.write(f"{wavs_dir}\n")
        for name, length in valid_data:
            f.write(f"{name}\t{length}\n")
    print(f"Written: {valid_path}")
    
    return train_path, valid_path


def main():
    parser = argparse.ArgumentParser(
        description="Create fairseq audio manifest files from a directory of wav files"
    )
    parser.add_argument(
        "wavs_dir",
        type=Path,
        help="Directory containing wav files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for manifest files (default: parent of wavs_dir)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of files to use for training (default: 0.9)"
    )
    parser.add_argument(
        "--sample-length",
        type=int,
        default=None,
        help="Sample length for all files (default: 245 or detected)"
    )
    parser.add_argument(
        "--detect-length",
        action="store_true",
        help="Detect actual length of each wav file (slower)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    wavs_dir = args.wavs_dir.resolve()
    if not wavs_dir.exists():
        raise ValueError(f"Directory does not exist: {wavs_dir}")
    
    output_dir = args.output_dir if args.output_dir else wavs_dir.parent
    
    create_manifest(
        wavs_dir=wavs_dir,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        sample_length=args.sample_length,
        seed=args.seed,
        detect_length=args.detect_length,
    )


if __name__ == "__main__":
    main()
