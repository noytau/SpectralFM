"""
Create train.tsv / valid.tsv manifest files for fairseq audio_pretraining.

TSV format:
  Line 1 : root directory (absolute path to the folder containing wav files)
  Line N  : <relative_filename>\t<num_frames>

Usage on Geoffrey:
  python create_manifests.py \
      --wav_dir /mnt5/noy/fairseq/data/single_channel_1m/wavs \
      --out_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_all \
      --runai_root /storage/noy/fairseq/data/single_channel_1m/wavs \
      --valid_frac 0.001 \
      --subset all

  For a 10k subset:
  python create_manifests.py \
      --wav_dir /mnt5/noy/fairseq/data/single_channel_1m/wavs \
      --out_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k \
      --runai_root /storage/noy/fairseq/data/single_channel_1m/wavs \
      --valid_frac 0.01 \
      --max_train 10000 \
      --subset 10k
"""
import os
import random
import argparse
import torchaudio


def get_num_frames(wav_path: str) -> int:
    info = torchaudio.info(wav_path)
    return info.num_frames


def create_manifests(
    wav_dir: str,
    out_dir: str,
    runai_root: str,
    valid_frac: float = 0.001,
    max_train: int = None,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Scanning wav files in: {wav_dir}")
    files = sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav"))
    print(f"Found {len(files):,} wav files")

    if not files:
        raise FileNotFoundError(f"No .wav files found in {wav_dir}")

    random.seed(seed)
    random.shuffle(files)

    n_valid = max(1, int(len(files) * valid_frac))
    if max_train is not None:
        n_train = min(max_train, len(files) - n_valid)
    else:
        n_train = len(files) - n_valid

    valid_files = files[:n_valid]
    train_files = files[n_valid : n_valid + n_train]

    print(f"Split: {n_train:,} train, {n_valid:,} valid")

    # Infer num_frames from one file (all SpectralFM wavs have same length)
    sample_path = os.path.join(wav_dir, files[0])
    try:
        sample_frames = get_num_frames(sample_path)
        print(f"Detected {sample_frames} frames per file (from {files[0]})")
        fixed_frames = sample_frames
    except Exception as e:
        print(f"Warning: could not read frames from {files[0]}: {e}. Using 245.")
        fixed_frames = 245

    def write_tsv(split_files, path, root):
        with open(path, "w") as f:
            f.write(root + "\n")
            for fname in split_files:
                wav_path = os.path.join(wav_dir, fname)
                try:
                    n_frames = get_num_frames(wav_path)
                except Exception:
                    n_frames = fixed_frames
                f.write(f"{fname}\t{n_frames}\n")
        print(f"Written: {path}  ({len(split_files):,} entries)")

    train_tsv = os.path.join(out_dir, "train.tsv")
    valid_tsv = os.path.join(out_dir, "valid.tsv")

    # Use runai_root so the TSV works on the RunAI cluster (/storage/ mount)
    write_tsv(train_files, train_tsv, runai_root)
    write_tsv(valid_files, valid_tsv, runai_root)

    print(f"\nManifests written to: {out_dir}")
    print(f"  train.tsv: {n_train:,} files")
    print(f"  valid.tsv: {n_valid:,} files")
    print(f"  TSV root: {runai_root}")
    print("\nVerify on RunAI with:")
    print(f"  head -3 {out_dir}/train.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav_dir", required=True,
        help="Local path to directory containing .wav files (Geoffrey path)"
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Directory where train.tsv and valid.tsv will be written"
    )
    parser.add_argument(
        "--runai_root",
        help="Root path to write in TSV line 1 (RunAI /storage/ path). "
             "Defaults to wav_dir if not set.",
    )
    parser.add_argument(
        "--valid_frac", type=float, default=0.001,
        help="Fraction of files to use for validation (default: 0.001)"
    )
    parser.add_argument(
        "--max_train", type=int, default=None,
        help="Cap the number of training files (e.g. 10000 for 10k subset)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runai_root = args.runai_root or args.wav_dir

    create_manifests(
        wav_dir=args.wav_dir,
        out_dir=args.out_dir,
        runai_root=runai_root,
        valid_frac=args.valid_frac,
        max_train=args.max_train,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
