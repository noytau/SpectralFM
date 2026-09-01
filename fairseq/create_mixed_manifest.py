"""
Create a mixed train.tsv / valid.tsv manifest that combines multiple existing
nova_data subsets into one, with an independent oversample multiplier per source.

Unlike create_manifests.py (which scans a wav directory from scratch), this reads
each source's own existing train.tsv/valid.tsv, re-roots every filename to an
ABSOLUTE path against that source's *correct* wav directory (not whatever root
happens to be recorded on that source's own manifest — multi_channel/labeled_data's
existing manifests declare a /mnt5/... root that doesn't exist on gpu55/gpu56),
and concatenates. fairseq's raw_audio_dataset does `os.path.join(root_dir, fname)`
per file (raw_audio_dataset.py:306) — os.path.join returns `fname` unchanged when
it is already absolute, so a shared manifest can freely mix files that live under
completely different roots as long as every entry is absolute. The root line
written at the top of the output file is therefore a placeholder ("/"), never
actually used for resolution.

Usage — the 10%-labeled mix used for the "single_channel_mixed" subset:
  python create_mixed_manifest.py \
    --source single_channel_all:/storage/noy/SpectralFM/fairseq/data/nova_data/single_channel_all/wav:1 \
    --source multi_channel:/storage/noy/SpectralFM/fairseq/data/nova_data/multi_channel/wavs:1 \
    --source labeled_data:/storage/noy/SpectralFM/fairseq/data/nova_data/labeled_data/wavs:22 \
    --out_dir /storage/noy/SpectralFM/fairseq/data/nova_data/single_channel_mixed

Each --source is NAME:WAV_DIR:OVERSAMPLE, where WAV_DIR is the *correct* directory
those files actually live in on the machine that will train (not necessarily what
that source's own train.tsv declares), and OVERSAMPLE is an integer repeat count
applied only to the train split (valid stays at 1x from every source, no
duplication, so validation loss isn't inflated by repeated rows).

The existing train.tsv/valid.tsv split for each source is reused as-is — this
script does not re-shuffle or re-split at the file level, just re-roots and
repeats. Each source's manifest must already exist (created via create_manifests.py
or already present under nova_data/<name>/).
"""
import argparse
import os


def parse_source(spec: str):
    name, wav_dir, oversample = spec.rsplit(":", 2)
    return name, wav_dir, int(oversample)


def read_manifest_rows(manifest_path: str):
    """Returns list of (fname, num_frames) — ignores the declared root entirely."""
    rows = []
    with open(manifest_path) as f:
        next(f)  # skip root line
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            fname, frames = line.split("\t")
            rows.append((fname, frames))
    return rows


def build_split(sources, split: str, apply_oversample: bool):
    out_lines = []
    counts = {}
    for name, wav_dir, oversample in sources:
        src_manifest_dir = SRC_MANIFEST_DIRS[name]
        manifest_path = os.path.join(src_manifest_dir, f"{split}.tsv")
        if not os.path.isfile(manifest_path):
            print(f"  [{name}] no {split}.tsv found at {manifest_path}, skipping")
            continue
        rows = read_manifest_rows(manifest_path)
        repeat = oversample if apply_oversample else 1
        for _ in range(repeat):
            for fname, frames in rows:
                abs_path = os.path.join(wav_dir, fname)
                out_lines.append(f"{abs_path}\t{frames}")
        counts[name] = (len(rows), repeat, len(rows) * repeat)
        print(f"  [{name}] {split}: {len(rows):,} rows x{repeat} = {len(rows) * repeat:,}")
    return out_lines, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True,
                     help="NAME:WAV_DIR:OVERSAMPLE, repeatable. NAME must match an existing "
                          "nova_data/<NAME>/ directory (its train.tsv/valid.tsv are read for rows).")
    ap.add_argument("--nova_data_root", default="/storage/noy/SpectralFM/fairseq/data/nova_data",
                     help="Parent directory containing each --source NAME's own train.tsv/valid.tsv")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    sources = [parse_source(s) for s in args.source]

    global SRC_MANIFEST_DIRS
    SRC_MANIFEST_DIRS = {name: os.path.join(args.nova_data_root, name) for name, _, _ in sources}

    os.makedirs(args.out_dir, exist_ok=True)

    print("Building train.tsv (oversample applied)...")
    train_lines, train_counts = build_split(sources, "train", apply_oversample=True)
    print("Building valid.tsv (no oversample — 1x every source)...")
    valid_lines, valid_counts = build_split(sources, "valid", apply_oversample=False)

    with open(os.path.join(args.out_dir, "train.tsv"), "w") as f:
        f.write("/\n")
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(args.out_dir, "valid.tsv"), "w") as f:
        f.write("/\n")
        f.write("\n".join(valid_lines) + "\n")

    total_train = len(train_lines)
    print(f"\nWrote {total_train:,} train rows, {len(valid_lines):,} valid rows to {args.out_dir}")
    print("\nPer-source share of train epoch:")
    for name, (base, repeat, effective) in train_counts.items():
        pct = 100 * effective / total_train
        print(f"  {name:20s} base={base:>10,}  oversample={repeat:>3d}x  effective={effective:>10,}  ({pct:5.2f}% of epoch)")


if __name__ == "__main__":
    main()
