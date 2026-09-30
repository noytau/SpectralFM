"""
Combine several label sets (see label_sets.resolve_label_sets) into ONE
directory that label_probe / label_regression can read as if it were a
single dataset -- for "all these datasets together" runs, alongside the
per-set sweep eval.runner already does when pointed at the parent
directly.

Every source wav is named `dataset<D>_comp<C>_spec_<S>.wav`, with D the
SOURCE dataset's own numeric id embedded in the filename -- so wavs from
different label sets never collide by name, and merging is just: symlink
every wav into one wavs/ directory, concatenate every labels.tsv into one.
Symlinks (not copies) because the source lives on a read-only NFS mount
this user cannot write into, and the merge is disposable and cheap to
rebuild.

  python -m eval.label_probe.merge_label_sets <parent_dir> <merged_out_dir>

Then point --labeled_data_dir / --data at <merged_out_dir> like any other
single label set.
"""
from __future__ import annotations

import argparse
import glob
import os

from .label_sets import resolve_label_sets


def _wav_root(set_dir: str) -> str:
    """Same search load_labeled_data uses: wav/, wavs/, or the directory
    itself, whichever actually holds the .wav files."""
    for sub in ("wav", "wavs"):
        cand = os.path.join(set_dir, sub)
        if glob.glob(os.path.join(cand, "*.wav")):
            return cand
    return set_dir


def merge(parent_dir: str, out_dir: str) -> dict:
    sets = resolve_label_sets(parent_dir)
    if not sets:
        raise RuntimeError(f"No usable label sets (non-empty labels.tsv) under {parent_dir}")

    wavs_out = os.path.join(out_dir, "wavs")
    os.makedirs(wavs_out, exist_ok=True)

    per_set_rows = {}
    label_lines = []
    seen_wavs = set()
    for name, set_dir in sorted(sets.items()):
        wav_root = _wav_root(set_dir)
        n_linked = 0
        for wav_path in glob.glob(os.path.join(wav_root, "*.wav")):
            fname = os.path.basename(wav_path)
            link_path = os.path.join(wavs_out, fname)
            if fname in seen_wavs:
                print(f"[merge_label_sets] WARNING: {fname!r} already linked "
                      f"(from an earlier set) -- skipping the copy from {name!r}. "
                      f"This means two source sets embed the same dataset id.")
                continue
            seen_wavs.add(fname)
            if not os.path.islink(link_path) and not os.path.exists(link_path):
                os.symlink(os.path.abspath(wav_path), link_path)
            n_linked += 1

        labels_path = os.path.join(set_dir, "labels.tsv")
        with open(labels_path) as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        label_lines.extend(lines)
        per_set_rows[name] = len(lines)
        print(f"[merge_label_sets] {name}: {n_linked} wavs linked, {len(lines)} label rows")

    with open(os.path.join(out_dir, "labels.tsv"), "w") as f:
        f.write("\n".join(label_lines) + "\n")

    print(f"[merge_label_sets] wrote {out_dir}/labels.tsv "
          f"({len(label_lines)} rows from {len(sets)} sets), "
          f"{len(seen_wavs)} wavs linked under {wavs_out}")
    return {"sets": per_set_rows, "total_label_rows": len(label_lines), "total_wavs": len(seen_wavs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parent_dir", help="directory containing one or more label sets, at any depth")
    ap.add_argument("out_dir", help="where to write the merged wavs/ + labels.tsv")
    args = ap.parse_args()
    merge(args.parent_dir, args.out_dir)


if __name__ == "__main__":
    main()
