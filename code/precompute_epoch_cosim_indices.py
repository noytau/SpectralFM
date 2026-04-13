#!/usr/bin/env python3
"""Precompute FileAudioDataset *train-split* indices for epoch cosine maps.

The list is the structured-similarity panel (see ``build_structured_similarity_subset`` in
``eval_utils``), filtered to ``--task_data`` — not the full training set. Output filename
should reflect that (e.g. ``structured_similarity_<dataset>.npy``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    code_dir = os.path.dirname(os.path.abspath(__file__))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    from eval_utils import structured_subset_epoch_cosim_train_indices

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nova_data_dir", required=True)
    p.add_argument("--task_data", required=True)
    p.add_argument("--min_sample_size", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefer_manifest", choices=("train", "valid"), default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--dump_json_meta", default="")
    args = p.parse_args()

    indices = structured_subset_epoch_cosim_train_indices(
        args.nova_data_dir,
        args.task_data,
        min_sample_size=args.min_sample_size,
        seed=args.seed,
        prefer_manifest=args.prefer_manifest,
    )

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if out.endswith(".txt"):
        with open(out, "w") as f:
            for i in indices:
                f.write(f"{i}\n")
    else:
        import numpy as np

        np.save(out, np.array(indices, dtype=np.int64))

    print(f"[+] Wrote {len(indices)} indices to {out}")

    if args.dump_json_meta:
        meta = {
            "task_data": os.path.abspath(args.task_data),
            "min_sample_size": args.min_sample_size,
            "seed": args.seed,
            "prefer_manifest": args.prefer_manifest,
            "indices_train_dataset": indices,
        }
        jp = os.path.expanduser(args.dump_json_meta)
        os.makedirs(os.path.dirname(jp) or ".", exist_ok=True)
        with open(jp, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[+] Wrote meta to {jp}")


if __name__ == "__main__":
    main()
