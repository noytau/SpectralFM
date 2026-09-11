"""
Usage:
  python -m eval.label_probe --checkpoint <path> --data <labeled_data_dir> \
      --out_dir <dir> [--device cuda] [--comps 1 2 3]
"""
from __future__ import annotations

import argparse

from .study import run_study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="labeled_data directory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--comps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_study(args.checkpoint, args.data, args.out_dir, device=args.device,
              comps_for_ladder=tuple(args.comps), seed=args.seed)


if __name__ == "__main__":
    main()
