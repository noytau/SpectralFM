"""
Usage:
  python -m eval.label_probe --checkpoint <path> --data <labeled_data_dir> \
      --out_dir <dir> [--device cuda] [--comps 1 2 3]

  # redraw the figures from a finished run, no recompute (seconds):
  python -m eval.label_probe --plots_only <out_dir>/label_probe_results.json
"""
from __future__ import annotations

import argparse

from .study import replot_from_results, run_study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots_only", metavar="RESULTS_JSON",
                    help="redraw figures from a finished run's results JSON "
                         "and exit; no checkpoint or data needed")
    ap.add_argument("--checkpoint")
    ap.add_argument("--data", help="labeled_data directory")
    ap.add_argument("--out_dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--comps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.plots_only:
        replot_from_results(args.plots_only, args.out_dir)
        return

    missing = [f"--{n}" for n in ("checkpoint", "data", "out_dir")
               if getattr(args, n) is None]
    if missing:
        ap.error(f"{', '.join(missing)} required unless --plots_only is given")

    run_study(args.checkpoint, args.data, args.out_dir, device=args.device,
              comps_for_ladder=tuple(args.comps), seed=args.seed)


if __name__ == "__main__":
    main()
