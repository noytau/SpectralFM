"""
One command, any HuggingFace-style Transformer backbone (see readouts.py's
module docstring for what "any" covers): extraction, the recipe search, and
the label-efficiency panel, all in one run -- writes both
label_probe_results.json and recipe_panel.json, plus every figure.

  python -m eval.label_probe --checkpoint <path> --data <labeled_data_dir> \
      --out_dir <dir> [--device cuda] [--comps 1 2 3]

Trying a second backbone: point --checkpoint at it and --out_dir at a fresh
directory. Nothing else changes -- the run is self-identifying (meta.backbone
is auto-derived from the model class). Then line the two runs up:

  python -m eval.label_probe.compare <out_dir_1> <out_dir_2> [-o out.html]

Redraw figures without recomputing (seconds), after editing a plot:

  python -m eval.label_probe --plots_only <out_dir>/label_probe_results.json
  python -m eval.label_probe --panel_plots_only <out_dir>/recipe_panel.json

--plots_only redraws depth_profile.png, recipe_search.png,
probe_comparison.png (and true_vs_pred_grid.png if it already exists);
--panel_plots_only redraws crossover_panel.png, the one figure that comes
from the per-n_train panel rather than the full-pool search.
"""
from __future__ import annotations

import argparse

from .panel import write_panel_figures
from .study import replot_from_results, run_study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots_only", metavar="RESULTS_JSON",
                    help="redraw figures from a finished run's results JSON "
                         "and exit; no checkpoint or data needed")
    ap.add_argument("--panel_plots_only", metavar="RECIPE_PANEL_JSON",
                    help="redraw the label-efficiency panel's figure "
                         "(crossover_panel.png) and exit")
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
    if args.panel_plots_only:
        write_panel_figures(args.panel_plots_only, args.out_dir)
        return

    missing = [f"--{n}" for n in ("checkpoint", "data", "out_dir")
               if getattr(args, n) is None]
    if missing:
        ap.error(f"{', '.join(missing)} required unless --plots_only is given")

    run_study(args.checkpoint, args.data, args.out_dir, device=args.device,
              comps_for_ladder=tuple(args.comps), seed=args.seed)


if __name__ == "__main__":
    main()
