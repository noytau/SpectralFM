"""
CLI entry point: python -m eval.label_probe --step N ...
See plan.md's "Verification" section for the canonical invocations.
"""
from __future__ import annotations

import argparse
import os

from . import plots
from .study import run_step1


def main():
    ap = argparse.ArgumentParser(description="Label-probe study (see plan.md)")
    ap.add_argument("--step", type=int, required=True, choices=(1, 2, 3, 4, 5))
    # steps 2-5 read step 1's cache and need neither the checkpoint nor the
    # data dir; only step 1 does the extraction.
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--labeled_data_dir", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--comps", type=int, nargs="+", default=None,
                     help="restrict the component ladder to these counts, e.g. --comps 1 2")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                     help="if given, only run this many repeats (len(seeds))")
    ap.add_argument("--n_draws", type=int, default=100,
                     help="step 5 only: random training draws per (n_train, probe)")
    ap.add_argument("--n_eval", type=int, default=1000,
                     help="step 5 only: size of the fixed held-out eval set")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    comp_counts = tuple(args.comps) if args.comps else (1, 2, 3, 7, 12)
    # step 5 is expensive per cell (n_draws x n_train x probes), so it defaults
    # to a narrower ladder: the historical 3-comp config and the full 12.
    comp_counts_fewshot = tuple(args.comps) if args.comps else (3, 12)
    n_repeats = len(args.seeds) if args.seeds else 5

    if args.step == 1:
        if not args.checkpoint or not args.labeled_data_dir:
            ap.error("--step 1 requires --checkpoint and --labeled_data_dir")
        results, reps, y = run_step1(
            args.checkpoint, args.labeled_data_dir, args.output_dir,
            device=args.device, batch_size=args.batch_size,
            max_samples=args.max_samples, comp_counts=comp_counts,
            n_repeats=n_repeats,
        )
        fig_path = plots.plot_stage_comparison(results, os.path.join(
            args.output_dir, "label_reg_stages.png"))
        print(f"[label_probe] wrote {fig_path}")
    elif args.step == 5:
        # Few-shot / label-efficiency study. Reuses step 1's cache -- no GPU,
        # no re-extraction; --checkpoint/--labeled_data_dir are accepted for
        # a uniform CLI but unused here.
        from .study import run_step5
        results = run_step5(args.output_dir, n_draws=args.n_draws,
                             n_eval=args.n_eval, comp_counts=comp_counts_fewshot)
        for n_comp in comp_counts_fewshot:
            p = plots.plot_label_efficiency(
                results, os.path.join(args.output_dir,
                                       f"label_efficiency_{n_comp}comp.png"),
                n_comp=n_comp)
            print(f"[label_probe] wrote {p}")
    else:
        raise NotImplementedError(
            f"--step {args.step} not yet wired into __main__ (see study.py "
            "run_step2/3/4, callable directly)")


if __name__ == "__main__":
    main()
