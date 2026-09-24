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
    ap.add_argument("--step", type=int, required=True, choices=(1, 2, 3, 4, 5, 6, 7))
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
    ap.add_argument("--bank_dir", default=None,
                     help="step 6 only: directory holding (or to hold) bank.npz")
    ap.add_argument("--phase", default="all", choices=("bank", "screen", "confirm", "all"),
                     help="step 6 only: which phase to run")
    ap.add_argument("--ladder", type=int, nargs="+", default=None,
                     help="step 7 only: override the n_train ladder")
    ap.add_argument("--sel_draws", type=int, default=None,
                     help="step 7 only: draws for the eval_a SELECTION pass at "
                          "every rung, overriding DRAWS_BY_N (reporting on "
                          "eval_b always keeps DRAWS_BY_N unchanged)")
    ap.add_argument("--sel_top_emb", type=int, default=None,
                     help="step 7 only: only rank the first N embedding "
                          "candidates during selection (HISTORICAL is always "
                          "carried regardless)")
    ap.add_argument("--sel_top_raw", type=int, default=None,
                     help="step 7 only: only rank the first N raw candidates "
                          "during selection (HISTORICAL is always carried "
                          "regardless)")
    ap.add_argument("--no_tabpfn", action="store_true",
                     help="step 6 only: drop TabPFN from the confirmation panel")
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
    elif args.step == 6:
        import json

        from . import readouts as ro
        from . import report6, study6

        bank_dir = args.bank_dir or os.path.join(args.output_dir, "bank")
        bank_path = os.path.join(bank_dir, "bank.npz")
        if args.phase in ("bank", "all"):
            if not args.checkpoint or not args.labeled_data_dir:
                ap.error("--step 6 --phase bank requires --checkpoint and "
                         "--labeled_data_dir")
            bank_path = ro.build_bank_cache(
                args.checkpoint, args.labeled_data_dir, bank_dir,
                comps=(0, 1, 2), max_samples=args.max_samples,
                device=args.device, batch_size=args.batch_size)

        screen_path = os.path.join(args.output_dir, "step6_screen.json")
        if args.phase in ("screen", "all"):
            screen = study6.run_screen(bank_path, args.output_dir,
                                        n_draws=30)
        elif os.path.exists(screen_path):
            with open(screen_path) as f:
                screen = json.load(f)
        else:
            screen = None

        if args.phase in ("confirm", "all"):
            if screen is None:
                ap.error(f"--phase confirm needs {screen_path}; run --phase screen first")
            confirm = study6.run_confirm(bank_path, screen, args.output_dir,
                                          n_draws=args.n_draws,
                                          use_tabpfn=not args.no_tabpfn)
            v = study6.verdict(confirm)
            with open(os.path.join(args.output_dir, "step6_verdict.json"), "w") as f:
                json.dump(v, f, indent=2)
            report6.write_step6_report(screen, confirm, v, args.output_dir)
            for n_comp in (1, 2, 3):
                report6.plot_screen(screen, os.path.join(
                    args.output_dir, f"step6_screen_{n_comp}comp.png"), n_comp)
            report6.plot_confirm(confirm, v, os.path.join(
                args.output_dir, "step6_confirm.png"))
            print(f"[step6] VERDICT: {v['verdict']} "
                  f"({v['n_cells_won']}/{v['n_cells']} cells)")
    elif args.step == 7:
        from . import crossover, report7

        bank_dir = args.bank_dir or os.path.join(args.output_dir, "bank")
        bank_path = os.path.join(bank_dir, "bank.npz")
        if not os.path.exists(bank_path):
            ap.error(f"--step 7 needs an existing bank at {bank_path}; "
                     "build it with --step 6 --phase bank")
        comp_counts = tuple(args.comps) if args.comps else (1, 2, 3)
        results = crossover.run_crossover(
            bank_path, args.output_dir, comp_counts=comp_counts,
            n_ladder=tuple(args.ladder) if args.ladder else None,
            sel_draws=args.sel_draws,
            sel_top_emb=args.sel_top_emb,
            sel_top_raw=args.sel_top_raw)
        report7.write_step7_report(results, args.output_dir)
        for n_comp in comp_counts:
            report7.plot_crossover(results, os.path.join(
                args.output_dir, f"step7_crossover_{n_comp}comp.png"), n_comp)
        for nc, cr in results["crossings"].items():
            print(f"[step7] {nc}-comp: crossed={cr['crossed']} "
                  f"n_cross={cr['n_cross']} interp={cr['n_cross_interp']} "
                  f"ci={cr['ci']}")
    else:
        raise NotImplementedError(
            f"--step {args.step} not yet wired into __main__ (see study.py "
            "run_step2/3/4, callable directly)")


if __name__ == "__main__":
    main()
