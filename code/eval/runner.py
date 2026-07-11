"""
SpectralFM Evaluation Runner
------------------------------
Unified entry point. Supports all evaluation types with a single EvalRunner class.

Usage:
    runner = EvalRunner.from_args(args)
    results = runner.run()
    runner.report(results, output_dir="outputs/eval/")
"""
from __future__ import annotations
import os
import torch
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List

from .checkpoint_loader import CheckpointLoader
from .data_loader import load_data, build_dataloader, split_stack_holdout, split_partial_stack
from .evaluations import (
    EmbeddingSimilarityEval,
    SignalCompletionEval,
    NoiseRobustnessEval,
    CheckpointComparisonEval,
    ClusteringEval,
    LabelRegressionEval,
)
from .report import generate_report


EVAL_TYPES = [
    "embedding_similarity", "signal_completion", "noise_robustness",
    "checkpoint_comparison", "clustering", "label_regression",
]
CHECKPOINT_MODES = ["hf", "file", "dir", "multiple"]
SPLIT_MODES = ["none", "stack_holdout", "partial_stack"]


@dataclass
class EvalConfig:
    # Data
    data_source: str                        # path to dir of .wav files or CSV
    split_mode: str = "stack_holdout"       # none | stack_holdout | partial_stack
    n_holdout_stacks: int = 5
    holdout_ratio: float = 0.3

    # Checkpoint
    checkpoint_mode: str = "file"           # hf | file | dir | multiple
    checkpoint_path: Optional[str] = None   # used by file / dir
    checkpoint_pattern: str = "checkpoint*.pt"
    checkpoint_paths: List[str] = field(default_factory=list)  # used by multiple
    hf_model_name: str = "facebook/data2vec-audio-base"
    arch: str = "conv1d"

    # Eval
    evals: List[str] = field(default_factory=lambda: ["embedding_similarity"])
    k: int = 5
    run_noise_in_comparison: bool = True
    run_clustering_in_comparison: bool = True
    run_completion_in_comparison: bool = True
    run_label_regression_in_comparison: bool = True
    label_reg_max_samples: int = 2000

    # Runtime
    batch_size: int = 16
    mask_ratio: float = 0.15
    masking_type: str = "random"
    device: str = "auto"

    # Structured similarity (optional)
    nova_data_dir: Optional[str] = None   # path to nova_data/ root for canonical 100-sample panel

    # Label regression (optional) — defaults to <nova_data_dir>/labeled_data
    labeled_data_dir: Optional[str] = None

    # Output
    output_dir: str = "eval_outputs"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.labeled_data_dir is None and self.nova_data_dir:
            candidate = os.path.join(self.nova_data_dir, "labeled_data")
            if os.path.isdir(candidate):
                self.labeled_data_dir = candidate


class EvalRunner:
    def __init__(self, config: EvalConfig):
        self.cfg = config

    @classmethod
    def from_args(cls, args) -> "EvalRunner":
        """Build EvalRunner from argparse Namespace or dict."""
        if isinstance(args, dict):
            cfg = EvalConfig(**{k: v for k, v in args.items() if hasattr(EvalConfig, k)})
        else:
            cfg = EvalConfig(**{k: v for k, v in vars(args).items() if hasattr(EvalConfig, k)})
        return cls(cfg)

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def _load_model(self):
        cfg = self.cfg
        mode = cfg.checkpoint_mode
        if mode == "hf":
            return CheckpointLoader.from_hf(cfg.hf_model_name, arch=cfg.arch)
        elif mode == "file":
            return CheckpointLoader.from_file(cfg.checkpoint_path, arch=cfg.arch)
        elif mode == "dir":
            return CheckpointLoader.from_dir(cfg.checkpoint_path, arch=cfg.arch)
        elif mode == "multiple":
            source = cfg.checkpoint_paths or cfg.checkpoint_path
            return CheckpointLoader.load_multiple(source, arch=cfg.arch, pattern=cfg.checkpoint_pattern)
        raise ValueError(f"Unknown checkpoint_mode: {mode!r}. Choose from: {CHECKPOINT_MODES}")

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        cfg = self.cfg
        df = load_data(cfg.data_source)

        if "data" not in df.columns:
            # Assume numeric columns are the signal; combine into 'data' list column
            num_cols = [c for c in df.columns if c != "stack_idx"]
            df["data"] = df[num_cols].values.tolist()

        if cfg.split_mode == "stack_holdout":
            df = split_stack_holdout(df, n_holdout=cfg.n_holdout_stacks)
        elif cfg.split_mode == "partial_stack":
            df = split_partial_stack(df, holdout_ratio=cfg.holdout_ratio)

        return df

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        cfg = self.cfg
        print(f"[EvalRunner] Device: {cfg.device}")
        print(f"[EvalRunner] Loading data from: {cfg.data_source}")
        df = self._load_data()
        print(f"[EvalRunner] Data shape: {df.shape}, stacks: {df['stack_idx'].nunique()}")

        dataloader, df = build_dataloader(
            df,
            mask_ratio=cfg.mask_ratio,
            masking_type=cfg.masking_type,
            batch_size=cfg.batch_size,
        )

        all_results = {}

        # ── checkpoint_comparison handles its own model loading ───────────────
        if "checkpoint_comparison" in cfg.evals:
            print("\n[EvalRunner] Running: checkpoint_comparison")
            checkpoints = self._load_model() if cfg.checkpoint_mode == "multiple" else None
            if checkpoints is None:
                print("[EvalRunner] checkpoint_comparison requires checkpoint_mode='multiple'. Skipping.")
            else:
                all_results["checkpoint_comparison"] = CheckpointComparisonEval.run(
                    checkpoints=checkpoints,
                    df=df,
                    dataloader=dataloader,
                    device=cfg.device,
                    run_noise=cfg.run_noise_in_comparison,
                    run_clustering=cfg.run_clustering_in_comparison,
                    run_completion=cfg.run_completion_in_comparison,
                    run_label_regression=cfg.run_label_regression_in_comparison,
                    k=cfg.k,
                    nova_data_dir=cfg.nova_data_dir,
                    labeled_data_dir=cfg.labeled_data_dir,
                    label_reg_max_samples=cfg.label_reg_max_samples,
                )

        # ── single-model evals ────────────────────────────────────────────────
        single_evals = [e for e in cfg.evals if e != "checkpoint_comparison"]
        if single_evals:
            model = self._load_model() if cfg.checkpoint_mode != "multiple" else None
            if model is None and cfg.checkpoint_mode == "multiple":
                print("[EvalRunner] Loading first checkpoint for single-model evals.")
                model = self._load_model()[0][1]

            model.to(cfg.device)

            for eval_name in single_evals:
                print(f"\n[EvalRunner] Running: {eval_name}")
                if eval_name == "embedding_similarity":
                    all_results[eval_name] = EmbeddingSimilarityEval.run(
                        df=df, model=model, dataloader=dataloader, k=cfg.k, device=cfg.device
                    )
                elif eval_name == "signal_completion":
                    all_results[eval_name] = SignalCompletionEval.run(
                        df=df, model=model, dataloader=dataloader, device=cfg.device
                    )
                elif eval_name == "noise_robustness":
                    all_results[eval_name] = NoiseRobustnessEval.run(
                        df=df, model=model, device=cfg.device, batch_size=cfg.batch_size
                    )
                elif eval_name == "clustering":
                    all_results[eval_name] = ClusteringEval.run(
                        df=df, model=model, device=cfg.device,
                        batch_size=cfg.batch_size, k=cfg.k,
                    )
                elif eval_name == "label_regression":
                    if not cfg.labeled_data_dir:
                        print("[EvalRunner] label_regression requires labeled_data_dir. Skipping.")
                    else:
                        all_results[eval_name] = LabelRegressionEval.run(
                            model=model, labeled_data_dir=cfg.labeled_data_dir,
                            device=cfg.device, batch_size=cfg.batch_size,
                            max_samples=cfg.label_reg_max_samples,
                        )
                else:
                    print(f"[EvalRunner] Unknown eval: {eval_name!r}")

        return all_results

    def report(self, results: dict, output_dir: str | None = None) -> tuple[str, str]:
        """Generate markdown + HTML reports. Returns (md_path, html_path)."""
        out = output_dir or self.cfg.output_dir
        os.makedirs(out, exist_ok=True)
        return generate_report(results, output_dir=out, config=self.cfg)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="SpectralFM Evaluation Runner")
    parser.add_argument("--data_source", required=True, help="Path to .wav directory or CSV")
    parser.add_argument("--checkpoint_mode", default="file", choices=CHECKPOINT_MODES)
    parser.add_argument("--checkpoint_path", default=None, help="Path to .pt file or checkpoint dir")
    parser.add_argument("--checkpoint_pattern", default="checkpoint*.pt")
    parser.add_argument("--arch", default="conv1d")
    parser.add_argument("--evals", nargs="+", default=["embedding_similarity"], choices=EVAL_TYPES)
    parser.add_argument("--split_mode", default="stack_holdout", choices=SPLIT_MODES)
    parser.add_argument("--n_holdout_stacks", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--masking_type", default="random")
    parser.add_argument("--output_dir", default="eval_outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nova_data_dir", default=None, help="nova_data/ root for structured similarity")
    parser.add_argument("--labeled_data_dir", default=None, help="Dataset with labels.tsv for label regression")

    args = parser.parse_args()
    runner = EvalRunner.from_args(args)
    results = runner.run()
    md_path, html_path = runner.report(results)
    print(f"\nMD:   {md_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
