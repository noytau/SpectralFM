"""
SpectralFM Evaluation Runner
------------------------------
Unified entry point. Runs any combination of evaluations and writes a timestamped
run directory (report + figures + CSVs) under --output_dir.

Zero fairseq dependency — see requirements.txt in this package.

USAGE MODES
===========

1. Single-checkpoint evals (embedding similarity / noise / clustering / label regression):

   python -m eval.runner \\
     --data_source /mnt5/noy/SpectralFM/fairseq/data/nova_data/single_channel_10k/wav \\
     --checkpoint_mode file --checkpoint_path /path/to/checkpoint.pt \\
     --evals embedding_similarity noise_robustness clustering label_regression \\
     --labeled_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data/labeled_data \\
     --output_dir eval_outputs

   checkpoint_mode: hf (HuggingFace facebook/data2vec-audio-base, no file needed),
   file (single .pt — fairseq or state_dict format), dir (picks best/last/latest in a dir).

2. Multi-checkpoint comparison (one row + one report section per checkpoint):

   python -m eval.runner \\
     --data_source .../single_channel_10k/wav \\
     --checkpoint_mode multiple --checkpoint_path /dir/of/checkpoints \\
     --evals checkpoint_comparison \\
     --nova_data_dir /mnt5/noy/SpectralFM/fairseq/data/nova_data \\
     --labeled_data_dir .../nova_data/labeled_data

   (or pass explicit files via EvalConfig.checkpoint_paths — see EvalConfig)

3. Signal reconstruction (multi-component checkpoints — FE and transformer parts
   are trained and saved SEPARATELY, so this eval takes its own checkpoint args,
   independent of --checkpoint_mode/--checkpoint_path):

   python -m eval.runner \\
     --data_source .../single_channel_10k/wav \\
     --evals signal_reconstruction \\
     --recon_fe_ckpt /path/to/ckpt_lr..._pretrained_base_libri.pt \\
     --recon_proj_ckpt /path/to/step2_projMlp2048_....pt \\
     --recon_tr_ckpt /path/to/ckpt_tr_lr....pt

   --recon_fe_ckpt : FE autoencoder ckpt (keys: encoder/layer_norm/decoder), or a
                     native fairseq checkpoint carrying an fe_recon_decoder.
                     Pathway: input → FE → LN → MirrorDecoder → signal.
   --recon_proj_ckpt : native fairseq checkpoint carrying a proj_recon_decoder.
                     Pathway: input → FE → LN → proj → mirror head → signal
                     (stops before the transformer encoder).
   --recon_tr_ckpt : transformer-recon ckpt (keys: transformer_mirror/backbone_ckpt).
                     Pathway: input → FE → LN → proj → transformer → mirror head → signal.
                     The backbone is loaded from the ckpt's recorded backbone_ckpt path.
   Any combination may be given. Per-sample layer_norm normalization is applied when
   recorded in the checkpoint (--recon_normalize true/false overrides).
   signal_reconstruction can be combined with checkpoint_comparison in one run.

Python API:
    runner = EvalRunner(EvalConfig(...))
    results = runner.run()
    runner.report(results)
"""
from __future__ import annotations
import os
import torch
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List

from .checkpoint_loader import CheckpointLoader
from .data_loader import (
    load_data, build_dataloader, split_stack_holdout, split_partial_stack,
    load_manifest_subset,
)
from .evaluations import (
    EmbeddingSimilarityEval,
    SignalReconstructionEval,
    NoiseRobustnessEval,
    CheckpointComparisonEval,
    ClusteringEval,
    LabelRegressionEval,
)
from .report import generate_report


EVAL_TYPES = [
    "embedding_similarity", "signal_reconstruction", "noise_robustness",
    "checkpoint_comparison", "clustering", "label_regression",
    "structured_similarity",
]

# ── E4: multi-dataset evaluation ───────────────────────────────────────────────
# alias → (nova_data subdir, manifest split, default sample count)
DATASET_SPECS = {
    "sanity":   ("single_channel_1k",  "train", 100),
    "in_dist":  ("single_channel_10k", "valid", 500),
    "multi_ch": ("multi_channel",      "valid", 1000),
    "samples":  ("sampled_data",       "valid", 1000),
    # "labeled" is loaded inside label_regression (labels.tsv, comp-grouped)
}

# Dataset × eval matrix: which datasets each eval runs on by default.
# label_regression runs only on the labeled set (handled separately);
# structured_similarity is run-level (its panel already spans datasets).
EVAL_DATASET_MATRIX = {
    "embedding_similarity":  ["sanity", "in_dist"],          # single-channel only
    "clustering":            ["in_dist"],                    # needs enough stacks
    "noise_robustness":      ["sanity", "in_dist"],
    "signal_reconstruction": ["sanity", "in_dist", "multi_ch", "samples"],
}
CHECKPOINT_MODES = ["hf", "file", "dir", "multiple"]
SPLIT_MODES = ["none", "stack_holdout", "partial_stack"]


@dataclass
class EvalConfig:
    # Data
    data_source: Optional[str] = None       # legacy single-dataset mode: .wav dir or CSV
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
    run_label_regression_in_comparison: bool = True
    label_reg_max_samples: int = 1000

    # Multi-dataset evaluation (E4). Aliases resolve under nova_data_dir; sizes
    # overridable via eval_set_sizes ("sanity=100,in_dist=500,...") or the global
    # eval_set_size shortcut. multi_dataset=False falls back to the single
    # data_source behaviour.
    multi_dataset: bool = True
    eval_set_sizes: Optional[str] = None
    eval_set_size: Optional[int] = None

    # Signal reconstruction (per-component checkpoints, independent of checkpoint_mode)
    recon_ckpt: Optional[str] = None         # single 3AE ckpt: all pathways (fe/proj/transformer)
    recon_fe_ckpt: Optional[str] = None      # FE AE ckpt: encoder/layer_norm/decoder keys
    recon_proj_ckpt: Optional[str] = None    # native ckpt carrying a proj_recon_decoder
    recon_tr_ckpt: Optional[str] = None      # TR recon ckpt: transformer_mirror/backbone_ckpt keys
    recon_normalize: Optional[bool] = None   # None → use flag recorded in ckpt
    recon_max_samples: int = 200

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
        d = args if isinstance(args, dict) else vars(args)
        # NB: hasattr(EvalConfig, k) misses default_factory fields (evals, checkpoint_paths)
        fields = set(EvalConfig.__dataclass_fields__)
        cfg = EvalConfig(**{k: v for k, v in d.items() if k in fields and v is not None})
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

    # ── Multi-dataset helpers (E4) ────────────────────────────────────────────

    def _dataset_sizes(self) -> dict:
        cfg = self.cfg
        sizes = {alias: n for alias, (_, _, n) in DATASET_SPECS.items()}
        if cfg.eval_set_size:
            sizes = {alias: int(cfg.eval_set_size) for alias in sizes}
        if cfg.eval_set_sizes:
            for part in str(cfg.eval_set_sizes).split(","):
                alias, _, val = part.strip().partition("=")
                if alias in sizes and val:
                    sizes[alias] = int(val)
        return sizes

    def _needed_aliases(self) -> set:
        cfg = self.cfg
        needed = set()
        comp_subevals = ["embedding_similarity"]
        if cfg.run_clustering_in_comparison:
            comp_subevals.append("clustering")
        if cfg.run_noise_in_comparison:
            comp_subevals.append("noise_robustness")
        if "checkpoint_comparison" in cfg.evals:
            for e in comp_subevals:
                needed |= set(EVAL_DATASET_MATRIX.get(e, []))
        for e in cfg.evals:
            needed |= set(EVAL_DATASET_MATRIX.get(e, []))
        if cfg.recon_ckpt or cfg.recon_fe_ckpt or cfg.recon_proj_ckpt or cfg.recon_tr_ckpt:
            needed |= set(EVAL_DATASET_MATRIX["signal_reconstruction"])
        return needed

    def _load_datasets(self, aliases: set) -> dict:
        cfg = self.cfg
        sizes = self._dataset_sizes()
        datasets = {}
        for alias in sorted(aliases):
            subdir, split, _ = DATASET_SPECS[alias]
            df_raw = load_manifest_subset(
                os.path.join(cfg.nova_data_dir, subdir), split=split,
                n=sizes[alias], seed=42,
            )
            loader, df = build_dataloader(
                df_raw.copy(deep=True),
                mask_ratio=cfg.mask_ratio, masking_type=cfg.masking_type,
                batch_size=cfg.batch_size,
            )
            datasets[alias] = {"df": df, "loader": loader, "df_raw": df_raw}
        return datasets

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self) -> dict:
        cfg = self.cfg
        print(f"[EvalRunner] Device: {cfg.device}")

        if not (cfg.multi_dataset and cfg.nova_data_dir):
            return self._run_single_dataset()

        # ── E4: multi-dataset flow ────────────────────────────────────────────
        aliases = self._needed_aliases()
        print(f"[EvalRunner] Multi-dataset mode — loading: {sorted(aliases)}")
        datasets = self._load_datasets(aliases)

        all_results = {"_dataset_info": {
            "sizes": {a: len(datasets[a]["df"]) for a in datasets},
            "matrix": {e: list(v) for e, v in EVAL_DATASET_MATRIX.items()},
        }}

        recon_requested = ("signal_reconstruction" in cfg.evals
                           or cfg.recon_ckpt or cfg.recon_fe_ckpt or cfg.recon_proj_ckpt or cfg.recon_tr_ckpt)

        # ── checkpoint_comparison ─────────────────────────────────────────────
        if "checkpoint_comparison" in cfg.evals:
            print("\n[EvalRunner] Running: checkpoint_comparison")
            checkpoints = self._load_model() if cfg.checkpoint_mode == "multiple" else None
            if checkpoints is None:
                print("[EvalRunner] checkpoint_comparison requires checkpoint_mode='multiple'. Skipping.")
            else:
                all_results["checkpoint_comparison"] = CheckpointComparisonEval.run(
                    checkpoints=checkpoints,
                    datasets=datasets,
                    eval_datasets=EVAL_DATASET_MATRIX,
                    device=cfg.device,
                    run_noise=cfg.run_noise_in_comparison,
                    run_clustering=cfg.run_clustering_in_comparison,
                    run_label_regression=cfg.run_label_regression_in_comparison,
                    k=cfg.k,
                    nova_data_dir=cfg.nova_data_dir,
                    labeled_data_dir=cfg.labeled_data_dir,
                    label_reg_max_samples=cfg.label_reg_max_samples,
                )

        # ── signal reconstruction: one result per matrix dataset ──────────────
        if recon_requested:
            # Reconstruction results are filed under the recon model's own name;
            # when it matches a compared checkpoint, they land inside its directory.
            recon_src = cfg.recon_ckpt or cfg.recon_fe_ckpt or cfg.recon_proj_ckpt or cfg.recon_tr_ckpt
            recon_model_name = os.path.splitext(os.path.basename(recon_src))[0] if recon_src else ""
            for alias in EVAL_DATASET_MATRIX["signal_reconstruction"]:
                if alias not in datasets:
                    continue
                print(f"\n[EvalRunner] Running: signal_reconstruction on {alias}")
                try:
                    res = SignalReconstructionEval.run(
                        df=datasets[alias]["df_raw"],
                        recon_ckpt=cfg.recon_ckpt,
                        fe_ckpt=cfg.recon_fe_ckpt,
                        proj_ckpt=cfg.recon_proj_ckpt,
                        tr_ckpt=cfg.recon_tr_ckpt,
                        device=cfg.device, arch=cfg.arch,
                        normalize=cfg.recon_normalize,
                        max_samples=cfg.recon_max_samples,
                        batch_size=cfg.batch_size,
                    )
                except (ValueError, FileNotFoundError, RuntimeError) as e:
                    print(f"[EvalRunner] signal_reconstruction ({alias}) failed: {e}")
                    res = {"skipped": True, "error": str(e)}
                res["_model_name"] = recon_model_name
                all_results[f"signal_reconstruction_{alias}"] = res

        # ── single-model evals ────────────────────────────────────────────────
        single_evals = [e for e in cfg.evals
                        if e not in ("checkpoint_comparison", "signal_reconstruction")]
        if single_evals:
            model = self._load_model() if cfg.checkpoint_mode != "multiple" else None
            if model is None and cfg.checkpoint_mode == "multiple":
                print("[EvalRunner] Loading first checkpoint for single-model evals.")
                model = self._load_model()[0][1]
            model.to(cfg.device)

            for eval_name in single_evals:
                if eval_name == "label_regression":
                    if not cfg.labeled_data_dir:
                        print("[EvalRunner] label_regression requires labeled_data_dir. Skipping.")
                    else:
                        print("\n[EvalRunner] Running: label_regression (labeled)")
                        all_results[eval_name] = LabelRegressionEval.run(
                            model=model, labeled_data_dir=cfg.labeled_data_dir,
                            device=cfg.device, batch_size=cfg.batch_size,
                            max_samples=cfg.label_reg_max_samples,
                        )
                elif eval_name == "structured_similarity":
                    from .evaluations import structured_similarity as struct_eval
                    print("\n[EvalRunner] Running: structured_similarity (run-level panel)")
                    all_results[eval_name] = struct_eval.run(
                        model, cfg.nova_data_dir, device=cfg.device,
                        batch_size=cfg.batch_size,
                    )
                elif eval_name in EVAL_DATASET_MATRIX:
                    for alias in EVAL_DATASET_MATRIX[eval_name]:
                        if alias not in datasets:
                            continue
                        ds = datasets[alias]
                        print(f"\n[EvalRunner] Running: {eval_name} on {alias}")
                        if eval_name == "embedding_similarity":
                            all_results[f"{eval_name}_{alias}"] = EmbeddingSimilarityEval.run(
                                df=ds["df"], model=model, dataloader=ds["loader"],
                                k=cfg.k, device=cfg.device)
                        elif eval_name == "noise_robustness":
                            all_results[f"{eval_name}_{alias}"] = NoiseRobustnessEval.run(
                                df=ds["df"], model=model, device=cfg.device,
                                batch_size=cfg.batch_size)
                        elif eval_name == "clustering":
                            all_results[f"{eval_name}_{alias}"] = ClusteringEval.run(
                                df=ds["df"], model=model, device=cfg.device,
                                batch_size=cfg.batch_size, k=cfg.k)
                else:
                    print(f"[EvalRunner] Unknown eval: {eval_name!r}")

        return all_results

    # ── Legacy single-dataset flow (multi_dataset=False) ──────────────────────

    def _run_single_dataset(self) -> dict:
        cfg = self.cfg
        print(f"[EvalRunner] Loading data from: {cfg.data_source}")
        df = self._load_data()
        print(f"[EvalRunner] Data shape: {df.shape}, stacks: {df['stack_idx'].nunique()}")

        # Keep raw (un-normalized) signals for signal_reconstruction — build_dataloader
        # layer-norms df['data'] in place, but the recon checkpoints expect raw input
        # unless their recorded normalize flag says otherwise.
        df_raw = df.copy(deep=True)

        dataloader, df = build_dataloader(
            df,
            mask_ratio=cfg.mask_ratio,
            masking_type=cfg.masking_type,
            batch_size=cfg.batch_size,
        )
        datasets = {"data": {"df": df, "loader": dataloader, "df_raw": df_raw}}
        matrix = {e: ["data"] for e in EVAL_DATASET_MATRIX}

        all_results = {}

        if "checkpoint_comparison" in cfg.evals:
            print("\n[EvalRunner] Running: checkpoint_comparison")
            checkpoints = self._load_model() if cfg.checkpoint_mode == "multiple" else None
            if checkpoints is None:
                print("[EvalRunner] checkpoint_comparison requires checkpoint_mode='multiple'. Skipping.")
            else:
                all_results["checkpoint_comparison"] = CheckpointComparisonEval.run(
                    checkpoints=checkpoints,
                    datasets=datasets,
                    eval_datasets=matrix,
                    device=cfg.device,
                    run_noise=cfg.run_noise_in_comparison,
                    run_clustering=cfg.run_clustering_in_comparison,
                    run_label_regression=cfg.run_label_regression_in_comparison,
                    k=cfg.k,
                    nova_data_dir=cfg.nova_data_dir,
                    labeled_data_dir=cfg.labeled_data_dir,
                    label_reg_max_samples=cfg.label_reg_max_samples,
                )

        if ("signal_reconstruction" in cfg.evals
                or cfg.recon_ckpt or cfg.recon_fe_ckpt or cfg.recon_proj_ckpt or cfg.recon_tr_ckpt):
            print("\n[EvalRunner] Running: signal_reconstruction")
            try:
                all_results["signal_reconstruction"] = SignalReconstructionEval.run(
                    df=df_raw,
                    recon_ckpt=cfg.recon_ckpt,
                    fe_ckpt=cfg.recon_fe_ckpt,
                    proj_ckpt=cfg.recon_proj_ckpt,
                    tr_ckpt=cfg.recon_tr_ckpt,
                    device=cfg.device, arch=cfg.arch,
                    normalize=cfg.recon_normalize,
                    max_samples=cfg.recon_max_samples,
                    batch_size=cfg.batch_size,
                )
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                print(f"[EvalRunner] signal_reconstruction failed: {e}")
                all_results["signal_reconstruction"] = {"skipped": True, "error": str(e)}

        single_evals = [e for e in cfg.evals
                        if e not in ("checkpoint_comparison", "signal_reconstruction")]
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
                elif eval_name == "structured_similarity":
                    if not cfg.nova_data_dir:
                        print("[EvalRunner] structured_similarity requires nova_data_dir. Skipping.")
                    else:
                        from .evaluations import structured_similarity as struct_eval
                        all_results[eval_name] = struct_eval.run(
                            model, cfg.nova_data_dir, device=cfg.device,
                            batch_size=cfg.batch_size,
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

_CLI_EPILOG = """
usage modes:

  1) single checkpoint:
     python -m eval.runner --data_source <wav_dir> \\
       --checkpoint_mode file --checkpoint_path ckpt.pt \\
       --evals embedding_similarity noise_robustness clustering label_regression

  2) multi-checkpoint comparison:
     python -m eval.runner --data_source <wav_dir> \\
       --checkpoint_mode multiple --checkpoint_path <dir_of_checkpoints> \\
       --evals checkpoint_comparison --nova_data_dir <nova_data>

  3) signal reconstruction (multi-component: FE and transformer parts have
     SEPARATE checkpoints; independent of --checkpoint_mode):
     python -m eval.runner --data_source <wav_dir> \\
       --evals signal_reconstruction \\
       --recon_fe_ckpt <fe_ae_ckpt.pt> --recon_tr_ckpt <tr_recon_ckpt.pt>

  Modes combine freely, e.g. --evals checkpoint_comparison signal_reconstruction.
  Each run writes a timestamped directory under --output_dir with
  eval_report.html / eval_report.md / run_info.md, all figures, and CSV exports.
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SpectralFM Evaluation Runner (fairseq-free)",
        epilog=_CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_source", default=None,
                        help="Legacy single-dataset mode: path to .wav directory or CSV "
                             "(with --single_dataset). Multi-dataset mode (default) draws "
                             "the E4 datasets from --nova_data_dir instead.")
    parser.add_argument("--single_dataset", action="store_true",
                        help="Disable E4 multi-dataset mode; evaluate --data_source only.")
    parser.add_argument("--eval_set_sizes", default=None,
                        help="Per-dataset sample counts, e.g. 'sanity=100,in_dist=500,"
                             "multi_ch=1000,samples=1000'. Defaults per DATASET_SPECS.")
    parser.add_argument("--eval_set_size", type=int, default=None,
                        help="Global sample-count override for all datasets (smoke runs).")
    parser.add_argument("--checkpoint_mode", default="file", choices=CHECKPOINT_MODES)
    parser.add_argument("--checkpoint_path", default=None, help="Path to .pt file or checkpoint dir")
    parser.add_argument("--checkpoint_paths", nargs="+", default=None,
                        help="Explicit list of checkpoint files for checkpoint_mode=multiple "
                             "(alternative to --checkpoint_path <dir>)")
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
    parser.add_argument("--recon_ckpt", default=None,
                        help="Signal reconstruction: single 3AE checkpoint (keys: data2vec_audio + "
                             "fe_mirror/proj_mirror/transformer_mirror). Runs ALL contained pathways: "
                             "FE, projection, transformer. Overrides --recon_fe_ckpt/--recon_tr_ckpt.")
    parser.add_argument("--recon_fe_ckpt", default=None,
                        help="Signal reconstruction: FE autoencoder checkpoint "
                             "(keys: encoder/layer_norm/decoder), or a native fairseq "
                             "checkpoint carrying an fe_recon_decoder. Pathway: FE→LN→MirrorDecoder.")
    parser.add_argument("--recon_proj_ckpt", default=None,
                        help="Signal reconstruction: native fairseq checkpoint carrying a "
                             "proj_recon_decoder (e.g. a Step 2-style run). Pathway: "
                             "FE→LN→proj→TransformerMirrorDecoder (stops before the transformer).")
    parser.add_argument("--recon_tr_ckpt", default=None,
                        help="Signal reconstruction: transformer-recon checkpoint "
                             "(keys: transformer_mirror/backbone_ckpt). Pathway: full "
                             "backbone→TransformerMirrorDecoder; backbone loaded from the "
                             "ckpt's recorded backbone_ckpt path.")
    parser.add_argument("--recon_normalize", default=None, choices=["true", "false"],
                        help="Override per-sample layer_norm normalization for reconstruction "
                             "(default: use the flag recorded in the checkpoint).")
    parser.add_argument("--recon_max_samples", type=int, default=200)

    args = parser.parse_args()
    if args.recon_normalize is not None:
        args.recon_normalize = args.recon_normalize == "true"
    args.multi_dataset = not args.single_dataset
    del args.single_dataset
    runner = EvalRunner.from_args(args)
    results = runner.run()
    md_path, html_path = runner.report(results)
    print(f"\nMD:   {md_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
