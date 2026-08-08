"""
Regenerate structured-similarity comparison plots from a saved evaluation run,
adding CNN feature-extractor (FE) output cosine similarity alongside the existing
transformer-embedding cosine similarity.

Usage
-----
  python code/plot_structured_similarity_with_fe.py \\
      --eval_dir  code/eval_results/20260317_213209 \\
      --nova_data_dir /mnt5/noy/fairseq/data \\
      [--seed 42] \\
      [--device cuda]

The script writes updated plots into <eval_dir>/plots/ with a "_with_fe" suffix so
the originals are not overwritten.

What it regenerates
-------------------
  1. all_models_vs_input_structured_similarity_with_fe.png
       Multi-model 2-row heatmap:  row 0 = embeddings, row 1 = FE outputs
  2. plots/<run>/ embedding_similarity_comparison_structured_similarity_with_fe.png
       Per-run 2-row heatmap
  3. per_group_<group>_spectrogram_similarity_with_fe.png
       Per 10-sample group, columns = [input, embed, FE output]
  4. per_dataset_<dataset>_structured_similarity_with_fe.png
       Per-dataset heatmaps

Every panel shows a "Mean: / Std:" annotation from the upper-triangular values.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Fairseq / project path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FAIRSEQ_PATH = os.path.join(_THIS_DIR, "..", "fairseq")
if _FAIRSEQ_PATH not in sys.path:
    sys.path.insert(0, _FAIRSEQ_PATH)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annotate(ax, sim_matrix: np.ndarray) -> None:
    """Add Mean / Std annotation box from upper-triangular values."""
    triu = np.triu_indices_from(sim_matrix, k=1)
    vals = sim_matrix[triu]
    ax.text(
        0.02, 0.98,
        f"Mean: {vals.mean():.3f}\nStd:  {vals.std():.3f}",
        transform=ax.transAxes, fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def _heatmap(ax, sim_matrix: Optional[np.ndarray], title: str,
             color: str = "#333333", ylabel: Optional[str] = None,
             vmin: float = 0.0, vmax: float = 1.0,
             cmap: str = "viridis") -> None:
    """Plot a cosine-similarity heatmap with mean/std annotation."""
    if sim_matrix is not None and sim_matrix.shape[0] > 1:
        sns.heatmap(
            sim_matrix, ax=ax, cmap=cmap,
            xticklabels=False, yticklabels=False,
            vmin=vmin, vmax=vmax,
            cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
        )
        _annotate(ax, sim_matrix)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12,
                color="#E74C3C")
        ax.axis("off")

    ax.set_title(title, fontsize=10, fontweight="bold", color=color)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Sample Index", fontsize=9)


def _sim_matrix(vectors: np.ndarray) -> Optional[np.ndarray]:
    if vectors is None or len(vectors) < 2:
        return None
    return cosine_similarity(vectors)


# ---------------------------------------------------------------------------
# FE output extraction
# ---------------------------------------------------------------------------

def extract_fe_outputs(inputs: np.ndarray, checkpoint_path: str,
                       device_str: str = "cpu") -> Optional[np.ndarray]:
    """
    Pass saved inputs through the checkpoint's CNN feature extractor.

    Args:
        inputs:          float32 array of shape (N, T) – the raw model inputs
                         saved as inputs_*_structured_similarity.npy.
        checkpoint_path: path to the fairseq .pt checkpoint file.
        device_str:      'cpu' or 'cuda'.

    Returns:
        float32 array of shape (N, D) or None on failure.
    """
    import torch
    from model_loader import load_fairseq_checkpoint

    device = torch.device(device_str)
    try:
        model, _, _ = load_fairseq_checkpoint(checkpoint_path)
        model = model.to(device).eval()
    except Exception as exc:
        print(f"    [!] Could not load checkpoint {checkpoint_path}: {exc}")
        return None

    fe_vecs = []
    with torch.no_grad():
        for inp in inputs:
            src = torch.from_numpy(inp).float().unsqueeze(0).to(device)  # [1, T]
            try:
                fe = model.feature_extractor(src)          # [1, 512, T']
                fe = fe.transpose(1, 2)                    # [1, T', 512]
                fe = model.layer_norm(fe)                  # [1, T', 512]
                if model.post_extract_proj is not None:
                    fe = model.post_extract_proj(fe)       # [1, T', D]
                fe_vecs.append(fe.mean(dim=1).squeeze(0).cpu().numpy())
            except Exception as exc:
                print(f"    [!] FE extraction failed for one sample: {exc}")
                fe_vecs.append(None)

    valid = [v for v in fe_vecs if v is not None]
    if len(valid) < 2:
        print("    [!] Too few valid FE outputs – skipping.")
        return None
    # Fill failed samples with zero vectors of the right dimension
    D = valid[0].shape[0]
    result = []
    for v in fe_vecs:
        result.append(v if v is not None else np.zeros(D, dtype=np.float32))
    return np.stack(result).astype(np.float32)


# ---------------------------------------------------------------------------
# Group / dataset structure recovery
# ---------------------------------------------------------------------------

def recover_groups(nova_data_dir: str, seed: int = 42
                   ) -> Tuple[List[Dict], Optional[List[str]]]:
    """
    Re-run build_structured_similarity_subset to recover which index maps to
    which group.  Returns (entries, group_names_per_sample).
    entries[i]["group"] is the human-readable label for sample i.
    """
    try:
        from eval_utils import build_structured_similarity_subset
        entries = build_structured_similarity_subset(nova_data_dir, seed=seed)
        groups = [e["group"] for e in entries]
        return entries, groups
    except Exception as exc:
        print(f"[!] Could not recover group structure: {exc}")
        print("    Falling back to sequential 10-sample blocks.")
        return [], None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_two_row_heatmap(
    embeddings: np.ndarray,
    fe_outputs: Optional[np.ndarray],
    inputs: np.ndarray,
    run_name: str,
    save_path: Path,
    title: str,
) -> None:
    """
    Save a (1 or 2)-row heatmap: [embeddings] / [FE outputs].
    Always adds the input sim as the first column if small enough (<= 40 samples).
    """
    include_input = inputs is not None and len(inputs) <= 40
    n_cols = 2 if include_input else 1
    has_fe = fe_outputs is not None and len(fe_outputs) >= 2
    n_rows = 2 if has_fe else 1

    emb_sim = _sim_matrix(embeddings)
    fe_sim = _sim_matrix(fe_outputs) if has_fe else None
    inp_sim = _sim_matrix(inputs) if include_input else None

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 5 * n_rows),
                             squeeze=False)

    if include_input:
        _heatmap(axes[0, 0], inp_sim, "Input Space", ylabel="Embeddings")
        _heatmap(axes[0, 1], emb_sim, f"{run_name}\n(Embeddings)")
        if has_fe:
            _heatmap(axes[1, 0], inp_sim, "Input Space", ylabel="FE Output")
            _heatmap(axes[1, 1], fe_sim, f"{run_name}\n(FE Output)")
    else:
        _heatmap(axes[0, 0], emb_sim, f"{run_name}\n(Embeddings)", ylabel="Embeddings")
        if has_fe:
            _heatmap(axes[1, 0], fe_sim, f"{run_name}\n(FE Output)", ylabel="FE Output")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] Saved: {save_path}")


def _plot_all_models(
    run_data: List[Dict],
    inputs: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    """
    Multi-model 2-row comparison:
      Row 0: Input | run1 embed | run2 embed | ...
      Row 1: Input | run1 FE    | run2 FE    | ...
    """
    from eval_plots import plot_structured_similarity_all_models

    plot_structured_similarity_all_models(run_data, inputs, str(save_path), title=title)
    print(f"[+] Saved: {save_path}")


def _plot_per_group(
    group_name: str,
    group_indices: List[int],
    run_data: List[Dict],
    inputs: np.ndarray,
    save_path: Path,
) -> None:
    """
    Per-group 10-sample plot.
    Columns: Input | run1 embed | run1 FE | run2 embed | run2 FE | ...
    """
    idx = np.array(group_indices)
    grp_inputs = inputs[idx]
    inp_sim = _sim_matrix(grp_inputs)

    has_fe = any(r.get("fe_outputs") is not None for r in run_data)
    cols_per_run = 2 if has_fe else 1
    n_cols = 1 + len(run_data) * cols_per_run

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
    fig.suptitle(f"Group: {group_name}  (N={len(idx)} samples)",
                 fontsize=12, fontweight="bold")

    _heatmap(axes[0, 0], inp_sim, "Input Space")

    col = 1
    for rd in run_data:
        emb_sim = _sim_matrix(rd["embeddings"][idx] if rd["embeddings"] is not None else None)
        _heatmap(axes[0, col], emb_sim, f"{rd['run_name']}\n(Embed)")
        col += 1

        if has_fe:
            fe_out = rd.get("fe_outputs")
            fe_sim = _sim_matrix(fe_out[idx] if fe_out is not None else None)
            _heatmap(axes[0, col], fe_sim, f"{rd['run_name']}\n(FE Out)")
            col += 1

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [+] Saved: {save_path}")


def _plot_per_dataset(
    dataset_name: str,
    dataset_indices: List[int],
    run_data: List[Dict],
    inputs: np.ndarray,
    save_path: Path,
) -> None:
    """
    Per-dataset heatmap: same layout as _plot_per_group but for larger index sets.
    """
    _plot_per_group(
        group_name=f"dataset: {dataset_name}",
        group_indices=dataset_indices,
        run_data=run_data,
        inputs=inputs,
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir)
    data_dir = eval_dir / "data"
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Parse eval report to get run names and checkpoint paths
    # ------------------------------------------------------------------ #
    report_files = sorted(eval_dir.glob("eval_report_*.json"))
    report_files = [f for f in report_files if "enriched" not in f.name]
    if not report_files:
        print("[!] No eval_report_*.json found in", eval_dir)
        sys.exit(1)

    with open(report_files[0]) as f:
        report = json.load(f)

    print(f"[+] Loaded report: {report_files[0]}")
    run_records = report if isinstance(report, list) else [report]

    COLORS = ["#3498DB", "#E74C3C", "#27AE60", "#8E44AD", "#F39C12"]

    run_data: List[Dict] = []
    for i, rec in enumerate(run_records):
        run_name = rec.get("run_name", f"run_{i}")
        ckpt_path = rec.get("checkpoint_path", "")
        print(f"\n[+] Run: {run_name}")
        print(f"    Checkpoint: {ckpt_path}")

        # Load saved inputs and embeddings
        inputs_path = data_dir / f"inputs_{run_name}_structured_similarity.npy"
        emb_path = data_dir / f"embeddings_{run_name}_structured_similarity.npy"

        if not inputs_path.exists() or not emb_path.exists():
            print(f"    [!] Missing .npy files for {run_name} – skipping.")
            continue

        inputs_arr = np.load(inputs_path)
        embeddings_arr = np.load(emb_path)
        print(f"    inputs: {inputs_arr.shape}, embeddings: {embeddings_arr.shape}")

        # Extract FE outputs
        fe_outputs = None
        fe_cache_path = data_dir / f"fe_outputs_{run_name}_structured_similarity.npy"
        if fe_cache_path.exists() and not args.recompute:
            print(f"    Loading cached FE outputs from {fe_cache_path.name}")
            fe_outputs = np.load(fe_cache_path)
            print(f"    fe_outputs: {fe_outputs.shape}")
        elif os.path.isfile(ckpt_path):
            print(f"    Extracting FE outputs from checkpoint...")
            fe_outputs = extract_fe_outputs(inputs_arr, ckpt_path,
                                            device_str=args.device)
            if fe_outputs is not None:
                np.save(fe_cache_path, fe_outputs)
                print(f"    Saved FE outputs to {fe_cache_path.name}")
        else:
            print(f"    [!] Checkpoint not found at {ckpt_path}; skipping FE extraction.")

        run_data.append({
            "run_name": run_name,
            "inputs": inputs_arr,
            "embeddings": embeddings_arr,
            "fe_outputs": fe_outputs,
            "color": COLORS[i % len(COLORS)],
        })

    if not run_data:
        print("[!] No valid runs found.")
        sys.exit(1)

    # Use inputs from first run (should be identical across runs)
    inputs_all = run_data[0]["inputs"]
    N = len(inputs_all)
    print(f"\n[+] Total samples: {N}")

    # ------------------------------------------------------------------ #
    # 2. Recover group / dataset structure
    # ------------------------------------------------------------------ #
    entries, groups_per_sample = recover_groups(args.nova_data_dir, seed=args.seed)

    if groups_per_sample is None or len(groups_per_sample) != N:
        # Fall back: 10-sample sequential blocks
        print("[!] Using sequential 10-sample blocks as groups.")
        groups_per_sample = [f"block_{i // 10}" for i in range(N)]
        entries = []

    # Build index maps
    group_to_indices: Dict[str, List[int]] = {}
    dataset_to_indices: Dict[str, List[int]] = {}
    for i, g in enumerate(groups_per_sample):
        group_to_indices.setdefault(g, []).append(i)
    if entries:
        for i, e in enumerate(entries):
            dataset_to_indices.setdefault(e["dataset"], []).append(i)
    else:
        dataset_to_indices = {"all": list(range(N))}

    # ------------------------------------------------------------------ #
    # 3. Plot 1: all_models_vs_input with FE row
    # ------------------------------------------------------------------ #
    print("\n[+] Generating all-models comparison plot...")
    _plot_all_models(
        run_data=run_data,
        inputs=inputs_all,
        save_path=plots_dir / "all_models_vs_input_structured_similarity_with_fe.png",
        title=(
            "Structured Similarity Comparison — All Models\n"
            "(top row: transformer embeddings  |  bottom row: CNN FE outputs)"
        ),
    )

    # ------------------------------------------------------------------ #
    # 4. Plot 2: per-run embedding_similarity_comparison with FE row
    # ------------------------------------------------------------------ #
    print("\n[+] Generating per-run comparison plots...")
    for rd in run_data:
        run_plots_dir = plots_dir / rd["run_name"]
        run_plots_dir.mkdir(parents=True, exist_ok=True)
        _plot_two_row_heatmap(
            embeddings=rd["embeddings"],
            fe_outputs=rd["fe_outputs"],
            inputs=inputs_all,
            run_name=rd["run_name"],
            save_path=run_plots_dir
            / "embedding_similarity_comparison_structured_similarity_with_fe.png",
            title=f"Structured Similarity — {rd['run_name']}",
        )

    # ------------------------------------------------------------------ #
    # 5. Plot 3: per-group (10-sample) heatmaps with FE column
    # ------------------------------------------------------------------ #
    print("\n[+] Generating per-group plots...")
    for group_name, idx_list in sorted(group_to_indices.items()):
        safe_name = group_name.replace(" ", "_")
        _plot_per_group(
            group_name=group_name,
            group_indices=idx_list,
            run_data=run_data,
            inputs=inputs_all,
            save_path=plots_dir / f"per_group_{safe_name}_spectrogram_similarity_with_fe.png",
        )

    # ------------------------------------------------------------------ #
    # 6. Plot 4: per-dataset heatmaps with FE column
    # ------------------------------------------------------------------ #
    print("\n[+] Generating per-dataset plots...")
    for dataset_name, idx_list in sorted(dataset_to_indices.items()):
        safe_name = dataset_name.replace(" ", "_")
        _plot_per_dataset(
            dataset_name=dataset_name,
            dataset_indices=idx_list,
            run_data=run_data,
            inputs=inputs_all,
            save_path=plots_dir / f"per_dataset_{safe_name}_structured_similarity_with_fe.png",
        )

    print("\n[+] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate structured-similarity plots with FE output row."
    )
    parser.add_argument(
        "--eval_dir", required=True,
        help="Path to timestamped eval result directory, e.g. code/eval_results/20260317_213209",
    )
    parser.add_argument(
        "--nova_data_dir", default="/mnt5/noy/fairseq/data",
        help="Parent directory of nova datasets (default: /mnt5/noy/fairseq/data)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed used when building the structured subset (default: 42)",
    )
    parser.add_argument(
        "--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Device for FE extraction (default: auto-detect)",
    )
    parser.add_argument(
        "--recompute", action="store_true",
        help="Recompute FE outputs even if cached .npy files exist",
    )
    args = parser.parse_args()
    main(args)
