"""
Cosine similarity heatmaps for collapse diagnosis — no evaluation_runner needed.

For every checkpoint in --checkpoint_dir, extracts:
  • transformer embeddings  (model forward, mean-pool over time)
  • CNN FE outputs          (feature_extractor only, mean-pool over time)

Then plots an (N_ckpts × 2) grid of N×N cosine similarity heatmaps.
A collapsed model shows a near-uniform matrix (all values ≈ 1).

Usage
-----
  python code/eval_collapse_similarity.py \\
      --checkpoint_dir /mnt5/noy/SpectralFM/checkpoints/runai/fe_vs_transformer_collapse \\
      --eval_data_dir  /mnt5/noy/fairseq/data/single_channel_1m \\
      --n_samples 100 \\
      --output_dir code/eval_results/collapse_similarity
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── path setup ───────────────────────────────────────────────────────────────
_THIS = Path(__file__).parent.resolve()
_FAIRSEQ = _THIS.parent / "fairseq"
for p in [str(_THIS), str(_FAIRSEQ)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_loader import load_fairseq_checkpoint


# ── helpers ──────────────────────────────────────────────────────────────────

def load_wavs(data_dir: str, n: int, seed: int = 42) -> list[np.ndarray]:
    """Return up to *n* normalised float32 arrays from wav files in data_dir."""
    data_dir = Path(data_dir)
    wav_files = sorted(data_dir.rglob("*.wav"))
    rng = np.random.default_rng(seed)
    if len(wav_files) > n:
        wav_files = [wav_files[i] for i in rng.choice(len(wav_files), n, replace=False)]
    samples = []
    for p in wav_files:
        try:
            x, _ = sf.read(str(p), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            # unit-norm (matches task normalize=True)
            rms = np.sqrt(np.mean(x ** 2))
            if rms > 1e-8:
                x = x / rms
            samples.append(x)
        except Exception as exc:
            print(f"  [!] skip {p.name}: {exc}")
    print(f"[+] Loaded {len(samples)} WAV files from {data_dir}")
    return samples


def extract_embeddings(model, samples: list[np.ndarray], device) -> np.ndarray:
    """Full transformer pass → mean-pool → [N, D]."""
    vecs = []
    model.eval()
    with torch.no_grad():
        for x in samples:
            src = torch.from_numpy(x).float().unsqueeze(0).to(device)  # [1, T]
            try:
                out = model(src, padding_mask=None, mask=False, features_only=True)
                emb = out["x"].mean(dim=1).squeeze(0).cpu().numpy()    # [D]
                vecs.append(emb)
            except Exception as exc:
                print(f"  [!] embedding failed: {exc}")
    return np.stack(vecs) if vecs else np.empty((0, 768))


def extract_fe_outputs(model, samples: list[np.ndarray], device) -> np.ndarray:
    """CNN FE only → layer_norm → (post_extract_proj) → mean-pool → [N, D]."""
    vecs = []
    model.eval()
    with torch.no_grad():
        for x in samples:
            src = torch.from_numpy(x).float().unsqueeze(0).to(device)  # [1, T]
            try:
                fe = model.feature_extractor(src)          # [1, C, T']
                fe = fe.transpose(1, 2)                    # [1, T', C]
                fe = model.layer_norm(fe)                  # [1, T', C]
                if model.post_extract_proj is not None:
                    fe = model.post_extract_proj(fe)       # [1, T', D]
                vec = fe.mean(dim=1).squeeze(0).cpu().numpy()
                vecs.append(vec)
            except Exception as exc:
                print(f"  [!] FE extraction failed: {exc}")
    return np.stack(vecs) if vecs else np.empty((0, 512))


def sim_stats(mat: np.ndarray) -> tuple[float, float]:
    """Mean and std of upper-triangle (off-diagonal) cosine similarities."""
    idx = np.triu_indices_from(mat, k=1)
    v = mat[idx]
    return float(v.mean()), float(v.std())


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collapse cosine-similarity heatmaps")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Dir containing flat *.pt checkpoint files.")
    parser.add_argument("--eval_data_dir", required=True,
                        help="Dir tree containing *.wav evaluation files.")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of WAV samples to embed.")
    parser.add_argument("--output_dir", default="code/eval_results/collapse_similarity",
                        help="Where to save plots.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── discover checkpoints ─────────────────────────────────────────────────
    ckpt_paths = sorted(ckpt_dir.glob("*.pt"))
    if not ckpt_paths:
        print(f"[!] No .pt files found in {ckpt_dir}"); return
    print(f"[+] Found {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"    {p.name}")

    # ── load shared evaluation data ──────────────────────────────────────────
    samples = load_wavs(args.eval_data_dir, args.n_samples, args.seed)
    if len(samples) < 2:
        print("[!] Not enough samples to compute similarity."); return

    # ── extract embeddings for every checkpoint ───────────────────────────────
    # run_name → {emb: ndarray, fe: ndarray}
    results = {}
    for ckpt_path in ckpt_paths:
        # derive clean run name from filename stem
        stem = ckpt_path.stem
        # e.g. "short_fe-train_trans-train_base_2026-04-02_22-11-05" → strip date suffix
        run_name = "_".join(stem.split("_")[:-2]) if stem.count("_") >= 2 else stem

        print(f"\n[+] Loading: {run_name}")
        try:
            model, _, _ = load_fairseq_checkpoint(str(ckpt_path))
            model = model.to(device)
        except Exception as exc:
            print(f"  [!] Could not load {ckpt_path.name}: {exc}"); continue

        print(f"    extracting transformer embeddings...")
        emb = extract_embeddings(model, samples, device)
        print(f"    extracting CNN FE outputs...")
        fe  = extract_fe_outputs(model, samples, device)

        del model
        torch.cuda.empty_cache()

        results[run_name] = {"emb": emb, "fe": fe}
        # print per-checkpoint collapse stats
        if emb.shape[0] >= 2:
            m, s = sim_stats(cosine_similarity(emb))
            print(f"    emb  cos-sim  mean={m:.4f}  std={s:.4f}")
        if fe.shape[0] >= 2:
            m, s = sim_stats(cosine_similarity(fe))
            print(f"    fe   cos-sim  mean={m:.4f}  std={s:.4f}")

    if not results:
        print("[!] No results — aborting."); return

    # ── plot ──────────────────────────────────────────────────────────────────
    run_names = list(results.keys())
    n_runs = len(run_names)

    # grid: n_runs rows × 2 cols  (left=embedding, right=FE output)
    fig = plt.figure(figsize=(5 * 2, 4 * n_runs))
    gs  = gridspec.GridSpec(n_runs, 2, figure=fig, hspace=0.55, wspace=0.3)

    vmin, vmax = -1.0, 1.0

    for row_i, name in enumerate(run_names):
        for col_i, key in enumerate(["emb", "fe"]):
            mat_raw = results[name][key]
            ax = fig.add_subplot(gs[row_i, col_i])

            if mat_raw.shape[0] < 2:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{name}\n({'Embeddings' if col_i == 0 else 'FE output'})", fontsize=8)
                continue

            sim = cosine_similarity(mat_raw)
            mean_v, std_v = sim_stats(sim)

            sns.heatmap(
                sim, ax=ax, cmap="coolwarm",
                vmin=vmin, vmax=vmax,
                xticklabels=False, yticklabels=False,
                cbar_kws={"shrink": 0.8},
            )
            col_label = "Transformer\nEmbeddings" if col_i == 0 else "CNN FE\nOutputs"
            ax.set_title(f"{name}\n{col_label}", fontsize=7, pad=4)
            ax.text(
                0.02, 0.98,
                f"mean={mean_v:.3f}\nstd={std_v:.3f}",
                transform=ax.transAxes, fontsize=7,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    fig.suptitle(
        f"Cosine Similarity Heatmaps — Collapse Ablation\n"
        f"({len(samples)} samples · left=transformer emb · right=CNN FE output)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    plot_path = out_dir / "collapse_cosine_similarity.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[+] Saved: {plot_path}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Run':<45} {'emb_mean':>9} {'emb_std':>8} {'fe_mean':>9} {'fe_std':>8}")
    print("-" * 70)
    for name in run_names:
        emb = results[name]["emb"]
        fe  = results[name]["fe"]
        em, es = sim_stats(cosine_similarity(emb)) if emb.shape[0] >= 2 else (float("nan"),) * 2
        fm, fs = sim_stats(cosine_similarity(fe))  if fe.shape[0]  >= 2 else (float("nan"),) * 2
        print(f"{name:<45} {em:>9.4f} {es:>8.4f} {fm:>9.4f} {fs:>8.4f}")
    print("=" * 70)
    print(f"[+] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
