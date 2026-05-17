#!/usr/bin/env python3
"""Plot spectrogram tensor shapes through Data2VecAudio (SpectralFM conv stack) + masking.

Output: PNG under code/eval_results/ by default.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _add_box(ax, x, y, w, h, title, sub, fc="#ecf0f1", ec="#2c3e50"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.4",
        facecolor=fc, edgecolor=ec, linewidth=1.2,
    ))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#1a1a1a")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=8, color="#444")


def _arrow(ax, x1, y1, x2, y2, color="#333", ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        color=color, linewidth=1.4, linestyle=ls,
    ))


def build_figure(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")
    ax.set_title(
        "SpectralFM Data2VecAudio — signal shapes (245-bin conv stack, T=47 after FE)\n"
        "Solid = always in forward; dashed = Hydra pretraining only (mask=True, EMA)",
        fontsize=13, fontweight="bold", pad=16,
    )

    # Main forward row (y ~ 48)
    y = 48
    boxes = [
        (2,  "input\nspectrogram", "[B, 245]"),
        (16, "Conv FE\n(5 layers)", "[B, 512, 47]"),
        (30, "transpose +\nLayerNorm", "[B, 47, 512]\nfe_seq"),
        (44, "post_extract_proj", "[B, 47, 768]"),
        (58, "Transformer\nencoder 12L", "[B, 47, 768]\nx"),
    ]
    for i, (x, title, sub) in enumerate(boxes):
        _add_box(ax, x, y, 12, 10, title, sub,
                 fc="#aed6f1" if i == 0 else "#85c1e9" if i == 1 else "#5dade2")
        if i < len(boxes) - 1:
            _arrow(ax, x + 12, y + 5, boxes[i + 1][0], y + 5)

    # Masking branch (dashed into transformer input)
    _add_box(ax, 30, 30, 12, 8, "mask_emb", "[768] Param", fc="#d5d8dc", ec="#7f8c8d")
    _arrow(ax, 36, 38, 52, 48, color="#555", ls=(0, (4, 3)))
    ax.text(40, 42, "replace masked\npositions", fontsize=8, style="italic", color="#555")

    # Regression head (Hydra)
    _add_box(ax, 58, 30, 12, 8, "final_proj", "768→768\n(masked x only)", fc="#d5d8dc", ec="#7f8c8d")
    _arrow(ax, 64, 48, 64, 38, color="#555", ls=(0, (4, 3)))
    _add_box(ax, 72, 30, 14, 8, "EMA teacher", "regression target\n(smooth-L1)", fc="#bdc3c7", ec="#7f8c8d")
    _arrow(ax, 70, 34, 72, 34, color="#555", ls=(0, (4, 3)))
    ax.text(78, 26, "loss: regression", fontsize=9, color="#555", style="italic")

    # Recon branches
    _add_box(ax, 72, 48, 14, 10, "trans_recon_decoder\n(MirrorDecoder)", "pred_trans [B, 245]\nλ·recon_trans", fc="#e67e22", ec="#d35400")
    _arrow(ax, 70, 53, 72, 53, color="#e67e22")
    ax.text(79, 44, "loss: recon_trans", fontsize=9, color="#e67e22", style="italic")

    _add_box(ax, 30, 14, 14, 10, "fe_recon_decoder\n(MirrorDecoder)", "pred_fe [B, 245]\nλ·recon_fe", fc="#c0392b", ec="#922b21")
    _arrow(ax, 36, 48, 36, 24, color="#c0392b")
    ax.text(44, 20, "from fe_seq\n(post-LN)", fontsize=8, color="#c0392b", style="italic")

    # Mask visualization strip
    ax.text(2, 8, "Masking (mask_prob≈0.4, mask_length≈10):", fontsize=10, fontweight="bold")
    ax.text(2, 4,
            "Random span mask on encoder time steps → masked positions replaced with mask_emb before transformer.\n"
            "EMA regression compares final_proj(x_masked) to EMA(teacher); recon losses compare full-sequence "
            "decoder outputs to the original [B,245] spectrogram.",
            fontsize=8.5, color="#444", wrap=True)

    # Legend
    leg_y = 66
    ax.plot([2, 8], [leg_y, leg_y], color="#333", lw=2)
    ax.text(9, leg_y, "main forward", va="center", fontsize=9)
    ax.plot([28, 34], [leg_y, leg_y], color="#555", lw=1.5, ls=(0, (4, 3)))
    ax.text(35, leg_y, "Hydra pretrain only (mask + EMA regression)", va="center", fontsize=9)
    ax.plot([72, 78], [leg_y, leg_y], color="#e67e22", lw=2)
    ax.text(79, leg_y, "recon_trans (trainable in this experiment)", va="center", fontsize=9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "eval_results", "data2vec_signal_flow.png"
        ),
    )
    args = p.parse_args()
    build_figure(args.out)


if __name__ == "__main__":
    main()
