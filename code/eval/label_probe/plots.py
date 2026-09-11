"""
Label-efficiency ladder plot and axis-fixed true-vs-predicted grid.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PALETTE = ["#1565C0", "#EF6C00", "#2E7D32", "#8E24AA", "#C62828"]


def plot_label_efficiency(ladder_results: dict, output_path: str) -> str:
    """
    ladder_results: {readout_label: {n_train: {r2_median, r2_p25, r2_p75, ...}}}
    x = n_train (log), y = median held-out R2 with an IQR band, one line per
    readout.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (label, by_n) in enumerate(ladder_results.items()):
        n_trains = sorted(by_n)
        med = [by_n[n]["r2_median"] for n in n_trains]
        lo = [by_n[n]["r2_p25"] for n in n_trains]
        hi = [by_n[n]["r2_p75"] for n in n_trains]
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(n_trains, med, marker="o", label=label, color=color)
        ax.fill_between(n_trains, lo, hi, color=color, alpha=0.15)
    ax.axhline(0, color="black", linestyle=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("labeled training samples (n_train)")
    ax.set_ylabel("held-out R² (median, IQR band)")
    ax.set_title("Label efficiency: raw input vs. embedding readouts")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def _scatter_cell(ax, y_true, y_pred, color, title, axis_limits, is_best):
    r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - y_true.mean()) ** 2) + 1e-12)
    ax.scatter(y_true, y_pred, s=3, alpha=0.2, color=color, rasterized=True)
    lo, hi = axis_limits
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    best_tag = "★ BEST  " if is_best else ""
    ax.set_title(f"{best_tag}{title}\nR²={r2:.3f}", fontsize=8.5,
                 fontweight="bold" if is_best else "normal",
                 color="darkgreen" if is_best else "black")
    if is_best:
        for spine in ax.spines.values():
            spine.set_edgecolor("gold")
            spine.set_linewidth(3)
        ax.patch.set_facecolor("#fffff0")
    ax.set_xlabel("True", fontsize=7)
    ax.set_ylabel("Pred", fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=6)


def plot_true_vs_pred_grid(cells: dict, output_path: str,
                            axis_limits=(-2.0, 2.0)) -> str:
    """
    cells: {(row_label, col_label): {"y_true": arr, "y_pred": arr}}.

    Every panel shares the SAME axis limits (default [-2, 2]) so panels are
    directly, visually comparable — a badly-scaled tight cloud cannot look
    as good as a properly-scaled fit here. Points outside `axis_limits` are
    clipped from view rather than silently rescaling the frame.
    """
    row_labels = sorted({r for r, _ in cells})
    col_labels = list(dict.fromkeys(c for _, c in cells))  # preserve insertion order

    n_rows, n_cols = len(row_labels), len(col_labels)
    r2s = {k: 1.0 - np.sum((v["y_true"] - v["y_pred"]) ** 2) /
           (np.sum((v["y_true"] - v["y_true"].mean()) ** 2) + 1e-12)
           for k, v in cells.items()}
    best_key = max(r2s, key=r2s.get)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.6 * n_rows),
                              squeeze=False)
    fig.suptitle("Label regression — true vs. predicted (axes fixed across all panels)",
                 fontsize=13, fontweight="bold")
    for i, row in enumerate(row_labels):
        for j, col in enumerate(col_labels):
            key = (row, col)
            ax = axes[i][j]
            if key not in cells:
                ax.axis("off")
                continue
            color = _PALETTE[j % len(_PALETTE)]
            title = f"{col}\n{row}" if i == 0 else row
            _scatter_cell(ax, cells[key]["y_true"], cells[key]["y_pred"], color,
                          title, axis_limits, is_best=(key == best_key))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path
