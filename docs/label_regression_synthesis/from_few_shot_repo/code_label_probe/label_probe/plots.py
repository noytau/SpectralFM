"""
The headline figure: label_reg_stages.png.

Left panel  -- R2 vs forward-pass stage, one line per component config,
               error bars = +-1 SD over folds x repeats (r2_repeat_sd),
               dashed horizontal rule at that config's best input R2 so
               "did the signal survive this stage" is one visual read.
Right panel -- same x-axis, one line per regressor family, at the best
               component config, showing whether a non-linear probe finds
               signal a linear one misses.

Reuses report.py's save idiom (_save_fig: bbox_inches='tight', dpi=150).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .study import STAGE_LABELS, STAGE_ORDER

_PALETTE = ["#1565C0", "#EF6C00", "#2E7D32", "#8E24AA", "#C62828"]


def plot_stage_comparison(results: dict, output_path: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xs = np.arange(len(STAGE_ORDER))
    xlabels = [STAGE_LABELS[s] for s in STAGE_ORDER]

    # left panel: one line per comp config
    ax = axes[0]
    for i, (n_comp, block) in enumerate(sorted(results["stages"].items())):
        means = [block[s]["best_r2"] for s in STAGE_ORDER]
        sds = []
        for s in STAGE_ORDER:
            probe = block[s]["best_probe"]
            sds.append(block[s]["probes"][probe]["r2_repeat_sd"])
        color = _PALETTE[i % len(_PALETTE)]
        ax.errorbar(xs, means, yerr=sds, marker="o", label=f"{n_comp}-comp",
                    color=color, capsize=3)
        input_stage = "input_raw" if block["input_raw"]["best_r2"] >= block["input_z"]["best_r2"] else "input_z"
        ax.axhline(block[input_stage]["best_r2"], color=color, linestyle="--",
                   alpha=0.4, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=25, ha="right")
    ax.set_ylabel("R² (best-of-panel, out-of-fold)")
    ax.set_title("Where does the label signal survive?\n(dashed = that config's best input R²)")
    ax.legend(title="components", fontsize=8)
    ax.grid(alpha=0.3)

    # right panel: one line per regressor family, at the best comp config
    ax2 = axes[1]
    best_n_comp = max(results["stages"], key=lambda n: results["stages"][n][STAGE_ORDER[-1]]["best_r2"])
    block = results["stages"][best_n_comp]
    probes = sorted({p for s in STAGE_ORDER for p in block[s]["probes"]})
    for j, probe in enumerate(probes):
        means = [block[s]["probes"].get(probe, {}).get("r2_mean", np.nan) for s in STAGE_ORDER]
        sds = [block[s]["probes"].get(probe, {}).get("r2_repeat_sd", 0) for s in STAGE_ORDER]
        ax2.errorbar(xs, means, yerr=sds, marker="s", label=probe,
                     color=_PALETTE[j % len(_PALETTE)], capsize=3)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(xlabels, rotation=25, ha="right")
    ax2.set_ylabel("R²")
    ax2.set_title(f"Regressor comparison at {best_n_comp}-comp")
    ax2.legend(title="probe", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("Label regression across the forward pass", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def _scatter_cell(ax, cell: dict, color: str, row_label: str, col_label: str,
                   is_best: bool, show_col_title: bool):
    """
    One true-vs-predicted panel. Matches the existing report.py idiom
    (report.py:_label_reg_sweep_cell) exactly: R²/r/MAE + train/pred mean/std
    in the title, gold border + cream background on the best cell.
    """
    y_true, y_pred = cell["y_true"], cell["y_pred"]
    ax.scatter(y_true, y_pred, s=3, alpha=0.2, color=color, rasterized=True)
    lo = min(y_true.min(), y_pred.min()) - 0.05
    hi = max(y_true.max(), y_pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    best_tag = "★ BEST  " if is_best else ""
    title = f"{col_label}\n{row_label}" if show_col_title else row_label
    ax.set_title(
        f"{best_tag}{title}\n"
        f"R²={cell['r2']:.3f}  r={cell['pearson_r']:.3f}  MAE={cell['mae']:.3f}\n"
        f"true  μ={y_true.mean():.2f} σ={y_true.std():.2f}\n"
        f"pred  μ={y_pred.mean():.2f} σ={y_pred.std():.2f}",
        fontsize=7.5, fontweight="bold" if is_best else "normal",
        color="darkgreen" if is_best else "black",
    )
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
                            comp_counts=None, stages=None) -> str:
    """
    Full true-vs-predicted grid: rows = component counts, columns = forward-
    pass stages. One scatter panel per cell (from study.collect_true_vs_pred_grid),
    global-best cell highlighted in gold. Mirrors the legacy
    label_reg_evaluation.py / report.py sweep-grid style so this eval's
    figures look like the rest of the project's.
    """
    if comp_counts is None:
        comp_counts = sorted({n for n, _ in cells})
    if stages is None:
        stages = STAGE_ORDER

    n_rows, n_cols = len(comp_counts), len(stages)
    best_key = max(cells, key=lambda k: cells[k]["r2"])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.8 * n_rows),
                              squeeze=False)
    fig.suptitle("Label regression — true vs predicted, every stage × component count",
                 fontsize=13, fontweight="bold")
    for i, n_comp in enumerate(comp_counts):
        for j, stage in enumerate(stages):
            key = (n_comp, stage)
            if key not in cells:
                axes[i][j].axis("off")
                continue
            color = _PALETTE[j % len(_PALETTE)]
            _scatter_cell(
                axes[i][j], cells[key], color,
                row_label=f"{n_comp}-comp", col_label=STAGE_LABELS[stage],
                is_best=(key == best_key), show_col_title=(i == 0),
            )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path


def plot_label_efficiency(results: dict, output_path: str, n_comp: int = 12,
                           mode: str = "transductive") -> str:
    """
    Step 5's headline: label-efficiency curves. x = n_train (log), y = median
    held-out R² with an IQR band, one line per stage, dashed rule at each
    stage's asymptotic (step 1) R². The gap between a curve and its own
    dashed rule is what "20 labels instead of 4,716" actually costs.
    """
    from .study import _best_fewshot

    n_trains = results["n_train_values"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    for i, stage in enumerate(STAGE_ORDER):
        med, lo, hi = [], [], []
        for n in n_trains:
            got = _best_fewshot(results, n_comp, stage, n, mode)
            if got is None:
                med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            v = got[1]
            med.append(v["r2_median"]); lo.append(v["r2_p25"]); hi.append(v["r2_p75"])
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(n_trains, med, marker="o", label=STAGE_LABELS[stage], color=color)
        ax.fill_between(n_trains, lo, hi, color=color, alpha=0.15)
        asym = results.get("asymptotic_r2", {}).get(f"{n_comp}|{stage}")
        if asym is not None:
            ax.axhline(asym, color=color, linestyle="--", alpha=0.35, linewidth=1)
    ax.axvline(20, color="black", linestyle=":", alpha=0.6)
    ax.annotate("client's 20-label budget", xy=(20, ax.get_ylim()[0]),
                xytext=(21, ax.get_ylim()[0]), fontsize=8, rotation=90,
                va="bottom", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xticks(n_trains)
    ax.set_xticklabels([str(n) for n in n_trains])
    ax.set_xlabel("labeled training samples")
    ax.set_ylabel("median held-out R² (IQR band)")
    ax.set_title(f"Label efficiency — {n_comp}-comp, {mode}\n"
                 "(dashed = that stage's asymptotic R² at n≈4,716)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # right panel: reliability -- how often does it beat predicting the mean?
    ax2 = axes[1]
    for i, stage in enumerate(STAGE_ORDER):
        frac = []
        for n in n_trains:
            got = _best_fewshot(results, n_comp, stage, n, mode)
            frac.append(got[1]["frac_positive_r2"] if got else np.nan)
        ax2.plot(n_trains, frac, marker="s", label=STAGE_LABELS[stage],
                 color=_PALETTE[i % len(_PALETTE)])
    ax2.axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax2.axvline(20, color="black", linestyle=":", alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xticks(n_trains)
    ax2.set_xticklabels([str(n) for n in n_trains])
    ax2.set_xlabel("labeled training samples")
    ax2.set_ylabel("fraction of draws with R² > 0")
    ax2.set_title("Reliability — how often it beats predicting the mean")
    ax2.legend(fontsize=7.5)
    ax2.grid(alpha=0.3)

    fig.suptitle("Few-shot label efficiency (step 5)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path
