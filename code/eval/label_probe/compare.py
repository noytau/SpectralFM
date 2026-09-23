"""
Line up several label-probe runs side by side -- different backbones on the
same labels, or the same backbone across different label sets (eval.runner's
multi-label-set mode); the row label disambiguates either case.

Each run directory is produced by `study.run_study` and is self-identifying:
`label_probe_results.json`'s `meta.backbone` is the model's class name,
auto-derived at extraction time (see readouts.py) -- never something a
caller has to name by hand, so this tool works on any set of runs without a
naming convention to agree on up front.

  python -m eval.label_probe.compare <run_dir> <run_dir> ... [-o out.html]

Reuses tables_report.py's table/style helpers rather than a second copy.
"""
from __future__ import annotations

import argparse
import json
import os

from .tables_report import STYLE, _e, _pm, _table


def _backbone_label(meta: dict) -> str:
    """`meta.backbone` if this run has it; older runs (from before that
    field existed) fall back to the checkpoint's own filename rather than
    showing a bare '?'."""
    if meta.get("backbone"):
        return meta["backbone"]
    ckpt = os.path.basename(meta.get("checkpoint", "") or "")
    return ckpt or "(unknown backbone)"


def _row_labels(runs: list) -> dict:
    """One label per run, keyed by run_dir. Backbone name by default; when
    two runs share a backbone (comparing label sets on one backbone, not
    backbones on one label set) each gets its run directory's name appended
    so rows stay distinguishable either way."""
    labels = [_backbone_label(r["results"]["meta"]) for r in runs]
    if len(set(labels)) < len(labels):
        labels = [f'{lab} ({os.path.basename(r["run_dir"].rstrip("/"))})'
                  for lab, r in zip(labels, runs)]
    return {r["run_dir"]: lab for r, lab in zip(runs, labels)}


def _load(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "label_probe_results.json")) as f:
        results = json.load(f)
    panel = None
    panel_path = os.path.join(run_dir, "recipe_panel.json")
    if os.path.exists(panel_path):
        with open(panel_path) as f:
            panel = json.load(f)
    return {"run_dir": run_dir, "results": results, "panel": panel}


def _crossing_summary(panel: dict, n_comp: str) -> str:
    """Reads the crossing off the panel's per-draw data directly, rather
    than re-deriving it -- keeps this tool from silently drifting out of
    sync with whatever crossing definition panel_plots.py uses. "Crosses"
    means the best of the named embedding layers overtakes raw input's own
    best recipe at that label budget."""
    if panel is None or n_comp not in panel.get("by_n_comp", {}):
        return "—"
    from . import panel_plots as pp

    d = panel["by_n_comp"][n_comp]
    ns_r, med_r, *_ = pp._selected_series(d["selected"])
    layer_meds = [pp._selected_series(curve)[1] for curve in d["layer_curves"].values()]
    best_layer_med = [max(vals) for vals in zip(*layer_meds)]
    gaps = [e - r for e, r in zip(best_layer_med, med_r)]
    cross = pp._crossing_n(ns_r, gaps)
    return f"n≈{cross:,.0f}" if cross is not None else "never crosses"


def build(run_dirs: list, title: str = "Run comparison") -> str:
    runs = [_load(d) for d in run_dirs]
    row_label = _row_labels(runs)
    out = [f'<h1>{_e(title)}</h1>']

    id_rows = []
    for r in runs:
        m = r["results"]["meta"]
        id_rows.append([_e(row_label[r["run_dir"]]),
                        f'<code>{_e(os.path.basename(m.get("checkpoint", "")))}</code>',
                        f'n={m.get("n", "?"):,}' if isinstance(m.get("n"), int) else "?",
                        _e(os.path.basename(r["run_dir"].rstrip("/")))])
    out.append(_table(["Run", "Checkpoint", "Labeled spectra", "Run directory"],
                      id_rows))

    comps = sorted({c for r in runs for c in r["results"]["by_n_comp"]}, key=int)
    for n_comp in comps:
        out.append(f'<h3>{n_comp} component{"s" if n_comp != "1" else ""} — full pool</h3>')
        rows = []
        for r in runs:
            by_comp = r["results"]["by_n_comp"].get(n_comp)
            label = row_label[r["run_dir"]]
            if by_comp is None:
                rows.append([_e(label), "—", "—", "—", "—"])
                continue
            fp = by_comp["full_pool"]
            raws = [v for k, v in fp.items() if k.startswith("raw input")]
            raw = max(raws, key=lambda v: v["r2_mean"]) if raws else None
            embs = [v for k, v in fp.items() if k.startswith("embedding")]
            emb = max(embs, key=lambda v: v["r2_mean"]) if embs else None
            raw_r2 = _pm(raw["r2_mean"], raw["r2_bootstrap_sd"]) if raw else "—"
            emb_r2 = _pm(emb["r2_mean"], emb["r2_bootstrap_sd"]) if emb else "—"
            gap = f'{emb["r2_mean"] - raw["r2_mean"]:+.3f}' if raw and emb else "—"
            crossing = _crossing_summary(r["panel"], n_comp)
            rows.append([_e(label), raw_r2, emb_r2, gap, crossing])
        out.append(_table(["Run", "Raw (best)", "Embedding (best)",
                           "Gap (emb − raw)", "Crossing"], rows))

    # One figure: full-pool embedding R² per run, 1 component (the headline
    # number every run has, so every run can always appear).
    try:
        import base64
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .plots import _INK, _INK_2, _PALETTE, _style_axes

        names, vals, errs = [], [], []
        for r in runs:
            fp = r["results"]["by_n_comp"].get("1", {}).get("full_pool", {})
            embs = [v for k, v in fp.items() if k.startswith("embedding")]
            emb = max(embs, key=lambda v: v["r2_mean"]) if embs else None
            if emb:
                names.append(row_label[r["run_dir"]])
                vals.append(emb["r2_mean"])
                errs.append(emb["r2_bootstrap_sd"])
        if len(names) >= 2:
            fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 4))
            x = range(len(names))
            ax.bar(x, vals, yerr=errs, color=_PALETTE[0], width=0.6,
                  error_kw=dict(elinewidth=1.2, capsize=3, ecolor=_INK_2), zorder=3)
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.3f}", xy=(xi, v), xytext=(0, 4),
                           textcoords="offset points", ha="center", fontsize=9,
                           color=_INK)
            _style_axes(ax, grid_axis="y")
            ax.set_xticks(list(x))
            ax.set_xticklabels(names, fontsize=9)
            ax.set_ylabel("R² (embedding, best recipe, full pool)", fontsize=9,
                          color=_INK_2)
            ax.set_title("1 component — across runs", fontsize=11,
                        color=_INK, fontweight="600", loc="left")
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format="png", dpi=150, facecolor="#ffffff",
                       bbox_inches="tight")
            plt.close(fig)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            out.append(f'<figure class="fig"><img src="data:image/png;base64,{b64}" '
                       f'alt="Embedding R² per run, 1 component, full pool">'
                       f'<figcaption>full-pool embedding R², 1 component, one bar per run'
                       f'</figcaption></figure>')
    except Exception as exc:  # pragma: no cover - a missing figure must never fail the table
        out.append(f'<p style="color:#b00">figure skipped: {_e(exc)}</p>')

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+",
                    help="two or more run directories, each from study.run_study")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("docs", "html", "label-regression-compare.html")
    html = STYLE.replace("Label Regression Data", "Backbone Comparison") + \
        '<div class="wrap">\n' + build(a.run_dirs) + "\n</div>"
    os.makedirs(os.path.dirname(out), exist_ok=True) if os.path.dirname(out) else None
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
