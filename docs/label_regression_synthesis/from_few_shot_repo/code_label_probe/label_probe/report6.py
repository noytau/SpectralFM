"""
Step-6 reporting. The report template is deliberately verdict-driven: it
states WIN/PARTIAL/NO from the pre-registered rule and never editorializes
past what was measured (see test_report_does_not_claim_a_win_it_did_not_measure).
"""
from __future__ import annotations

import os

import numpy as np

DIAG_KEYS = ("participation_ratio", "effective_rank", "ridgecv_full_r2",
             "whitened_topk_r2_5", "whitened_topk_r2_10",
             "whitened_topk_r2_20", "whitened_topk_r2_50")


def _diag_value(d: dict, key: str):
    if key.startswith("whitened_topk_r2_"):
        sub = d["whitened_topk_r2"]
        k_int = int(key.rsplit("_", 1)[1])
        # JSON round-trips int dict keys to strings (e.g. loaded from
        # step6_screen.json), while in-memory diagnostics use real ints --
        # accept either so the lookup doesn't silently vanish via KeyError.
        if k_int in sub:
            return sub[k_int]
        return sub[str(k_int)]
    return d[key]


def diagnostic_correlation(screen: dict) -> dict:
    """
    Spearman rho between each cheap diagnostic and the readout's measured
    few-shot score, across every screened readout.

    This is the study's secondary claim: if whitened_topk_r2 predicts
    few-shot R2, readout quality can be screened without running the few-shot
    grid at all -- a result that would transfer to other checkpoints.
    """
    from scipy.stats import spearmanr

    keys = sorted(set(screen["readouts"]) & set(screen["diagnostics"]))
    scores = np.array([screen["readouts"][k]["best_r2"] for k in keys])
    out = {}
    for dk in DIAG_KEYS:
        try:
            vals = np.array([_diag_value(screen["diagnostics"][k], dk) for k in keys])
        except KeyError as e:
            msg = f"missing from diagnostics ({e})"
            print(f"[step6] diagnostic_correlation: `{dk}` unavailable -- "
                  f"{msg}", flush=True)
            out[dk] = {"rho": None, "p": None, "n": 0,
                       "status": f"unavailable: missing key ({e})"}
            continue
        ok = np.isfinite(vals) & np.isfinite(scores)
        if ok.sum() < 3:
            print(f"[step6] diagnostic_correlation: `{dk}` unavailable -- "
                  f"only {int(ok.sum())} usable point(s) (<3)", flush=True)
            out[dk] = {"rho": None, "p": None, "n": int(ok.sum()),
                       "status": f"unavailable: n<3 (n={int(ok.sum())})"}
            continue
        res = spearmanr(vals[ok], scores[ok])
        out[dk] = {"rho": float(res.statistic), "p": float(res.pvalue),
                   "n": int(ok.sum()), "status": "ok"}
    return out


def write_step6_report(screen: dict, confirm: dict, verdict_dict: dict,
                       output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "step6_REPORT.md")
    corr = diagnostic_correlation(screen)

    L = []
    L.append("# Step 6 — Readout & geometry sweep for few-shot label efficiency\n")
    L.append(f"**Verdict: {verdict_dict['verdict']}** "
             f"({verdict_dict['n_cells_won']}/{verdict_dict['n_cells']} cells won "
             "under the rule pre-registered in "
             "`docs/superpowers/specs/2026-09-09-few-shot-readout-geometry-design.md` §2).\n")
    L.append(f"Screened {len(screen['readouts'])} readouts on `eval_a` "
             f"({screen['n_draws']} draws) to pick a finalist set (3 embedding "
             "readouts, 2 raw readouts per cell -- an asymmetry that if "
             "anything favours the embedding). The **per-cell result** table "
             "below is an argmax over those finalists x probes, computed on "
             f"the disjoint **`eval_b`** ({confirm['n_draws']} draws), so the "
             "finalist set never saw the data it is scored on; the argmax "
             "itself is symmetric across both arms and run identically for "
             "embedding and raw. The later sections on stage/pooling/"
             "normalizer choice and the diagnostic-correlation table are "
             "`eval_a` **screen** numbers, not `eval_b` confirm numbers.\n")

    L.append("\n## Per-cell result (eval_b, paired draws)\n")
    L.append("| n_comp | n_train | embedding readout | probe | emb R² | "
             "raw readout | probe | raw R² | ΔR² | boot P(Δ>0) | emb frac>0 | "
             "won | n_pairs | n_dropped |")
    L.append("|---:|---:|---|---|---:|---|---|---:|---:|---:|---:|:--:|---:|---:|")
    any_dropped = False
    for key in sorted(confirm["cells"], key=lambda k: (int(k.split("|")[0]),
                                                       int(k.split("|")[1]))):
        n_comp, n_train = key.split("|")
        c, vc = confirm["cells"][key], verdict_dict["cells"][key]
        n_dropped = vc.get("n_dropped", 0)
        any_dropped = any_dropped or n_dropped > 0
        L.append(
            f"| {n_comp} | {n_train} | `{c['emb']['readout']}` | "
            f"`{c['emb']['probe']}` | {c['emb']['r2_median']:+.3f} | "
            f"`{c['raw']['readout']}` | `{c['raw']['probe']}` | "
            f"{c['raw']['r2_median']:+.3f} | {vc['delta_median']:+.3f} | "
            f"{vc['boot_frac_positive']:.2f} | "
            f"{vc['emb_frac_positive_r2']:.2f} | "
            f"{'yes' if vc['won'] else 'no'} | "
            f"{vc.get('n_pairs', '—')} | {n_dropped} |")
    if any_dropped:
        L.append("\n*n_dropped counts paired draws excluded from the delta "
                 "above because the embedding or raw probe failed on that "
                 "draw (a `-inf` score, see `n_failed_draws` in the detail "
                 "JSON); a cell with n_dropped > 0 lost that many of its "
                 "paired comparisons.*\n")

    L.append("\n## Which hypothesis moved the number (eval_a screen numbers)\n")
    for n_comp, stages in sorted(screen.get("finalist_stages", {}).items()):
        L.append(f"- **{n_comp}-comp** — best stages by screen score: "
                 f"{', '.join(map(str, stages))}.")
    best_by_nrm = {}
    for k, v in screen["readouts"].items():
        if v["family"] == "emb":
            best_by_nrm[v["normalizer"]] = max(
                best_by_nrm.get(v["normalizer"], -np.inf), v["best_r2"])
    if best_by_nrm:
        L.append("- **Normalizer (H-C)** — best screen R² per normalizer: "
                 + ", ".join(f"`{k}` {v:+.3f}" for k, v in
                             sorted(best_by_nrm.items(), key=lambda t: -t[1])) + ".")
    best_by_pool = {}
    for v in screen["readouts"].values():
        if v["family"] == "emb" and v["pooling"] != "-":
            best_by_pool[v["pooling"]] = max(
                best_by_pool.get(v["pooling"], -np.inf), v["best_r2"])
    if len(best_by_pool) > 1:
        L.append("- **Pooling (H-A)** — best screen R² per pooling: "
                 + ", ".join(f"`{k}` {v:+.3f}" for k, v in
                             sorted(best_by_pool.items(), key=lambda t: -t[1])) + ".")

    L.append("\n## Do the cheap diagnostics predict few-shot performance? "
             "(eval_a screen numbers)\n")
    L.append("| diagnostic | Spearman ρ vs screen R² | p | n readouts |")
    L.append("|---|---:|---:|---:|")
    for k, v in sorted(corr.items(),
                       key=lambda t: -abs(t[1]["rho"]) if t[1]["rho"] is not None
                       else 1.0):
        if v["rho"] is None:
            L.append(f"| `{k}` | unavailable (n={v['n']}) | — | {v['n']} |")
        else:
            L.append(f"| `{k}` | {v['rho']:+.3f} | {v['p']:.2g} | {v['n']} |")

    L.append("\n## Scope\n")
    L.append("- One checkpoint (Feb-25 SSL), components (0,1,2), frozen backbone, "
             "random label draws.")
    L.append("- Layer and pooling have no raw-input analogue; the raw family gets "
             "the normalizer, reduction and full probe panel instead. That "
             "asymmetry favours the embedding.")
    L.append("- A NO verdict bounds this readout family on this backbone; it does "
             "not prove no readout exists.")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[step6] wrote {path}", flush=True)
    return path


def plot_screen(screen: dict, path: str, n_comp: int) -> str:
    """Screen R² by stage, one line per normalizer -- the H-B x H-C picture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [v for v in screen["readouts"].values()
            if v["n_comp"] == n_comp and v["family"] == "emb"
            and v["pooling"] == "mean"]
    stages = [s for s in ("fe", "extract_features")
              + tuple(f"layer{i}" for i in range(13))
              if any(r["stage"] == s for r in rows)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for nrm in sorted({r["normalizer"] for r in rows}):
        ys = [next((r["best_r2"] for r in rows
                    if r["stage"] == s and r["normalizer"] == nrm), np.nan)
              for s in stages]
        ax.plot(range(len(stages)), ys, marker="o", label=nrm)
    raw_best = max((v["best_r2"] for v in screen["readouts"].values()
                    if v["n_comp"] == n_comp and v["family"] == "raw"),
                   default=None)
    if raw_best is not None:
        ax.axhline(raw_best, ls="--", color="k",
                   label=f"best raw input ({raw_best:+.3f})")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha="right")
    ax.set_ylabel("best screen R² (eval_a)")
    ax.set_title(f"Step 6 screen — {n_comp}-comp, mean pooling")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confirm(confirm: dict, verdict_dict: dict, path: str) -> str:
    """Paired embedding-vs-raw medians per cell, on eval_b."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(confirm["cells"], key=lambda k: (int(k.split("|")[0]),
                                                   int(k.split("|")[1])))
    x = np.arange(len(keys))
    emb = [confirm["cells"][k]["emb"]["r2_median"] for k in keys]
    raw = [confirm["cells"][k]["raw"]["r2_median"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, emb, 0.4, label="embedding (best readout)")
    ax.bar(x + 0.2, raw, 0.4, label="raw input (best readout)")
    for i, k in enumerate(keys):
        if verdict_dict["cells"][k]["won"]:
            ax.text(i, max(emb[i], raw[i]) + 0.01, "✓", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k.split('|')[0]}c\nn={k.split('|')[1]}" for k in keys])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("median held-out R² (eval_b)")
    ax.set_title(f"Step 6 confirmation — verdict: {verdict_dict['verdict']} "
                 f"({verdict_dict['n_cells_won']}/{verdict_dict['n_cells']} cells)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
