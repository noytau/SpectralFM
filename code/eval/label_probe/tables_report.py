"""
Generate the data-only report: titles, tables and figures, no prose.

Everything is read from a finished run's `label_probe_results.json` and the
PNGs beside it, so this regenerates rather than being transcribed by hand.

  python -m eval.label_probe.tables_report <run_dir> [-o out.html]
"""
from __future__ import annotations

import argparse
import base64
import json
import os

from . import readouts as ro

# ── measured outside the main run (see the baseline diagnostic) ───────────
BASELINE_DIAGNOSTICS = [
    ("Condition number of the raw design matrix", "9.9 × 10⁸"),
    ("Directions holding 99% of the variance", "4 of 245"),
    ("Directions holding 99.9% of the variance", "7 of 245"),
    ("Directions holding 99.99% of the variance", "10 of 245"),
]
BASELINE_WRONG = [
    ("RidgeCV, unnormalized", "0.409", "0.001"),
    ("OLS, float32", "0.404", "0.001"),
    ("OLS, float64", "0.825", "0.001"),
]
BASELINE_CANDIDATES = [
    ("z-scored + RidgeCV", "0.7627", "0.0005", ""),
    ("OLS, float64", "0.8245", "0.0006", ""),
    ("z-scored + PLS-64 (fold-internal)", "0.8246", "0.0005", ""),
    ("whitened + RidgeCV", "0.8248", "0.0005", "adopted"),
]
WHITEN_BY_N = [
    ("10", "−0.234", "−0.023", "whitened"), ("20", "−0.145", "0.002", "whitened"),
    ("50", "0.126", "0.096", "z-scored"), ("100", "0.510", "0.234", "z-scored"),
    ("200", "0.609", "0.457", "z-scored"), ("500", "0.672", "0.699", "tie"),
    ("1,000", "0.697", "0.782", "whitened"), ("4,716", "0.762", "0.825", "whitened"),
]
LEAK_DEMO = [
    ("PLS fitted on all rows → CV'd ridge", "0.794", "+0.001"),
    ("PLS fitted inside each fold", "0.824", "−0.072"),
]

FIGURES = [
    ("depth_profile.png", "R² by pipeline block · 1 component, full pool · error bars ±1 split SD"),
    ("recipe_search.png",
     "Pooling schemes and probes on the winning block · 1 component, full pool"),
    ("crossover_panel.png", "Honest per-rung crossover: absolute R² and the embedding − raw gap, "
                            "best recipe chosen per label budget on a disjoint selection split"),
    ("probe_comparison.png", "RidgeCV vs OLS at every normalizer · raw input, Projector, "
                             "layer 2, layer 12 (all plain mean-pooled) · 1 component, full pool"),
    ("true_vs_pred_grid.png", "True vs predicted · full pool · axes fixed to [−2, 2]"),
]


def _e(x):
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _table(headers, rows, aligns=None):
    aligns = aligns or ["left"] * len(headers)
    th = "".join(f'<th class="{"num" if a == "num" else ""}">{h}</th>'
                 for h, a in zip(headers, aligns))
    body = []
    for r in rows:
        cls = ' class="win-row"' if r and str(r[-1]) == "__win__" else ""
        cells = r[:-1] if r and str(r[-1]) == "__win__" else r
        tds = "".join(f'<td class="{"num" if a == "num" else ""}">{c}</td>'
                      for c, a in zip(cells, aligns))
        body.append(f"<tr{cls}>{tds}</tr>")
    return ('<div class="table-wrap"><table><thead><tr>' + th + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def _pm(v, e, dp=3):
    return f'{v:.{dp}f} <span class="pm">±{e:.{dp}f}</span>'


def build(run_dir: str) -> str:
    with open(os.path.join(run_dir, "label_probe_results.json")) as f:
        d = json.load(f)
    meta, es = d["meta"], d.get("embedding_search", {})
    n = meta.get("n")
    comps = sorted(d["by_n_comp"], key=lambda k: int(k))
    out = []

    def h2(t, num):
        out.append(f'<section><div class="sec-head"><span class="sec-num mono">{num}</span>'
                   f"<h2>{t}</h2></div>")

    def h3(t):
        out.append(f"<h3>{t}</h3>")

    # ── run facts ────────────────────────────────────────────────────────
    out.append('<h1>Label regression — data</h1>')
    out.append(_table(["", ""], [
        ["Checkpoint", f'<code>{_e(os.path.basename(meta.get("checkpoint", "")))}</code>'],
        ["Label", "<code>parameter_0</code> (labeled_data)"],
        ["Spectra", f"n = {n:,}"],
        ["Component counts", ", ".join(comps)],
        ["Search protocol", "3 repeated shuffled 5-fold splits"],
        ["± split SD", "across repeated splits"],
        ["± bootstrap SD", "across resampled spectra"],
        ["[IQR]", "across repeated training draws"],
    ]))

    # ── baseline ─────────────────────────────────────────────────────────
    h2("Setting the baseline", "01")
    h3("Raw-input geometry")
    out.append(_table(["Quantity", "Value"], [[a, b] for a, b in BASELINE_DIAGNOSTICS],
                      ["left", "num"]))
    h3("Two ways to get the same wrong answer — 1 component, n_train=4,716")
    out.append(_table(["Route", "R²"],
                      [[r, _pm(float(v), float(e))] for r, v, e in BASELINE_WRONG],
                      ["left", "num"]))
    h3("Baseline candidates — 1 component, n_train=4,716")
    out.append(_table(["Recipe", "R²", ""],
                      [[r, _pm(float(v), float(e), 4),
                        f'<span class="badge raw">{tag}</span>' if tag else "",
                        "__win__" if tag else ""]
                       for r, v, e, tag in BASELINE_CANDIDATES],
                      ["left", "num", "left"]))
    h3("Normalizer by label budget — raw input, 1 component")
    out.append(_table(["n_train", "z-scored", "whitened", "better"],
                      [[a, b, c, dd] for a, b, c, dd in WHITEN_BY_N],
                      ["num", "num", "num", "left"]))
    h3("Leak check on the supervised reducer — 1 component, n_train=4,716")
    out.append(_table(["Pipeline", "R², real labels", "R², shuffled labels"],
                      [[a, b, c] for a, b, c in LEAK_DEMO], ["left", "num", "num"]))
    out.append("</section>")

    # ── block scoreboard ─────────────────────────────────────────────────
    if es.get("stage_scores"):
        h2("Block scoreboard", "02")
        h3(f"1 component, n_train={n:,}, mean-pooled, RidgeCV")
        sc = es["stage_scores"]
        blocks = sorted({k.split("|")[0] for k in sc},
                        key=lambda s: -max(sc[f"{s}|mean|{nm}"]["r2"]
                                           for nm in ("standardize", "whiten")))
        rows = []
        for i, b in enumerate(blocks, 1):
            a, w = sc[f"{b}|mean|standardize"], sc[f"{b}|mean|whiten"]
            best_w = w["r2"] >= a["r2"]
            rows.append([str(i), ro.stage_display_name(b),
                         _pm(a["r2"], a["repeat_sd"]) if not best_w
                         else _pm(a["r2"], a["repeat_sd"]),
                         f'<strong>{_pm(w["r2"], w["repeat_sd"])}</strong>' if best_w
                         else _pm(w["r2"], w["repeat_sd"]),
                         "__win__" if i == 1 else ""])
        out.append(_table(["Rank", "Block", "Standardized", "Whitened"], rows,
                          ["num", "left", "num", "num"]))
        out.append("</section>")

        # ── recipe search ────────────────────────────────────────────────
        h2("Recipe search", "03")
        h3("Pooling scheme — winning block")
        out.append(_table(["Recipe", "R²"],
                          [[k.split("|", 1)[1].replace("|", " · "), _pm(v["r2"], v["repeat_sd"])]
                           for k, v in sorted(es["pooling_scores"].items(),
                                              key=lambda kv: -kv[1]["r2"])],
                          ["left", "num"]))
        if es.get("concat_scores"):
            h3("Block concatenation")
            out.append(_table(["Recipe", "R²"],
                              [[k.replace("|", " · "), _pm(v["r2"], v["repeat_sd"])]
                               for k, v in sorted(es["concat_scores"].items(),
                                                  key=lambda kv: -kv[1]["r2"])],
                              ["left", "num"]))
        h3("Probe — winning block and pooling")
        out.append(_table(["Probe", "R²"],
                          [[k, _pm(v["r2"], v["repeat_sd"])]
                           for k, v in sorted(es["probe_scores"].items(),
                                              key=lambda kv: -kv[1]["r2"])],
                          ["left", "num"]))
        out.append("</section>")

    # ── full pool ────────────────────────────────────────────────────────
    h2("Full pool", "04")
    h3(f"n_train = {n:,}")
    labels = list(d["by_n_comp"][comps[0]]["full_pool"])
    rows = []
    for c in comps:
        fp = d["by_n_comp"][c]["full_pool"]
        best = max(fp, key=lambda k: fp[k]["r2_mean"])
        cells = [c] + [(f'<strong>{_pm(fp[lb]["r2_mean"], fp[lb]["r2_bootstrap_sd"])}</strong>'
                        if lb == best else _pm(fp[lb]["r2_mean"], fp[lb]["r2_bootstrap_sd"]))
                       for lb in labels]
        rows.append(cells + ["__win__"])
    out.append(_table(["n-comp"] + [_e(lb) for lb in labels], rows,
                      ["num"] + ["num"] * len(labels)))
    out.append("</section>")

    # ── label efficiency, honest per-rung selection ─────────────────────────
    panel_path = os.path.join(run_dir, "recipe_panel.json")
    if os.path.exists(panel_path):
        with open(panel_path) as f:
            pd = json.load(f)
        h2("Label efficiency — honest per-rung selection", "05")
        for c in sorted(pd["by_n_comp"], key=int):
            sel = pd["by_n_comp"][c]["selected"]
            keys = sorted(sel["raw"], key=int)
            h3(f"{c} component{'s' if c != '1' else ''} — best recipe chosen per budget on a "
               "disjoint selection split, median R² [IQR]")
            rows = []
            for k in keys:
                r, e = sel["raw"][k], sel["embedding"][k]

                def cell(v, is_best):
                    txt = (f'{v["r2_median"]:.3f} <span class="pm">'
                           f'[{v["r2_p25"]:.2f}, {v["r2_p75"]:.2f}] · {_e(v["recipe"])}</span>')
                    return f"<strong>{txt}</strong>" if is_best else txt

                e_better = e["r2_median"] >= r["r2_median"]
                rows.append([f"{int(k):,}", cell(r, not e_better), cell(e, e_better)])
            out.append(_table(["n_train", "raw (best)", "embedding (best)"], rows,
                              ["num", "left", "left"]))
        out.append("</section>")

    # ── canary ───────────────────────────────────────────────────────────
    if d.get("canary"):
        h2("Shuffled-label canary", "06")
        rows = []
        for c in comps:
            for lab, v in d["canary"][c].items():
                rows.append([c, _e(lab), f'{v["real_r2"]:.3f}', f'{v["shuffled_r2"]:+.3f}',
                             "pass" if v["passed"] else "FAIL"])
        out.append(_table(["n-comp", "Recipe", "R² real", "R² shuffled", ""], rows,
                          ["num", "left", "num", "num", "left"]))
        out.append("</section>")

    # ── figures ──────────────────────────────────────────────────────────
    h2("Figures", "07")
    for fn, cap in FIGURES:
        path = os.path.join(run_dir, fn)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        out.append(f'<figure class="fig"><img src="data:image/png;base64,{b64}" alt="{_e(cap)}">'
                   f"<figcaption><code>{fn}</code> — {cap}</figcaption></figure>")
    out.append("</section>")
    return "\n".join(out)


STYLE = """<title>SpectralFM — Label Regression Data</title>
<style>
  :root{--bg:#0a0d12;--panel:#11151c;--ink:#dfe4ec;--dim:#8791a3;--faint:#545e6e;
        --line:#232a38;--accent:#4fd1c5;--accent2:#f0a860;--code:#0d1117;
        --shadow:0 1px 2px rgba(0,0,0,.4);}
  :root[data-theme="light"]{--bg:#f6f7f9;--panel:#fff;--ink:#1b212b;--dim:#5b6472;
        --faint:#909aab;--line:#e2e6ec;--accent:#0e8f86;--accent2:#b5691f;--code:#f1f3f6;
        --shadow:0 1px 3px rgba(20,25,35,.08);}
  @media (prefers-color-scheme: light){:root:not([data-theme="dark"]){
        --bg:#f6f7f9;--panel:#fff;--ink:#1b212b;--dim:#5b6472;--faint:#909aab;
        --line:#e2e6ec;--accent:#0e8f86;--accent2:#b5691f;--code:#f1f3f6;
        --shadow:0 1px 3px rgba(20,25,35,.08);}}
  *{box-sizing:border-box}
  html{color-scheme:dark light}
  :root[data-theme="light"]{color-scheme:light}
  :root[data-theme="dark"]{color-scheme:dark}
  body{margin:0;background:var(--bg);color:var(--ink);font-size:15px;line-height:1.5;
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
       -webkit-font-smoothing:antialiased;padding:2.5rem 1rem 5rem;}
  .wrap{max-width:64rem;margin:0 auto;}
  .mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
  h1{font-family:ui-monospace,"SF Mono",monospace;font-size:1.7rem;font-weight:600;
     margin:0 0 1.4rem;letter-spacing:-.01em;}
  section{margin:2.6rem 0 0}
  .sec-head{display:flex;align-items:baseline;gap:.6rem;margin:0 0 .9rem;
     border-bottom:1px solid var(--line);padding-bottom:.5rem;}
  .sec-num{font-family:ui-monospace,monospace;font-size:.8rem;color:var(--faint)}
  h2{font-family:ui-monospace,"SF Mono",monospace;font-size:1.15rem;font-weight:600;margin:0}
  h3{font-size:.9rem;font-weight:650;margin:1.5rem 0 .5rem;color:var(--dim);
     font-family:ui-monospace,monospace;letter-spacing:.01em;}
  code{font-family:ui-monospace,Menlo,monospace;background:var(--code);
     border:1px solid var(--line);border-radius:4px;padding:.08em .35em;font-size:.85em;
     color:var(--accent);}
  .table-wrap{overflow-x:auto;margin:.5rem 0 1rem;border:1px solid var(--line);
     border-radius:8px;}
  table{width:100%;border-collapse:collapse;font-size:.85rem}
  th,td{text-align:left;padding:.45rem .7rem;border-bottom:1px solid var(--line);
     white-space:nowrap;vertical-align:top;}
  th{font-family:ui-monospace,monospace;font-size:.68rem;letter-spacing:.05em;
     text-transform:uppercase;color:var(--faint);background:var(--panel);}
  td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
  tr:last-child td{border-bottom:none}
  tbody tr:hover{background:var(--panel)}
  tr.win-row td{background:color-mix(in srgb,var(--accent) 7%,transparent)}
  .pm{font-size:.85em;color:var(--dim);font-weight:400}
  .badge{display:inline-block;font-family:ui-monospace,monospace;font-size:.63rem;
     letter-spacing:.03em;text-transform:uppercase;padding:.08rem .38rem;border-radius:4px;
     color:var(--accent2);background:color-mix(in srgb,var(--accent2) 16%,transparent);}
  .fig{margin:1rem 0 1.6rem;background:var(--panel);border:1px solid var(--line);
     border-radius:10px;overflow:hidden;box-shadow:var(--shadow);}
  .fig img{display:block;width:100%;height:auto;background:#fff}
  .fig figcaption{padding:.6rem .9rem;font-size:.78rem;color:var(--dim);
     border-top:1px solid var(--line);white-space:normal;}
</style>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("docs", "html", "label-regression-data.html")
    html = STYLE + '<div class="wrap">\n' + build(a.run_dir) + "\n</div>"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
