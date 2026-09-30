"""
One self-contained HTML report comparing several backbones against raw input
on every label set, from finished label_probe + nested runs.

  python -m eval.label_probe.backbone_report <eval_outputs_dir> -o report.html \
      [--findings findings.json] [--metrics_out metrics.json]

<eval_outputs_dir> holds the run directories named in backbone_metrics.py
(one per backbone per label set, one pooled run, one labeled_data run), each
already processed by `python -m eval.label_probe.nested`. The backbone list
lives at the top of backbone_metrics.py.

--findings is an optional JSON of written observations keyed by section
(bottom_line, fullpool, ranking, rawnorm, merged, merged_ld, depth, depth_ld,
efficiency, efficiency_ld, caveats; plus title, subtitle, dataset_notes,
footer). Without it the report renders numbers only.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os

from .backbone_metrics import LD, MERGED, collect
from .nested import MIN_N, NORMALIZERS, PROBES

MIX_COLOR = {"single-channel": "#4a7aa8", "multi-channel (pos)": "#4f9d4a", "sampled": "#c05a3a"}
STAGE_ORDER = ["fe", "extract_features"] + [f"layer{i}" for i in range(13)]
STAGE_SHORT = {"fe": "FE", "extract_features": "FE+LN", "layer0": "Proj",
               **{f"layer{i}": f"L{i}" for i in range(1, 13)}}
NORM_LABEL = {"none": "none", "standardize": "z-scored", "whiten": "whitened",
              "whiten8": "whiten 8", "whiten32": "whiten 32", "whiten128": "whiten 128"}
NORM_SHORT = {"none": "none", "standardize": "z", "whiten": "whiten",
              "whiten8": "w8", "whiten32": "w32", "whiten128": "w128"}
PROBE_LABEL = {"ridgecv": "RidgeCV", "ols": "OLS"}
PROBE_SHORT = {"ridgecv": "Ridge", "ols": "OLS"}
PROBE_COLOR = {"ridgecv": "var(--s1)", "ols": "var(--s2)"}
STRONG = ' class="strong"'


def e(x):
    return html.escape(str(x))


def pm(v, sd):
    return f'{v:.3f}<span class="pm"> ±{sd:.3f}</span>'


def recipe_short(norm, probe):
    return f"{NORM_SHORT[norm]}+{PROBE_SHORT[probe]}"


def tip_attr(rows):
    """rows: list of (label, value) or a str heading -> escaped data-tip."""
    parts = []
    for r in rows:
        parts.append(f"<b>{r}</b>" if isinstance(r, str) else
                     f"<div class='tr'><span>{r[0]}</span><span class='mono'>{r[1]}</span></div>")
    return html.escape("<div class='tt'>" + "".join(parts) + "</div>", quote=True)


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t, out = math.ceil(lo / step) * step, []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


class Report:
    def __init__(self, M, F):
        self.M, self.F = M, F
        self.BB = M["backbones"]
        self.TAGS = [b["tag"] for b in self.BB]
        self.NAME = {b["tag"]: b["name"] for b in self.BB}
        self.SHORT = {b["tag"]: b["short"] for b in self.BB}
        self.SERIES = {t: f"var(--s{i + 1})" for i, t in enumerate(self.TAGS)}
        self.R = M["runs"]

    # ── data access ─────────────────────────────────────────────────────────
    def get(self, tag, ds):
        return self.R.get(tag, {}).get(ds)

    def first(self, ds):
        return next((self.get(t, ds) for t in self.TAGS if self.get(t, ds)), None)

    def has(self, ds):
        return self.first(ds) is not None

    def probed_sets(self):
        return [ds for ds in self.M["label_sets"] if self.has(ds)]

    def compared(self):
        return self.probed_sets() + [ds for ds in (MERGED, LD) if self.has(ds)]

    def n(self, ds):
        return self.M["sizes"].get(ds)

    def dname(self, ds):
        return {MERGED: f"all {len(self.M['label_sets'])} merged"}.get(ds, ds)

    def block(self, arm):
        return STAGE_SHORT.get(arm, arm)

    # ── small pieces ────────────────────────────────────────────────────────
    def swatch(self, tag):
        return f'<span class="sw" style="background:{self.SERIES[tag]}"></span>'

    def section(self, title, sub=None, small=False):
        s = ' style="margin-top:32px;"' if small else ""
        h = f'<h2 style="font-size:16px;">{title}</h2>' if small else f"<h2>{title}</h2>"
        return (f'<div class="section-heading"{s}>{h}'
                + (f'<div class="section-sub">{sub}</div>' if sub else "") + "</div>")

    def para(self, text):
        return f'<p style="margin:12px 0 0; max-width:95ch; font-size:14px; color:var(--ink-soft);">{text}</p>'

    def verdict(self, key, warn=False, label=None):
        if key not in self.F:
            return ""
        body = self.F[key] if isinstance(self.F[key], list) else [self.F[key]]
        lab = label or ("Caveats" if warn else "Finding")
        return (f'<div class="verdict{" warn" if warn else ""}"><div class="vlabel">{lab}</div>'
                + "".join(f"<p>{p}</p>" for p in body) + "</div>")

    def legend(self, extra=None):
        items = [f'<span class="lg">{self.swatch(t)}{e(self.SHORT[t])}</span>' for t in self.TAGS]
        return f'<div class="legend">{"".join(([extra] if extra else []) + items)}</div>'

    def choice_sub(self, ch, with_block=True):
        top = ch["top"]
        txt = (f"{self.block(top[0])} · {recipe_short(top[1], top[2])}" if with_block
               else recipe_short(top[0], top[1]))
        return f'{e(txt)} <span class="pm">({ch["count"]}/{ch["of"]})</span>'

    # ── charts ──────────────────────────────────────────────────────────────
    def chart_lines(self, xs_labels, series, ref=None, log_x=False, xs_values=None,
                    y_label="R²", aria="", x_title=""):
        """series: {tag | 'raw': [y or None per x]}; ref: (y, sd, label) band."""
        allv = [v for s in series.values() for v in s if v is not None]
        if ref:
            allv += [ref[0] - ref[1], ref[0] + ref[1]]
        lo, hi = min(allv), max(allv)
        pad = (hi - lo) * 0.08 or 0.05
        lo, hi = lo - pad, hi + pad
        W, H, left, right, top, bottom = 980, 400, 56, 150, 18, 54
        n = len(xs_labels)
        if log_x:
            lv = [math.log10(v) for v in xs_values]
            xpos = lambda i: left + (lv[i] - lv[0]) / (lv[-1] - lv[0] or 1) * (W - left - right)
        else:
            xpos = lambda i: left + i / (n - 1) * (W - left - right)
        y = lambda v: top + (hi - v) / (hi - lo) * (H - top - bottom)
        svg = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{e(aria)}">']
        for tv in nice_ticks(lo, hi, 6):
            svg.append(f'<line x1="{left}" x2="{W-right}" y1="{y(tv):.1f}" y2="{y(tv):.1f}" class="grid"/>')
            svg.append(f'<text x="{left-8}" y="{y(tv)+4:.1f}" class="tick" text-anchor="end">{tv:.2f}</text>')
        if lo < 0 < hi:
            svg.append(f'<line x1="{left}" x2="{W-right}" y1="{y(0):.1f}" y2="{y(0):.1f}" class="zero"/>')
        for i, lab in enumerate(xs_labels):
            svg.append(f'<text x="{xpos(i):.1f}" y="{H-bottom+18}" class="tick" text-anchor="middle">{e(lab)}</text>')
        if x_title:
            svg.append(f'<text x="{(left + W - right)/2:.1f}" y="{H-8}" class="ann" text-anchor="middle">{e(x_title)}</text>')
        mid = top + (H - top - bottom) / 2
        svg.append(f'<text x="14" y="{mid:.1f}" class="ann" text-anchor="middle" '
                   f'transform="rotate(-90 14 {mid:.1f})">{e(y_label)}</text>')
        if ref:
            ry, rsd, rlab = ref
            svg.append(f'<rect x="{left}" width="{W-left-right}" y="{y(ry+rsd):.1f}" '
                       f'height="{y(ry-rsd)-y(ry+rsd):.1f}" class="band"/>')
            svg.append(f'<line x1="{left}" x2="{W-right}" y1="{y(ry):.1f}" y2="{y(ry):.1f}" class="refline"/>')
            svg.append(f'<text x="{W-right+8}" y="{y(ry)+4:.1f}" class="endlab">{e(rlab)}</text>')
        names = {"raw": "raw input"}
        ends = []
        for key, vals in series.items():
            col = "var(--ink)" if key.startswith("raw") else self.SERIES[key]
            pts = [(xpos(i), y(v)) for i, v in enumerate(vals) if v is not None]
            if len(pts) > 1:
                dash = ' stroke-dasharray="5 3"' if key == "raw" else ""
                svg.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" '
                           f'fill="none" stroke="{col}" stroke-width="2"{dash}/>')
            for a, b in pts:
                svg.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{col}" class="mk"/>')
            if pts:
                ends.append([pts[-1][1], key])
        ends.sort()
        for i in range(1, len(ends)):
            if ends[i][0] - ends[i - 1][0] < 13:
                ends[i][0] = ends[i - 1][0] + 13
        for yy, key in ends:
            svg.append(f'<text x="{W-right+8}" y="{yy+4:.1f}" class="endlab">'
                       f'{e(names.get(key) or self.SHORT[key])}</text>')
        tipname = {"raw": "raw input"}
        for i, lab in enumerate(xs_labels):
            x0 = (xpos(i - 1) + xpos(i)) / 2 if i else left - 10
            x1 = (xpos(i) + xpos(i + 1)) / 2 if i < n - 1 else W - right + 4
            rows = [str(lab)]
            for k in sorted(series, key=lambda k: -(series[k][i] if series[k][i] is not None else -9)):
                if series[k][i] is not None:
                    rows.append((tipname.get(k) or self.NAME[k], f"{series[k][i]:.3f}"))
            svg.append(f'<rect x="{x0:.1f}" y="{top}" width="{x1-x0:.1f}" height="{H-top-bottom}" '
                       f'class="col" tabindex="0" data-x="{xpos(i):.1f}" data-tip="{tip_attr(rows)}"/>')
        svg.append(f'<line class="xhair" x1="0" x2="0" y1="{top}" y2="{H-bottom}" style="display:none"/></svg>')
        extra = None
        if "raw" in series:
            extra = '<span class="lg"><span class="sw dash"></span>raw input</span>'
        elif ref:
            extra = f'<span class="lg"><span class="sw dash"></span>{e(ref[2])}</span>'
        return self.legend(extra) + "".join(svg)

    # ── sections ────────────────────────────────────────────────────────────
    def method_box(self):
        return self.para(
            "<b>How every number is scored.</b> Each spectrum is its 245-point component 0. "
            "The embedding is a backbone block's output, mean-pooled over time. Every arm chooses "
            f"its recipe from the same {len(NORMALIZERS) * len(PROBES)}: "
            f"{len(NORMALIZERS)} feature normalizers ({', '.join(NORM_LABEL[n] for n in NORMALIZERS)}) "
            "× {RidgeCV, OLS}. For a backbone it also chooses the block, from the feature extractor to the "
            "last Transformer layer. The choice is made by <b>nested cross-validation</b>. The outer loop is "
            "2× repeated 5-fold. Inside each outer training fold, a 5-fold inner CV picks the block and recipe, "
            "the winner is refit on the whole training fold, and it predicts the held-out fold. Normalizers are "
            "fit on training rows only. No score is ever chosen on the rows that grade it, and raw input and "
            "the embeddings choose the same way. "
            "<b>± is a bootstrap SD</b> over spectra. Every arm on a label set shares the same folds, so a "
            "<b>difference between two arms is paired</b>: both are rescored on the same resampled spectra, "
            "which gives a much tighter SD for the difference than combining two independent ones.")

    def backbone_table(self):
        rows = []
        for b in self.BB:
            bar = "".join(f'<span style="display:inline-block;height:8px;width:{max(p, 0.6):.2f}%;'
                          f'background:{MIX_COLOR[k]}"></span>' for k, p in b["mix"])
            mix = ", ".join(f"{k} {p:g}%" for k, p in b["mix"])
            rows.append(
                f'<tr><td>{self.swatch(b["tag"])}<b>{e(b["short"])}</b>'
                f'<div class="sub mono">{e(b["name"])}</div><div class="sub mono">{e(b["ckpt"])}</div></td>'
                f'<td class="wrap">{e(b["stage"])}</td>'
                f'<td><div class="mix" title="{e(mix)}">{bar}</div>'
                f'<div class="mixlab">{e(b["data"])} · n = {b["n_pretrain"]:,}</div></td>'
                f'<td class="mono">{e(b["projector"])}</td><td class="mono">{e(b["updates"])}</td></tr>')
        legend = "".join(f'<span class="lg"><span class="sw" style="background:{c}"></span>{e(k)}</span>'
                         for k, c in MIX_COLOR.items())
        return (f'<div class="legend" style="margin-top:14px">{legend}</div>'
                '<div class="table-wrap"><table><thead><tr><th>Backbone · checkpoint under checkpoints/</th>'
                '<th>Training stage</th><th>Pretraining data</th><th>Projector</th><th>Updates</th>'
                '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")

    def dataset_table(self):
        notes = self.F.get("dataset_notes", {})
        rows = []
        for ds in self.M["label_sets"] + [MERGED, LD]:
            n = self.n(ds)
            if n is None:
                continue
            ok = self.has(ds)
            status = ('<span class="tag ok">probed</span>' if ok else
                      f'<span class="tag bad">skipped · n &lt; {MIN_N}</span>')
            rows.append(f'<tr{STRONG if ds in (MERGED, LD) else ""}><td class="mono">{e(self.dname(ds))}</td>'
                        f'<td class="mono">{n:,}</td><td>{status}</td>'
                        f'<td class="wrap">{notes.get(ds, "")}</td></tr>')
        return ('<div class="table-wrap"><table><thead><tr><th>Label set</th><th>Labeled spectra</th>'
                '<th>Status</th><th>Note</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")

    def full_pool_table(self):
        head = ('<tr><th rowspan="2">Label set</th><th rowspan="2">raw input</th>'
                f'<th colspan="{len(self.TAGS)}">embedding: nested-CV R² ± bootstrap SD · most-chosen block · recipe</th>'
                '<th rowspan="2">best backbone − raw<br><span style="font-weight:400;text-transform:none">'
                '(paired SD)</span></th></tr><tr>'
                + "".join(f"<th>{self.swatch(t)}{e(self.SHORT[t])}</th>" for t in self.TAGS) + "</tr>")
        body = []
        for ds in self.compared():
            rs = {t: self.get(t, ds) for t in self.TAGS}
            ref = self.first(ds)
            cells = {"raw": ref["raw"]["r2"], **{t: r["emb"]["r2"] for t, r in rs.items() if r}}
            best_k, worst_k = max(cells, key=cells.get), min(cells, key=cells.get)
            td = lambda k, c: (f'<td class="mono{" best" if k == best_k else ""}'
                               f'{" worst" if k == worst_k else ""}">{c}</td>')
            row = [f'<td class="mono">{e(self.dname(ds))}<div class="sub">n = {self.n(ds):,}</div></td>',
                   td("raw", pm(ref["raw"]["r2"], ref["raw"]["sd"])
                      + f'<div class="sub">{self.choice_sub(ref["raw"]["choice"], False)}</div>')]
            for t in self.TAGS:
                r = rs[t]
                row.append('<td class="mono">—</td>' if not r else
                           td(t, pm(r["emb"]["r2"], r["emb"]["sd"])
                              + f'<div class="sub">{self.choice_sub(r["emb"]["choice"])}</div>'))
            bt = max((t for t in self.TAGS if rs[t]), key=lambda t: rs[t]["emb"]["r2"])
            d = rs[bt]["emb_minus_raw"]
            row.append(f'<td class="mono {"pos" if d["delta"] > 0 else "neg"}">{d["delta"]:+.3f}'
                       f'<span class="pm"> ({d["delta"] / d["sd"]:+.1f} SD)</span>'
                       f'<div class="sub">{e(self.SHORT[bt])}</div></td>')
            body.append(f'<tr{STRONG if ds in (MERGED, LD) else ""}>' + "".join(row) + "</tr>")
        return ('<div class="table-wrap"><table><thead>' + head + "</thead><tbody>"
                + "".join(body) + "</tbody></table></div>")

    def ranking_table(self):
        cols = [ds for ds in self.compared() if self.M["ranking"].get(ds)]
        rank = {}
        for ds in cols:
            for i, t in enumerate(self.M["ranking"][ds]["order"]):
                rank[(t, ds)] = i + 1
        head = ("<tr><th>Backbone</th>" + "".join(
            f'<th>{e(self.dname(ds).replace("dataset", ""))}<div class="thn">n = {self.n(ds):,}</div></th>'
            for ds in cols) + "</tr>")
        body = []
        for t in self.TAGS:
            cells = []
            for ds in cols:
                k = rank.get((t, ds))
                if k is None:
                    cells.append('<td class="mono">—</td>')
                    continue
                cls = " best" if k == 1 else " worst" if k == len(self.TAGS) else ""
                cells.append(f'<td class="mono{cls}"><b>{k}</b><div class="sub">{self.get(t, ds)["emb"]["r2"]:.3f}</div></td>')
            body.append(f'<tr><td>{self.swatch(t)}<span class="mono">{e(self.SHORT[t])}</span></td>'
                        + "".join(cells) + "</tr>")
        gap = lambda ds: self.M["ranking"][ds]["gaps"][0]
        body.append('<tr class="muted"><td>1st − 2nd <span style="font-weight:400">(paired SD)</span></td>'
                    + "".join(f'<td class="mono">{gap(ds)["delta"]:+.3f}'
                              f'<div class="sub">{gap(ds)["delta"] / gap(ds)["sd"]:.1f} SD</div></td>'
                              for ds in cols) + "</tr>")
        return ('<div class="table-wrap"><table><thead>' + head + "</thead><tbody>"
                + "".join(body) + "</tbody></table></div>")

    def raw_normalizer_panel(self, ds):
        """Raw input at every recipe held fixed (no selection), same outer
        folds, normalizer fit on training rows. Shared 0-1 axis."""
        r = self.first(ds)
        g = r["raw_fixed"]
        W, H, left, right, top, bottom = 460, 250, 34, 8, 26, 34
        y = lambda v: top + (1 - v) * (H - top - bottom)
        gw = (W - left - right) / len(NORMALIZERS)
        bw = (gw - 14) / 2
        best = max(g, key=lambda k: g[k]["r2_mean"])
        svg = [f'<svg viewBox="0 0 {W} {H}" class="chart mini" role="img" '
               f'aria-label="Raw input R² by normalizer and probe, {e(self.dname(ds))}">']
        for tv in (0, 0.25, 0.5, 0.75, 1.0):
            svg.append(f'<line x1="{left}" x2="{W-right}" y1="{y(tv):.1f}" y2="{y(tv):.1f}" '
                       f'class="{"zero" if tv == 0 else "grid"}"/>')
            svg.append(f'<text x="{left-6}" y="{y(tv)+3.5:.1f}" class="tick" text-anchor="end">{tv:g}</text>')
        for i, norm in enumerate(NORMALIZERS):
            gx = left + i * gw + 7
            svg.append(f'<text x="{gx + gw/2 - 7:.1f}" y="{H-bottom+15}" class="tick" '
                       f'text-anchor="middle">{e(NORM_LABEL[norm])}</text>')
            for j, probe in enumerate(PROBES):
                key = f"{norm}+{probe}"
                v, sd = g[key]["r2_mean"], g[key]["bootstrap_sd"]
                x = gx + j * (bw + 2)
                tip = tip_attr([f"{self.dname(ds)} · raw input", ("normalizer", NORM_LABEL[norm]),
                                ("probe", PROBE_LABEL[probe]), ("R²", f"{v:.3f} ± {sd:.3f}")])
                if v <= 0:
                    svg.append(f'<text x="{x + bw/2:.1f}" y="{y(0)-4:.1f}" class="neg-mark" text-anchor="middle">&lt;0</text>')
                    svg.append(f'<rect x="{x:.1f}" y="{y(0.12):.1f}" width="{bw:.1f}" height="{y(0)-y(0.12):.1f}" '
                               f'class="hit" tabindex="0" data-tip="{tip}"/>')
                    continue
                rad = min(4, y(0) - y(v))
                svg.append(f'<path d="M{x:.1f},{y(0):.1f} V{y(v)+rad:.1f} Q{x:.1f},{y(v):.1f} {x+rad:.1f},{y(v):.1f} '
                           f'H{x+bw-rad:.1f} Q{x+bw:.1f},{y(v):.1f} {x+bw:.1f},{y(v)+rad:.1f} V{y(0):.1f} Z" '
                           f'fill="{PROBE_COLOR[probe]}"/>')
                lo, hi = max(v - sd, 0), min(v + sd, 1)
                svg.append(f'<line x1="{x+bw/2:.1f}" x2="{x+bw/2:.1f}" y1="{y(hi):.1f}" y2="{y(lo):.1f}" class="whisk"/>')
                svg.append(f'<text x="{x+bw/2:.1f}" y="{y(hi)-4:.1f}" class="barval{" best" if key == best else ""}" '
                           f'text-anchor="middle">{v:.2f}</text>')
                svg.append(f'<rect x="{x-1:.1f}" y="{y(hi)-14:.1f}" width="{bw+2:.1f}" height="{y(0)-y(hi)+14:.1f}" '
                           f'class="hit" tabindex="0" data-tip="{tip}"/>')
        svg.append("</svg>")
        bn, bp = best.split("+")
        ch = r["raw"]["choice"]
        return (f'<div class="mini-card"><div class="mini-head"><span class="mono"><b>{e(self.dname(ds))}</b></span>'
                f'<span class="mini-sub">n = {self.n(ds):,} · best fixed: {e(NORM_LABEL[bn])} + {e(PROBE_LABEL[bp])}, '
                f'{g[best]["r2_mean"]:.3f} · nested picks {e(recipe_short(*ch["top"]))} in {ch["count"]}/{ch["of"]} folds</span></div>'
                + "".join(svg) + "</div>")

    def raw_normalizer_section(self):
        legend = "".join(f'<span class="lg"><span class="sw" style="background:{PROBE_COLOR[p]}"></span>'
                         f'{PROBE_LABEL[p]}</span>' for p in PROBES)
        panels = "".join(self.raw_normalizer_panel(ds) for ds in self.compared())
        return (f'<div class="legend" style="margin-top:16px">{legend}</div><div class="mini-grid">{panels}</div>'
                '<div class="cap">Raw input only: it never touches a checkpoint, so it is the same under every '
                'backbone. Each bar is one recipe held fixed on the same outer folds as everything else, normalizer '
                'fit on the training rows, ±1 bootstrap SD. The header gives the recipe that nested CV picks most '
                'often. <span class="mono">whiten K</span> keeps the top K principal directions before equalising '
                'them; full whitening keeps every direction the training rows support, at most one per two rows. '
                'A negative R² gets no bar. Hover or focus a bar for its value.</div>')

    def pool_table(self, ds):
        raw = self.first(ds)["raw"]
        body = [f'<tr class="strong muted"><td>raw input <span style="font-weight:400">(checkpoint-independent)</span></td>'
                f'<td class="mono">{self.choice_sub(raw["choice"], False)}</td>'
                f'<td class="mono">{pm(raw["r2"], raw["sd"])}</td><td class="mono">—</td><td class="mono">—</td><td></td></tr>']
        for t in self.TAGS:
            r = self.get(t, ds)
            if not r:
                continue
            d = r["emb_minus_raw"]
            blocks = ", ".join(f"{self.block(b)} {k}" for b, k in r["emb"]["choice"]["block_counts"].items())
            can = f'{r["canary"]["passed"]}/{r["canary"]["n"]}' if r.get("canary") else "—"
            body.append(
                f'<tr><td>{self.swatch(t)}<span class="mono">{e(self.SHORT[t])}</span></td>'
                f'<td class="mono">{self.choice_sub(r["emb"]["choice"])}<div class="sub">blocks chosen: {e(blocks)}</div></td>'
                f'<td class="mono">{pm(r["emb"]["r2"], r["emb"]["sd"])}</td>'
                f'<td class="mono {"pos" if d["delta"] > 0 else "neg"}">{d["delta"]:+.3f}'
                f'<span class="pm"> ({d["delta"] / d["sd"]:+.1f} SD)</span></td>'
                f'<td class="mono">{d["p_a_better"]:.0%}</td><td class="mono">{can}</td></tr>')
        return ('<div class="table-wrap"><table><thead><tr><th>Arm</th>'
                '<th>most-chosen block · recipe (outer folds)</th><th>nested R²</th>'
                '<th>− raw (paired SD)</th><th>bootstrap P(embedding &gt; raw)</th>'
                '<th>shuffled-label<br>canary</th></tr></thead><tbody>' + "".join(body) + "</tbody></table></div>")

    def depth_chart(self, ds):
        series = {t: [self.get(t, ds)["blocks"].get(s, {}).get("r2") for s in STAGE_ORDER]
                  for t in self.TAGS if self.get(t, ds)}
        raw = self.first(ds)["raw"]
        return self.chart_lines([STAGE_SHORT[s] for s in STAGE_ORDER], series,
                                ref=(raw["r2"], raw["sd"], "raw input"),
                                aria=f"Nested-CV R² by pipeline block, {self.dname(ds)}, one line per backbone",
                                x_title="pipeline block (mean-pooled; recipe chosen by nested CV per block)")

    def readouts_table(self):
        """The embedding three ways against raw input: full block search, one
        block fixed in advance (each backbone's labeled_data peak), and the
        average of the top 3 blocks. Small sets: mean paired gap over the
        probed sets; pools: paired gap and its SD."""
        kinds = [("search", "full block search"), ("fixed", "fixed block"), ("top3", "top-3 average")]
        small = self.probed_sets()
        head = ('<tr><th rowspan="2">Backbone<div class="thn">fixed block</div></th>'
                + "".join(f'<th colspan="3">{e(lab)}</th>' for _, lab in kinds) + "</tr><tr>"
                + "".join(f'<th>small sets<div class="thn">mean of {len(small)}</div></th>'
                          '<th>merged</th><th>labeled_data</th>' for _ in kinds) + "</tr>")
        body = []
        for t in self.TAGS:
            fb = self.M["fixed_blocks"].get(t)
            cells = []
            for kind, _ in kinds:
                ds_ = [self.get(t, ds)["readouts"][kind]["delta"] for ds in small
                       if self.get(t, ds) and kind in (self.get(t, ds)["readouts"] or {})]
                ahead = sum(d >= 0 for d in ds_)
                cells.append(f'<td class="mono {"pos" if ds_ and sum(ds_) > 0 else "neg"}">'
                             f'{sum(ds_) / len(ds_):+.3f}<div class="sub">≥ raw {ahead}/{len(ds_)}</div></td>'
                             if ds_ else '<td class="mono">—</td>')
                for pool in (MERGED, LD):
                    r = self.get(t, pool)
                    v = (r["readouts"] or {}).get(kind) if r else None
                    if not v:
                        cells.append('<td class="mono">—</td>')
                        continue
                    mark = "†" if kind == "fixed" and pool == LD else ""
                    cells.append(f'<td class="mono {"pos" if v["delta"] > 0 else "neg"}">{v["delta"]:+.3f}{mark}'
                                 f'<div class="sub">{v["delta"] / v["sd"]:+.1f} SD · {v["r2"]:.3f}</div></td>')
            body.append(f'<tr><td>{self.swatch(t)}<span class="mono">{e(self.SHORT[t])}</span>'
                        f'<div class="sub">{e(self.block(fb))}</div></td>' + "".join(cells) + "</tr>")
        return ('<div class="table-wrap"><table><thead>' + head + "</thead><tbody>" + "".join(body)
                + "</tbody></table></div>"
                '<div class="cap">Every cell is the embedding minus raw input, paired on the same outer folds; under a pool '
                'cell, its paired SD and the embedding’s R². '
                '<b>Full block search</b>: block and recipe chosen by nested CV (the numbers in the table above). '
                '<b>Fixed block</b>: the block is fixed in advance to the backbone’s best single block on '
                '<span class="mono">labeled_data</span>; only the recipe is chosen, from the same 12 as raw input. '
                '† On <span class="mono">labeled_data</span> itself that choice is in-sample, so this cell is mildly '
                'optimistic. <b>Top-3 average</b>: inside each outer training fold, the 3 blocks with the best inner-CV '
                'scores, each at its own best recipe, refit and averaged.</div>')

    def efficiency(self, ds):
        runs = {t: self.get(t, ds)["ladder"] for t in self.TAGS if self.get(t, ds) and self.get(t, ds)["ladder"]}
        L0 = next(iter(runs.values()))
        rungs = [str(n) for n in L0["rungs"]]
        series = {"raw": [L0["raw"][k]["median"] for k in rungs],
                  **{t: [L["emb"][k]["median"] for k in rungs] for t, L in runs.items()}}
        chart = self.chart_lines([f"{int(k):,}" for k in rungs], series, log_x=True,
                                 xs_values=[int(k) for k in rungs], y_label="median held-out R²",
                                 aria=f"Label efficiency on {self.dname(ds)}: raw input vs each backbone's fixed block",
                                 x_title="labels used for training (log scale)")
        head = ('<tr><th>labels</th><th>raw input</th>'
                + "".join(f"<th>{self.swatch(t)}{e(self.SHORT[t])} · {e(self.block(runs[t]['block']))}"
                          f'<div class="thn">median · paired gap to raw</div></th>' for t in runs) + "</tr>")
        body = []
        for k in rungs:
            cells = {"raw": L0["raw"][k]["median"], **{t: L["emb"][k]["median"] for t, L in runs.items()}}
            best_k = max(cells, key=cells.get)
            row = [f'<td class="mono">{int(k):,}<div class="sub">{L0["draws"][k]} draws</div></td>',
                   f'<td class="mono{" best" if best_k == "raw" else ""}">{cells["raw"]:.3f}'
                   f'<div class="sub">{e(recipe_short(*L0["raw"][k]["recipe"].split("+")))}</div></td>']
            for t, L in runs.items():
                g = L["gap"][k]
                row.append(f'<td class="mono{" best" if best_k == t else ""}">{cells[t]:.3f}'
                           f'<span class="pm"> ({g["median"]:+.3f})</span>'
                           f'<div class="sub">{e(recipe_short(*L["emb"][k]["recipe"].split("+")))} · '
                           f'{g["frac_positive"]:.0%} ahead</div></td>')
            body.append("<tr>" + "".join(row) + "</tr>")
        body.append('<tr class="strong"><td colspan="2">where the embedding leads</td>'
                    + "".join(f'<td class="wrap">{e(L["crossing"])}</td>' for L in runs.values()) + "</tr>")
        table = ('<div class="table-wrap"><table><thead>' + head + "</thead><tbody>" + "".join(body)
                 + "</tbody></table></div>")
        return chart, table

    # ── page ────────────────────────────────────────────────────────────────
    def pool_block(self, ds, fkey):
        return [self.section(e(self.dname(ds)), f"n = {self.n(ds):,}", small=True),
                self.pool_table(ds), self.verdict(fkey)]

    def depth_block(self, ds, fkey):
        return [self.section(e(self.dname(ds)), small=True),
                '<div class="chart-card">' + self.depth_chart(ds) +
                '<div class="cap">Each point is one block on its own, its recipe chosen by nested CV like every other '
                'number here. Dashed band: raw input, ±1 bootstrap SD. Hover or focus a column for all values.</div></div>',
                self.verdict(fkey)]

    def eff_block(self, ds, fkey):
        chart, table = self.efficiency(ds)
        return [self.section(e(self.dname(ds)), f"n = {self.n(ds):,}", small=True),
                '<div class="chart-card">' + chart +
                '<div class="cap">For each outer fold (5-fold) and label budget, random subsets of that many labeled '
                'spectra are drawn from the training fold. On each subset, a 5-fold inner CV over those labels only '
                'picks the recipe (the same 12 for both arms); the winner is refit on the subset and scored on the '
                'whole held-out fold. Both arms use the same subsets, so the bracketed gap is paired; “% ahead” is '
                'the share of draws where the embedding beats raw input. Each backbone uses its fixed block from the '
                'table in section 1. Hover or focus a column for all values.</div></div>',
                table, self.verdict(fkey)]

    def render(self, css, js):
        F = self.F
        pools = [ds for ds in (MERGED, LD) if self.has(ds)]
        title = F.get("title", "Label regression: raw input vs. pretrained backbones")
        parts = [
            f"<title>{e(F.get('title_tag', 'Backbone Label Probe'))}</title>",
            "<style>" + css + "</style>", '<div class="page">',
            f"<h1>{title}</h1>",
            f'<div class="subtitle">{F.get("subtitle", "")}</div>',
            self.verdict("bottom_line", label="Bottom line"),
            self.section("Backbones", f"{len(self.TAGS)} pretrained checkpoints"),
            self.backbone_table(),
            self.section("Label sets and protocol", "1 component (comp 0) per spectrum · labels standardized within each set"),
            self.method_box(),
            self.dataset_table(),
            self.section("1. Raw input vs. embedding, per label set", "nested 2×5-fold CV · paired differences"),
            self.full_pool_table(),
            self.verdict("fullpool"),
            self.section("Backbones ranked against each other", "rank by nested embedding R²", small=True),
            self.ranking_table(),
            self.verdict("ranking"),
            self.section("Embedding readouts compared", "full block search vs. a block fixed in advance vs. the top-3 average", small=True),
            self.readouts_table(),
            self.verdict("readouts"),
            self.section("2. Raw input, recipe by recipe", "each of the 12 recipes held fixed · same outer folds"),
            self.raw_normalizer_section(),
            self.verdict("rawnorm"),
            self.section("3. The two large pools", "which block and recipe nested CV picks, and how consistently"),
            *[p for ds, k in zip(pools, ("merged", "merged_ld")) for p in self.pool_block(ds, k)],
            self.section("4. Where the label signal lives", "every pipeline block on its own"),
            *[p for ds, k in zip(pools, ("depth", "depth_ld")) for p in self.depth_block(ds, k)],
            self.section("5. Label efficiency", "recipe chosen from the labels available · median over draws"),
            *[p for ds, k in zip(pools, ("efficiency", "efficiency_ld"))
              if any(self.get(t, ds) and self.get(t, ds)["ladder"] for t in self.TAGS)
              for p in self.eff_block(ds, k)],
            self.verdict("caveats", warn=True),
            f'<div class="page-footer">{F.get("footer", "")}</div>',
            "</div>", '<div id="tip" class="tooltip" role="tooltip" hidden></div>',
            "<script>" + js + "</script>",
        ]
        return "\n".join(parts)


JS = r"""
(function () {
  const tip = document.getElementById('tip');
  function show(el, evt) {
    tip.innerHTML = el.getAttribute('data-tip');
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    let x = evt && evt.clientX != null ? evt.clientX : r.left + r.width / 2;
    let y = evt && evt.clientY != null ? evt.clientY : r.top;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    x = Math.min(Math.max(8, x + 14), window.innerWidth - w - 8);
    y = y - h - 12 < 8 ? y + 18 : y - h - 12;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
    const svg = el.ownerSVGElement, xh = svg && svg.querySelector('.xhair');
    if (xh && el.dataset.x) { xh.setAttribute('x1', el.dataset.x); xh.setAttribute('x2', el.dataset.x); xh.style.display = ''; }
  }
  function hide(el) {
    tip.hidden = true;
    const xh = el.ownerSVGElement && el.ownerSVGElement.querySelector('.xhair');
    if (xh) xh.style.display = 'none';
  }
  document.querySelectorAll('[data-tip]').forEach(el => {
    el.addEventListener('mousemove', ev => show(el, ev));
    el.addEventListener('mouseleave', () => hide(el));
    el.addEventListener('focus', () => show(el));
    el.addEventListener('blur', () => hide(el));
  });
})();
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("eval_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--findings", help="JSON of written observations (see module docstring)")
    ap.add_argument("--metrics_out", help="also write the collected numbers as JSON")
    a = ap.parse_args()
    M = collect(a.eval_dir)
    if a.metrics_out:
        with open(a.metrics_out, "w") as f:
            json.dump(M, f, indent=1)
    F = json.load(open(a.findings)) if a.findings else {}
    css = open(os.path.join(os.path.dirname(__file__), "backbone_report.css")).read()
    with open(a.out, "w") as f:
        f.write(Report(M, F).render(css, JS))
    print(f"[backbone_report] wrote {a.out}")


if __name__ == "__main__":
    main()
