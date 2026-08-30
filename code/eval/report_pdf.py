"""
PDF build of the reconstruction report, laid out for an e-ink reader.

Differs from the HTML build in four ways, none of them cosmetic on e-paper: greyscale
figures with heads separated by line style, marker and hatch as well as grey value
(`recon_plots.set_style("eink")`); explanations set as real type rather than baked into
the bitmap; a page sized to the panel; one figure per page.

Needs pdflatex and uses only geometry, graphicx, longtable and array — the eval host's
TeX install lacks booktabs, xcolor, caption and microtype, and hyperref and fontenc both
fail there on missing dependencies. Non-ASCII is mapped to TeX macros instead.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

# A PDF page the same physical size as the panel renders 1:1, so the presets are the
# panels' actual dimensions, not their diagonals.
#
#   Onyx BOOX 10.3in (Note Air, Tab Ultra): 1404x1872 px @ 226 ppi = 6.2 x 8.3 in
#   Onyx BOOX 13.3in (Tab X, Max Lumi):     1650x2200 px @ 207 ppi = 8.0 x 10.6 in
#   Onyx BOOX 7.8in  (Nova Air):            1404x1872 px @ 300 ppi = 4.7 x 6.2 in
PAGE_PRESETS = {
    "onyx13": (8.0, 10.6),
    "onyx10": (6.2, 8.3),
    "onyx8": (4.7, 6.2),
}
DEFAULT_PAGES = ("onyx13", "onyx10")

PAGE_W_IN = 8.0
PAGE_H_IN = 10.6
MARGIN_IN = 0.42

# Above this height a figure and its explanation share a page; below it the figure needs
# the page to itself.
COMBINE_MIN_H_IN = 9.5


def parse_pages(spec: str) -> list:
    """
    Parse a page spec into [(label, w_in, h_in), ...]. Accepts preset names and explicit
    WxH, comma-separated: "onyx13,onyx10" or "8x10.6".
    """
    out = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token in PAGE_PRESETS:
            w, h = PAGE_PRESETS[token]
            out.append((token, w, h))
            continue
        try:
            w, h = (float(v) for v in token.lower().split("x"))
        except ValueError:
            print("[ReportPDF] bad page spec %r — expected a preset (%s) or WxH"
                  % (token, ", ".join(PAGE_PRESETS)))
            continue
        out.append(("%gx%g" % (w, h), w, h))
    return out


def available() -> bool:
    return shutil.which("pdflatex") is not None


# ── TeX escaping ──────────────────────────────────────────────────────────────

_TEX_ESCAPES = [
    ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
    ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    # ~ always means "approximately" here; \textasciitilde renders as a raised macron.
    ("~", r"$\sim$"), ("^", r"\textasciicircum{}"),
]

# Characters the report text uses that the default pdflatex font encoding cannot set.
_UNICODE_MAP = {
    "\u2014": "---", "\u2013": "--", "\u2212": "-", "\u2192": r"$\rightarrow$",
    "\u2191": r"$\uparrow$", "\u2193": r"$\downarrow$", "\u00d7": r"$\times$",
    "\u2248": r"$\approx$", "\u2264": r"$\leq$", "\u2265": r"$\geq$",
    "\u00b2": r"\textsuperscript{2}", "\u2080": "0", "\u2081": "1", "\u201c": "``",
    "\u201d": "''", "\u2018": "`", "\u2019": "'", "\u2026": r"\ldots{}",
    "\u00b1": r"$\pm$", "\u2261": r"$\equiv$", "\u00b7": r"$\cdot$",
    "\u2500": "-", "\u00a0": " ",
}


_WARNED_CHARS = set()


def _warn_unmapped(chars) -> None:
    new = [c for c in chars if c not in _WARNED_CHARS]
    if not new:
        return
    _WARNED_CHARS.update(new)
    print("[ReportPDF] unmapped non-ASCII replaced with '?': %s — add them to "
          "_UNICODE_MAP" % ", ".join("%r (U+%04X)" % (c, ord(c)) for c in new))


def tex_escape(text: str) -> str:
    """
    Plain text to TeX, before any markup is re-introduced.

    Specials are escaped before the unicode is mapped: the replacements are themselves
    TeX macros, so mapping first would get them escaped into literals.
    """
    for ch, rep in _TEX_ESCAPES:
        text = text.replace(ch, rep)
    for uni, rep in _UNICODE_MAP.items():
        text = text.replace(uni, rep)
    # OT1 without fontenc: anything unmapped would abort the run, so drop it and warn.
    if not text.isascii():
        leftover = sorted({c for c in text if not c.isascii()})
        _warn_unmapped(leftover)
        text = "".join(c if c.isascii() else "?" for c in text)
    return text


def _breakable_code(s: str) -> str:
    """
    Let a long identifier break inside \texttt. TeX will not hyphenate typewriter text,
    so offer zero-width breaks at camelCase humps and underscores, plus a fallback every
    14 atoms. \allowbreak avoids a hyphen, which would read as part of the identifier.
    """
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", r"\\allowbreak{}", s)
    s = s.replace(r"\_", r"\_\allowbreak{}")

    # Count TeX atoms, not characters: chunking mid-escape leaves a bare backslash and
    # an unescaped _, which aborts the build.
    atoms = re.findall(r"\\[a-zA-Z]+(?:\{\})?|\\.|.", s, flags=re.S)
    out, run = [], 0
    for atom in atoms:
        if atom == r"\allowbreak{}":
            out.append(atom)
            run = 0
            continue
        if run >= 14:
            out.append(r"\allowbreak{}")
            run = 0
        out.append(atom)
        run += 1
    return "".join(out)


def tex_inline(text: str) -> str:
    """
    Escape, then turn the report's `code`, **bold** and *italic* markup back into TeX.

    Code spans are stashed before the emphasis patterns run: `single_channel_*` has a
    lone asterisk that would otherwise pair with the next one in the paragraph.
    """
    out = tex_escape(text)

    spans = []
    def stash(m):
        spans.append(m.group(1))
        return "\x00%d\x00" % (len(spans) - 1)
    out = re.sub(r"`([^`]+)`", stash, out)

    out = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", out)
    out = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\\emph{\1}", out)

    for i, span in enumerate(spans):
        out = out.replace("\x00%d\x00" % i, r"\texttt{%s}" % _breakable_code(span))
    return out


# ── Document skeleton ─────────────────────────────────────────────────────────

_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[paperwidth=%(pw).2fin,paperheight=%(ph).2fin,margin=%(m).2fin,%%
            includefoot,footskip=0.22in]{geometry}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{array}
\usepackage{tabularx}

%% An e-paper panel renders hairlines poorly and has a limited contrast range, so rules
%% are thicker than print defaults and body text is pure black.
\setlength{\arrayrulewidth}{0.5pt}
\renewcommand{\arraystretch}{1.18}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\setlength{\emergencystretch}{3em}
\sloppy
\pagestyle{plain}

%% Section headings sized for a small page.
\makeatletter
\renewcommand\section{\@startsection{section}{1}{0pt}{1.6ex plus 1ex minus .2ex}%%
  {1.0ex plus .2ex}{\normalfont\Large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{0pt}{1.4ex plus 1ex minus .2ex}%%
  {0.8ex plus .2ex}{\normalfont\large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{0pt}%%
  {1.2ex plus 1ex minus .2ex}{0.7ex plus .2ex}{\normalfont\normalsize\bfseries}}
\makeatother

\begin{document}
"""


def _figure_page(img_path: str, caption: str, explain: list, title: str) -> str:
    """
    The figure alone on one page, its explanation on the next.

    Sharing a page means capping the image height to leave room for four paragraphs of
    text, and a near-page-shaped figure then cannot use the full width either - on a
    7.8in panel that lands somewhere between cramped and useless. Giving the figure the
    whole page costs one page turn and roughly doubles its linear size, which is the
    better trade on a device where the constraint is the screen, not the paper.

    The caption stays with the figure - it is one line and it is what the figure means.

    On a page tall enough (COMBINE_MIN_H_IN, i.e. a 13.3in panel) the explanation joins
    the figure instead: there is room for both, and on e-paper a page turn costs a
    full-screen refresh, so not needing one is worth more than the extra figure height.
    """
    combined = PAGE_H_IN >= COMBINE_MIN_H_IN and explain
    # Sharing the page means leaving room for the text below the image.
    img_h = "0.58" if combined else "0.90"

    out = [r"\subsubsection*{%s}" % tex_inline(title)]
    out.append(r"\begin{center}")
    out.append(r"\includegraphics[width=\textwidth,height=%s\textheight,"
               r"keepaspectratio]{%s}" % (img_h, img_path))
    out.append(r"\end{center}")
    out.append(r"\vspace{-0.4em}{\small\itshape %s\par}" % tex_inline(caption))
    if explain:
        if not combined:
            out.append(r"\clearpage")
            out.append(r"{\small\textbf{%s}\par}" % tex_inline(title))
        out.append(r"\vspace{0.3em}")
        out.append(r"\begin{itemize}\setlength{\itemsep}{0.45em}"
                   r"\setlength{\leftmargini}{1.1em}")
        for label, text in explain:
            out.append(r"\item[] \textbf{%s.} %s" % (tex_inline(label), tex_inline(text)))
        out.append(r"\end{itemize}")
    out.append(r"\clearpage")
    return "\n".join(out)


def _table(md_lines: list, caption: str = "") -> str:
    """
    A markdown pipe table as a longtable, sized to the page.

    The summary tables are twelve columns wide, which will not fit a 6-inch page at body
    size, so they are set smaller and column-wrapped. longtable so a long one breaks
    across pages instead of overflowing one.
    """
    rows = [ln for ln in md_lines if ln.strip().startswith("|")]
    if len(rows) < 2:
        return ""
    def cells(ln):
        return [c.strip() for c in ln.strip().strip("|").split("|")]
    head = cells(rows[0])
    body = [cells(ln) for ln in rows[2:]]
    ncol = len(head)

    # Two kinds of table appear here and a narrow page needs them handled differently.
    # Numeric summaries are many columns of short cells: fixed-width, right-aligned, set
    # small. Prose tables (the decoder-head table) have long cells that must wrap or the
    # last column runs off the paper - tabularx spreads them across exactly \textwidth.
    longest = max((len(c) for row in [head] + body for c in row), default=0)
    wrapping = longest > 22 and ncol <= 6

    out = []
    if caption:
        out.append(r"\textbf{%s}\par" % tex_inline(caption))
    size = r"\tiny" if ncol > 9 else (r"\scriptsize" if ncol > 6 else r"\small")

    if wrapping:
        col = ">{\\raggedright\\arraybackslash}X"
        out.append(r"{%s" % size)
        out.append(r"\begin{tabularx}{\textwidth}{@{}%s@{}}" % (col * ncol))
        out.append(r"\hline")
        out.append(" & ".join(r"\textbf{%s}" % tex_inline(h) for h in head) + r" \\")
        out.append(r"\hline")
        for row in body:
            out.append(" & ".join(tex_inline(c) for c in row) + r" \\")
        out.append(r"\hline")
        out.append(r"\end{tabularx}")
        out.append(r"}")
        return "\n".join(out)

    # First column left-aligned (labels), the rest right-aligned (numbers).
    spec = "l" + "r" * (ncol - 1) if ncol > 1 else "l"
    out.append(r"{%s" % size)
    out.append(r"\begin{longtable}{%s}" % spec)
    out.append(r"\hline")
    out.append(" & ".join(r"\textbf{%s}" % tex_inline(h) for h in head) + r" \\")
    out.append(r"\hline\endfirsthead")
    out.append(r"\hline")
    out.append(" & ".join(r"\textbf{%s}" % tex_inline(h) for h in head) + r" \\")
    out.append(r"\hline\endhead")
    for row in body:
        out.append(" & ".join(tex_inline(c) for c in row) + r" \\")
    out.append(r"\hline")
    out.append(r"\end{longtable}")
    out.append(r"}")
    return "\n".join(out)


# ── Assembly ──────────────────────────────────────────────────────────────────

def _prose(md_lines: list) -> str:
    """
    Report prose (already markdown) as TeX paragraphs.

    Tables are handled separately; a blockquote becomes an indented paragraph, and a
    numbered or bulleted item becomes an item. Everything else is a paragraph.
    """
    out, table_buf = [], []
    for ln in md_lines + [""]:
        s = ln.rstrip()
        if s.strip().startswith("|"):
            table_buf.append(s)
            continue
        if table_buf:
            out.append(_table(table_buf))
            table_buf = []
        if not s.strip():
            continue
        if s.startswith(">"):
            out.append(r"\begin{quote}\small %s\end{quote}"
                       % tex_inline(s.lstrip("> ")))
        elif re.match(r"^\d+\.\s", s.strip()):
            out.append(tex_inline(re.sub(r"^\d+\.\s*", "", s.strip())))
        elif s.strip().startswith("- "):
            out.append(tex_inline(s.strip()[2:]))
        else:
            out.append(tex_inline(s.strip()))
    return "\n\n".join(p for p in out if p)


def build_tex(results: dict, figures_by_eval: dict, summary_figs: list,
              run_dir: str, config=None) -> str:
    """
    Assemble the TeX source.

    Built from the same helpers as the HTML and markdown reports so the three cannot
    drift apart; only the layout differs.
    """
    from .report import (_RECON_INTRO, _RECON_TABLE_LEGEND, _explain_lines,
                         _recon_combined_table, _recon_dataset_heading, _recon_keys,
                         _recon_summary_table, _split_eval_key, _recon_fig_caption)

    keys = _recon_keys(results)
    if not keys:
        return ""
    first = results[keys[0]]
    model = first.get("_model_name") or "reconstruction"

    doc = [_PREAMBLE % {"pw": PAGE_W_IN, "ph": PAGE_H_IN, "m": MARGIN_IN}]

    # Title page.
    doc.append(r"\begin{center}")
    doc.append(r"{\LARGE\bfseries Signal Reconstruction (3AE)\par}")
    doc.append(r"\vspace{0.6em}{\large %s\par}" % tex_inline(model))
    doc.append(r"\vspace{0.4em}{\small normalize=%s\par}" % tex_inline(str(first.get("normalize"))))
    doc.append(r"\end{center}")
    doc.append(r"\vspace{0.8em}")
    doc.append(_prose(_RECON_INTRO))
    doc.append(r"\clearpage")

    # 1 — per dataset.
    doc.append(r"\section*{1. Per dataset}")
    doc.append(tex_inline("Headline numbers per head, over every sample drawn."))
    doc.append(r"{\small %s\par}" % tex_inline(_RECON_TABLE_LEGEND))
    for key in keys:
        r = results[key]
        alias = _split_eval_key(key)[1]
        heading = _recon_dataset_heading(r, alias)
        doc.append(r"\subsection*{%s}" % tex_inline(heading))
        if r.get("skipped"):
            doc.append(tex_inline("Skipped - %s." % r.get("error", "n/a")))
            continue
        doc.append(_table(_recon_summary_table(r)))
        spec = r.get("spectrum_df")
        if spec is not None and len(spec):
            doc.append(tex_inline(
                "Per-spectrum view: %d spectra, aggregating the components of each "
                "(dataset, spec) key." % len(spec)))
    doc.append(r"\clearpage")

    # 3 — across datasets.
    if summary_figs:
        doc.append(r"\section*{2. Across datasets}")
        doc.append(tex_inline(
            "Every dataset in one view, with single-component and multi-component blocks "
            "kept visually separate."))
        combined = _recon_combined_table(results)
        if combined:
            doc.append(_table(combined, caption="Summary table — all datasets"))
        doc.append(r"\clearpage")
        for f in summary_figs:
            title, explain = _explain_lines(f[1])
            doc.append(_figure_page(f[1], _recon_fig_caption(f[1]), explain, title))

    # Closing read of the numbers.
    try:
        from . import findings
        obs, nxt = findings.observations(results), findings.next_steps(results, config)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("[ReportPDF] findings section FAILED: %s: %s" % (type(e).__name__, e))
        obs, nxt = [], []
    for heading, items in (("What this run shows", obs), ("What to do next", nxt)):
        if not items:
            continue
        doc.append(r"\section*{%s}" % tex_inline(heading))
        doc.append(r"\begin{itemize}\setlength{\itemsep}{0.5em}")
        for item in items:
            # A finding's nested "  - " lines become sub-items.
            head, *subs = item.split("\n  - ")
            doc.append(r"\item %s" % tex_inline(head))
            if subs:
                doc.append(r"\begin{itemize}")
                for s in subs:
                    doc.append(r"\item %s" % tex_inline(s))
                doc.append(r"\end{itemize}")
        doc.append(r"\end{itemize}")
    doc.append(r"\clearpage")

    # Appendix.
    if config is not None:
        doc.append(r"\section*{Appendix — configuration}")
        doc.append(tex_inline("Every parameter this run was launched with."))
        rows = ["| Parameter | Value |", "|---|---|"]
        rows += ["| `%s` | `%s` |" % (k, v) for k, v in vars(config).items()]
        doc.append(_table(rows))

    doc.append(r"\end{document}")
    return "\n\n".join(doc)


def build_pdfs(results: dict, figures_by_eval: dict, summary_figs: list,
               run_dir: str, config=None, pages: str = None) -> list:
    """
    Build one PDF per requested page size. Returns the paths produced.

    Several sizes cost only an extra pdflatex pass each - they share the figures - so a
    run can hand out a copy for each device rather than making anyone guess which page
    matches their panel.
    """
    global PAGE_W_IN, PAGE_H_IN
    specs = parse_pages(pages or ",".join(DEFAULT_PAGES))
    if not specs:
        return []
    out = []
    for label, w, h in specs:
        PAGE_W_IN, PAGE_H_IN = w, h
        path = build_pdf(results, figures_by_eval, summary_figs, run_dir, config,
                         name="eval_report_%s" % label)
        if path:
            out.append(path)
    return out


def build_pdf(results: dict, figures_by_eval: dict, summary_figs: list,
              run_dir: str, config=None, name: str = "eval_report_eink") -> str:
    """
    Write <run_dir>/<name>.pdf. Returns the path, or "" if it could not be produced.

    pdflatex runs twice (longtable needs a second pass to settle column widths) with
    -interaction=nonstopmode, so a single bad character degrades one line rather than
    failing the build. Auxiliary files are cleaned up; the .tex is kept, since it is the
    thing to look at when the output is wrong.
    """
    if not available():
        print("[ReportPDF] pdflatex not found — skipping PDF build")
        return ""
    tex = build_tex(results, figures_by_eval, summary_figs, run_dir, config)
    if not tex:
        return ""

    tex_path = os.path.join(run_dir, name + ".tex")
    with open(tex_path, "w") as f:
        f.write(tex)

    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-jobname", name, tex_path],
            cwd=run_dir, capture_output=True, text=True)
    pdf_path = os.path.join(run_dir, name + ".pdf")
    if not os.path.isfile(pdf_path):
        # Report from the .log, not the captured stdout: pdflatex's stdout can come back
        # empty, and the log is the authoritative record anyway. Keep it on disk too.
        log_path = os.path.join(run_dir, name + ".log")
        lines = []
        if os.path.isfile(log_path):
            with open(log_path, errors="replace") as f:
                lines = f.read().splitlines()
        errs = [ln for ln in lines if ln.startswith("!")]
        print("[ReportPDF] pdflatex failed. Errors:\n  %s\n  Log tail:\n%s\n  "
              "Full log: %s"
              % ("\n  ".join(errs[:8]) or "(none found)",
                 "\n".join("    " + ln for ln in lines[-15:]), log_path))
        return ""

    for ext in (".aux", ".log", ".out", ".toc"):
        stale = os.path.join(run_dir, name + ext)
        if os.path.isfile(stale):
            os.remove(stale)
    size_mb = os.path.getsize(pdf_path) / 1e6
    # Counting /Type /Page in the raw bytes reads 0 here, because pdfTeX puts the page
    # objects in a compressed object stream. Ask pdfinfo when it is available.
    pages = ""
    if shutil.which("pdfinfo"):
        info = subprocess.run(["pdfinfo", pdf_path], capture_output=True, text=True)
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                pages = ", %s pages" % line.split(":", 1)[1].strip()
                break
    print("[ReportPDF] %s  (%.1f MB, %gx%gin page%s%s)"
          % (pdf_path, size_mb, PAGE_W_IN, PAGE_H_IN, pages,
             ", figure+text on one page" if PAGE_H_IN >= COMBINE_MIN_H_IN else ""))
    return pdf_path
