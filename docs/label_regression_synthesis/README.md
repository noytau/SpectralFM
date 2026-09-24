# Archive: label-regression / few-shot-probe source material

Read [`../LABEL_REGRESSION_FINDINGS.md`](../LABEL_REGRESSION_FINDINGS.md) first — that
file is the entry point and states the corrected, final verdict. Everything
in this folder is source material it was built from, rescued before two
sibling repos (`SpectralFM-few-shot`, `SpectralFM-label-regression`) were
deleted. Treat this folder as read-only provenance, not a place to develop.

- `from_few_shot_repo/` — committed material from `SpectralFM-few-shot`:
  its `code/eval/label_probe/` package (readout/pooling sweep, crossover
  estimator), step 1–7 reports and results, design specs.
- `from_label_regression_repo/` — committed + working-tree material from
  `SpectralFM-label-regression`: its (more complete) `code/eval/label_probe/`
  package, step reports/results, the polished HTML report, `plan.md`.
- `rescued_uncommitted/` — files that existed **only** in a Claude Code
  session's `/tmp` scratchpad in the `SpectralFM-few-shot` repo, never
  committed to git, and would have been permanently lost on deletion.
  Most important: `label_regression_findings_v5.html` / `body_v5.html` (the
  single best final synthesis of the few-shot study) and `step7_v3_real.png`
  / `fixed_step1_grid.png` (the corrected crossover and true-vs-predicted
  plots referenced in the findings doc).
- `fewshot_study_analysis.md`, `labelregression_study_analysis.md` — full
  file-by-file analyses of each source repo (methodology, what's trustworthy,
  what's superseded, exact file paths/line numbers), produced while rescuing
  this material. More detail than the top-level findings doc; consult these
  for provenance questions.

Extracted-feature caches (`step6/bank/`, `_cache/` — several GB of
reproducible intermediate tensors, not results) were deliberately **not**
copied.
