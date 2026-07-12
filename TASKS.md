# SpectralFM — Remaining Tasks

Work through sections in order within each track. E2 is blocked by E1 and should run after E3.
T1's reporting step is blocked by E1 (don't trust eval numbers until E1 is verified).
Mark tasks done with a short note: what changed, how it was verified, where the outputs are.

## Evaluation pipeline

### E1. Debug the embedding source (BLOCKING — do first)
The embedding-based scores don't look right. Trace where embeddings are extracted for the
evals (which layer, which pooling, pre/post projection) and debug until verified — add
sanity checks (e.g. embed the same input twice → similarity 1.0; embed a shuffled/random
input → low similarity) and confirm the numbers behave as expected.
- [x] **E1a. Structured cosine similarity** — the struct_sim_* figures look especially
      weird on the embedding side. Verify the pipeline: correct embeddings in, correct
      normalization, correct axis ordering in the similarity matrix.
      **DONE (2026-07-12):** Pipeline verified correct — same layer/pooling as the original
      (`extract_features(x, mask=False)["x"].mean(1)` ≡ `last_hidden_state.mean(1)`); input
      panels match the original figure to ±0.005. Two real issues found and fixed:
      (1) `embedding_similarity.py` embedded MASKED inputs (15% zeros) while the input
      baseline used clean signals — now embeds clean data (`use_masked` opt-in);
      (2) trained checkpoints have collapsed/anisotropic embeddings (mean pairwise cos
      0.78–0.94, noise-vs-real 0.75) — added centered-cosine matrices + all-models
      comparison figure (port of `compare_checkpoints._plot_similarity_rows`).
      Verified by recreating the original "Input Space vs All Models" figure: the
      2026-01-07 early-SSL checkpoint reproduces its original rich structure
      (mean 0.463/std 0.295 vs original 0.452/0.288). Sanity checks: same-input-twice
      cos=1.0 at all stages; probe script preserved. Output:
      `code/eval_outputs/2026-07-12_17-53-03/struct_sim_all_models{,_centered}.png`.
- [ ] **E1b. Label regression** — the eval is missing the latest training changes that use
      2–3 channels. There were additional runs with better results that aren't reflected.
      Find those runs/checkpoints, make the label-regression eval support the multi-channel
      inputs, and include those runs.
- Acceptance: sanity checks pass, and the embedding scores are explainable (you can say
  WHY each number is what it is).

### E2. Pre- vs post-reconstruction checkpoint comparison (blocked by E1; run after E3)
Once E1 is fixed and verified, run the full comparison eval on exactly 2 checkpoints:
one from before any reconstruction training, one from after. Goal: analyze what
reconstruction training changed. Deliverable: the eval output dir + a short written
analysis of the differences per metric/figure.

### E3. Clean up the eval output directory structure
Currently all figures/CSVs from a run land flat in one timestamped directory
(40+ files mixing checkpoints and eval types). Restructure the output so each run dir
contains one subdirectory per model/checkpoint, and inside it one subdirectory per
eval method, e.g.:

    eval_outputs/2026-07-11_19-35-01/
    ├── run_info.md
    ├── eval_report.html / eval_report.md
    ├── comparison/                      # cross-checkpoint figures + CSVs
    │   ├── checkpoint_comparison.png
    │   └── noise_robustness_comparison.png
    ├── ckpt_lr0.0001_..._base_libri/
    │   ├── signal_completion/
    │   ├── noise_robustness/
    │   ├── similarity/
    │   ├── clustering/
    │   └── label_regression/
    └── ckpt_tr_lr0.001_..._backbone/
        └── (same structure)

- Filenames no longer need the full checkpoint name embedded once they live in a
  per-checkpoint subdir — shorten them accordingly.
- Cross-checkpoint comparison figures go in a shared `comparison/` subdir, not under
  any single model.
- Update eval_report.md / eval_report.html generation so all image links point to the
  new relative paths (the HTML embeds figures — verify it still renders self-contained).
- Acceptance: run the suite once and confirm no file lands in the run-dir root except
  the reports and run_info.md, and both reports display all figures correctly.

## Training pipeline

### T1. Improve FE — regularization (training already done; collect + report)
Training with regularization was already run. Find those runs, collect the checkpoints
and results, run the (verified) evaluation on them, and prepare an email-ready summary:
what regularization was tried, metrics vs. baseline, key figures attached/linked.
- Note: run the eval only after E1 is done, otherwise the numbers can't be trusted.

### T2. Improve projection layer
Current projection output looks noisy. Experiment with a deeper projection head:
MLP with 1–2 hidden layers, try different hidden dims, consider activation/normalization
choices. Train the variants and compare with the eval suite.

### T3. Evaluate transformer without masking (eval experiment, no training)
Find where masking is applied in the transformer forward path and run the evaluation with
masking disabled entirely. Expectation: better results. Report masked vs. unmasked
side by side; make the no-masking mode a proper CLI flag, not a code edit.
