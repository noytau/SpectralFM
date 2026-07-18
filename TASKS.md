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
- [x] **E1b. Label regression** — the eval is missing the latest training changes that use
      2–3 channels. There were additional runs with better results that aren't reflected.
      Find those runs/checkpoints, make the label-regression eval support the multi-channel
      inputs, and include those runs.
      **IMPLEMENTED (2026-07-12, commit bf94422); evaluation run in progress.**
      Found: the multi-channel machinery was `label_reg_evaluation.py::_build_merged`
      (components of the same spectrum concatenated: 1-comp raw 245 / emb 768,
      2-comp 490/1536, 3-comp 735/2304), with the better runs on the Apr-15 checkpoints
      `checkpoints/runai/recon_loss_experiment_3/recon-fe1.0_recon-tr{0,1}.0_frozen-enc{False,True}_5k.pt`
      (prior outputs: `code/eval_results/label_reg_20260415_*`).
      Implemented: `load_labeled_data(comps=...)` groups `dataset<D>_comp<C>_spec_<S>.wav`
      per spectrum (labels verified shared across comps); `label_regression.run` evaluates
      1/2/3-comp configs on the SAME spectra (metrics suffixed `_2c`/`_3c`);
      comparison table exports the new columns.
      Root cause of the old bad numbers: the previous loader sampled component files as
      independent samples — input R² ~0.006. Fixed loader on the Apr-15 fe-recon ckpt:
      input R² 0.41 (1-comp) → 0.71 (3-comp, 500-sample smoke), emb R² 0.30 → 0.36.
      **DONE — full evaluation (2,000 spectra, Apr-15 checkpoints), output:
      `code/eval_outputs/2026-07-12_20-51-45/`:**

      | Checkpoint | emb R² 1c | emb R² 2c | emb R² 3c |
      |---|---|---|---|
      | (raw-input baseline) | 0.406 | 0.723 | 0.789 |
      | fe-recon 5k | 0.305 | 0.510 | 0.506 |
      | fe+tr-recon 5k | 0.308 | 0.504 | 0.514 |
      | fe-recon frozen-enc 5k | **0.400** | **0.557** | **0.576** |

      The previously-missing better results are reproduced (old broken numbers: ~0.04).
      frozen-enc gives the best embeddings. Raw input still beats embeddings at every
      config (improvement negative) — a finding, not a bug.
      Remaining nice-to-have: per-config (2c/3c) scatter figures — only the 1-comp
      true-vs-pred scatter is plotted.
- Acceptance: sanity checks pass, and the embedding scores are explainable (you can say
  WHY each number is what it is).
  **Status: E1a verified; E1b implemented and being re-run. General embedding-source
  verification done (probe: same-input cos=1.0 at fe/proj/emb stages; extraction =
  final encoder layer, temporal mean pooling, clean inputs). Post-fix full analysis
  (Jan-07 / Feb-25 / tv_fe_short_3, corrected retrieval embeddings) running as
  `/tmp/eval_run10.log` (GPU 4). E1 awaits user review when both runs land.**

### E2. Pre- vs post-reconstruction checkpoint comparison (blocked by E1; run after E3)
Once E1 is fixed and verified, run the full comparison eval on exactly 2 checkpoints:
one from before any reconstruction training, one from after. Goal: analyze what
reconstruction training changed. Deliverable: the eval output dir + a short written
analysis of the differences per metric/figure.
**Status: NOT STARTED (per instructions: waits for E1 user review, runs after E3).
Candidate checkpoint pair to confirm with user: pre = `runai_long_train_2026-02-25`
(SSL only), post = an Apr-15 `recon_loss_experiment_3` ckpt or `tv_fe_short_3` (3AE).**

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
**DONE (2026-07-15, verified on run `2026-07-15_11-08-48`).** Implemented via a
`_relocate` helper in `report.py`: cross-checkpoint figures + `comparison_df.csv` →
`comparison/`; per-checkpoint figures/CSVs → `<checkpoint>/<method>/` with the
checkpoint suffix stripped from filenames (methods: similarity, noise_robustness,
clustering, label_regression, structured_similarity); standalone-eval runs use
`<method>/` at root; signal reconstruction → `reconstruction/`. Markdown image links
are run-dir-relative; HTML stays self-contained (base64). Verified: root contains
only eval_report.{html,md} + run_info.md + directories; MD links resolve; 37 figures.

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
