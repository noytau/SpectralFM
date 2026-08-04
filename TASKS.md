# SpectralFM — Remaining Tasks

Work through sections in order within each track.
**Priority order (2026-08-03): training-improvement round T4 → T2 → T5 → T6 is the
active work (short runs on Geoffrey + full comparison eval). In the eval track E5 → E6
remain, then E2 (E1, E3, E4 are done). T1's reporting waits for the verified eval.**
Mark tasks done with a short note: what changed, how it was verified, where the outputs are.

## Evaluation pipeline — NEW HIGH-PRIORITY TASKS (do first, in order)

### E4. Multi-dataset evaluation as the default
Run each eval on the following datasets SEPARATELY (deterministic seed-42 draws):

| Alias | Source | Split | Size |
|---|---|---|---|
| `sanity` | `nova_data/single_channel_1k` (fixed) | **train.tsv** | 100 |
| `in_dist` | `nova_data/single_channel_10k` | valid.tsv | 500 (max) |
| `multi_ch` | `nova_data/multi_channel` | valid.tsv | 1,000 |
| `labeled` | `nova_data/labeled_data` | labels.tsv (comp-grouped) | 1,000 |
| `samples` | `nova_data/sampled_data` | valid.tsv | 1,000 |

Dataset × eval matrix (defaults; excluded combinations are skipped and noted in run_info.md):

| Eval | sanity | in_dist | multi_ch | labeled | samples |
|---|---|---|---|---|---|
| stack query (embedding_similarity) | ✓ | ✓ | — (single-channel only) | — | — |
| clustering | — (only 10 stacks at n=100) | ✓ | — (single-channel only) | — | — |
| noise robustness | ✓ | ✓ | — | — | — |
| label regression | — | — | — | ✓ | — |
| signal reconstruction | ✓ | ✓ | ✓ | — | ✓ |
| structured similarity | run-level, once per checkpoint (it already spans datasets) |

- The **unstructured cosine similarity maps are REMOVED** from the suite entirely.
- Directory layout: NO extra nesting level — results go to
  `run_dir/<checkpoint>/<method>_<dataset>/` (e.g. `similarity_in_dist/`,
  `noise_robustness_sanity/`); cross-checkpoint figures to `comparison/<dataset>/`.
- Sizes: per-dataset defaults above; `--eval_set_sizes sanity=100,in_dist=500,...`
  overrides; global `--eval_set_size N` shortcut for smoke runs.
- Acceptance: one full run produces the hierarchy above with per-dataset results,
  skipped cells listed in run_info.md, and reports rendering all figures.
**DONE (2026-07-18).** Implemented in commits 4606bf3 + 0c0205e: manifest-subset
loader (whole-stack seeded draws), DATASET_SPECS/EVAL_DATASET_MATRIX, alias-suffixed
metrics, `<checkpoint>/<method>_<dataset>/` layout, `comparison/<alias>/`,
alias-aware reports/CSVs, unstructured cosine maps removed, --eval_set_sizes/--eval_set_size.
Extras: label-reg plots per 1/2/3-comp config; reconstruction outputs filed under
their model's directory (inside the matching compared checkpoint's dir).
Accepted on the full-size run `code/eval_outputs/2026-07-18_19-49-10/`
(Feb-25 SSL vs 3ae_norm_exp2_long, 66 figures, skips noted in run_info.md).
Known minor gap: `labeled` size not yet wired to --eval_set_sizes
(label_reg_max_samples governs it separately).

### E5. Input-space comparison baselines (after E4)
Show the input-space counterpart next to the embedding result wherever applicable:
- stack query: already computed & shown — additionally put the input bar next to the
  embedding bar in the comparison scalar grid.
- structured similarity: already has the Input Space panel — no change.
- label regression: input R² already computed — extend the R² bar figure to the
  `_2c`/`_3c` configs (input vs embedding per config).
- clustering: NEW — run the identical clustering metrics on raw inputs; grouped bars
  ARI/NMI/silhouette input-vs-embedding.
- noise robustness / signal reconstruction: NO input-space comparison (per decision).
- Acceptance: every applicable figure shows the input counterpart side by side.

### E6. Per-checkpoint training metadata + upgraded HTML report (after E5)
Metadata card per evaluated checkpoint: training data (+n_samples), trained vs frozen,
AE reconstruction stages, per-component init (FE/proj/transformer: base_libri / past
ckpt / random), lr, loss, batch size, steps/epochs.
Recoverability (verified 2026-07-18): fairseq SSL ckpts → `cfg.task.data`,
`cfg.optimization`, criterion; 3AE ckpts → `freeze_map`, `init_manifest` (exact init
source per component), `components` (param counts), lambdas, lr/steps/batch —
missing only the training-data path; old fe/tr-recon ckpts → lr/steps/batch/tag
hints only.
- Implement `read_checkpoint_metadata()` per format; every missing field renders as
  **"unknown"** — never guessed.
- Going forward: training scripts write `metadata.yaml` next to each checkpoint.
- One-time backfill: ONLY the ~10 checkpoints evaluated so far
  (`checkpoints/metadata_backfill.yaml`, reviewed by user before use).
- HTML: metadata card at the top of each checkpoint section; side-by-side comparison
  plots + report stay at the run root (E3 layout).
- Acceptance: a mixed-format run (SSL + 3AE + old recon) renders correct cards with
  honest unknowns.

## Evaluation pipeline — original tasks

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

### T2. Improve projection layer (MLP) — active 2026-08-03
Current projection is a single bare `nn.Linear(512→768)` (`post_extract_proj`,
`fairseq/examples/data2vec/models/data2vec_audio.py:116` — no LN/activation/dropout).
Plan (confirmed with user): train the MLP variants via the 3AE recon trainer on Geoffrey.
- Model change (branch `recon/2ae-basemerge`): config fields `proj_mlp_hidden_dim` /
  `proj_mlp_layers`; when hidden_dim > 0, `post_extract_proj` becomes
  `Sequential(Linear(512,h), GELU, LayerNorm(h), Linear(h,768))`. Default keeps the
  legacy single Linear byte-compatible.
- Trainer: `--proj_mlp_hidden_dim` CLI arg; persist it (and `lambda_tv_fe`) in the
  3AE save object.
- Runs (base recipe = T4, but proj trained: `--random_init_proj`, no `--freeze_proj`,
  `--lambda_recon_proj 1.0 --lambda_recon_fe 1.0`, head_trans frozen):
  A) linear control (random init), B) MLP hidden 768, C) MLP hidden 2048.
- Eval side (`eval-methods`): `checkpoint_loader.py` + `model.py` build/remap a matching
  Sequential at `hf_model.feature_projection.projection`; MLP loads must not silently
  drop projection keys.
- Acceptance: all three variants train to completion; eval loader round-trips an MLP
  checkpoint with zero missing/unexpected projection keys; compared in T6.
**TRAINING + RECON EVAL DONE (2026-08-03); embedding eval in T6.** Discovery: the
model side (`build_post_extract_proj`, `post_extract_proj_type`/`_mlp_hidden` cfg)
already existed uncommitted in Geoffrey's data2vec_audio.py — only trainer plumbing
was missing (added in be3dbdd). 3 runs completed
(`fairseq/outputs/signal_recon_proj_mlp_local/`, launcher
`fairseq/launch_proj_mlp_geoffry.sh`; lr 1e-4, 2000 steps, eff. batch 128,
950 samples, λ_fe=λ_proj=1, λ_tv=0, proj random-init). Held-out proj-pathway MSE:
linear=0.09138, **mlp768=0.08139 (best, −11%)**, mlp2048=0.08542. FE-MSE identical
across variants (same frozen FE) — the MLP projection demonstrably preserves more
signal information. Eval-side MLP auto-detection added to
`code/eval/checkpoint_loader.py` (`_install_mlp_projection` +
`_assert_projection_loaded`); verified missing=0/unexpected=0 on an MLP ckpt and
legacy ckpts unchanged.

### T4. TV-regularized FE-decoder sweep (short) — active 2026-08-03
Rerun the already-implemented TV reconstruction training (`--lambda_tv_fe`,
`code/train_reconstruction.py:458` on `recon/2ae-basemerge`; prior runs tv_fe_short_1–3)
to see if TV helps. Recipe from `fairseq/submit_signal_recon_tv_fe_runai.sh`, run
directly on Geoffrey GPUs (as `signal_recon_tv_fe_local` precedent):
- FE+LN from `apr28_fe_recon_best.pt` (frozen); transformer `base_libri_official.pt`
  (frozen); proj + 3 heads warm-started from `3ae_norm_exp2_long.pt`; only `head_fe`
  trained with `--lambda_recon_fe 1.0 --lambda_tv_fe λ --normalize`.
- Sweep λ_tv ∈ {0 (baseline), 0.01, 0.1, 1.0}, one GPU each; manifest
  `single_channel_1k` (990→950 samples; `_with_valid` variant doesn't exist),
  steps 2000, warmup 200, lr 1e-4, effective batch 128.
- Acceptance: 4 completed checkpoints with logged `recon_fe` + `recon_fe_tv_loss`
  curves; best λ picked for T2; compared in T6.
**DONE (2026-08-03).** 4 runs completed on Geoffrey GPUs
(`fairseq/outputs/signal_recon_tv_fe_local/20260803_143012Z_tv*/`, launcher
`fairseq/launch_tv_fe_geoffry.sh`, commit be3dbdd on recon/2ae-basemerge).
Held-out recon eval (200 samples, single_channel_10k valid, seed 42 —
`code/eval_outputs/recon_quick_20260803/recon_results.csv`): FE-MSE
tv0=0.03735, tv0.01=0.03729, **tv0.1=0.03672 (best, −1.7%)**, tv1.0=0.04194;
train-final FE-MSE 0.0042/0.0043/0.0048/0.0187. Verdict: mild TV (0.1) improves
held-out reconstruction; λ=1.0 over-smooths (visible peak flattening in
`tv_sweep_fe_overlay.png`). Note: these ckpts predate the lambda_tv_fe
persistence fix — λ values documented here, not in the .pt files. Embeddings
are identical across the 4 ckpts by construction (only head_fe trains), so T6
carries one representative (tv0p1).

### T5. Reconstruction heads on the Feb-25 trained transformer — active 2026-08-03
All prior recon runs used the base_libri transformer. Train the 3 mirror heads from
scratch on top of the full frozen SSL backbone
`checkpoints/runai/runai_long_train_2026-02-25_13-46-46.pt` (FE+LN+proj+transformer all
loaded via `--ckpt`, all frozen; heads random-init for clean attribution):
`--lambda_recon_fe/proj/trans 1.0`, steps 3000, warmup 300, lr 1e-4, same manifest as T4.
- Optional variant: heads warm-started from `3ae_norm_exp2_long.pt` (spare GPU).
- Acceptance: completed 3AE checkpoint; all three recon losses decreasing; compared in T6.
**DONE (2026-08-03).** Run completed
(`fairseq/outputs/signal_recon_feb25_local/20260803_145436Z_feb25_3ae_scratch/`,
launcher `fairseq/launch_t5_feb25_recon_geoffry.sh`; 3000 steps, warmup 300, lr 1e-4,
eff. batch 128; final train L2: trans=0.081, fe=0.047, proj=0.044). Held-out recon:
fe=0.1361, proj=0.1141, **tr=0.0898**. Key finding: on the Feb-25 SSL backbone
reconstruction IMPROVES with depth (fe→proj→tr), the exact inverse of the
base_libri/apr28 stack (fe=0.037 → tr=0.13) — the SpectralFM-trained transformer
retains ~31% more end-to-end signal information at the encoder output than base_libri,
while its FE (never trained for reconstructability) is far less decodable than the
apr28 recon-trained FE. Figures: `backbone_{fe,tr}_overlay.png` in
`code/eval_outputs/recon_quick_20260803/`. Warm-start variant not run (scratch run
converged fine; clean attribution preserved).

### T6. Full comparison eval of the training-improvement round — active 2026-08-03
Run the multi-dataset eval suite (E4 default) on Geoffrey comparing: Feb-25 SSL baseline,
T4 best-TV + λ=0 control, T2 best MLP + linear control, T5 checkpoint; plus a
`signal_reconstruction` pass per 3AE checkpoint. Copy run dir to local
`code/eval_outputs/`, write a short per-task verdict (did TV help? did the MLP proj
help? what did the trained transformer change?).
- Acceptance: one eval run dir with all checkpoints in `comparison/comparison_df.csv`,
  HTML report renders, verdicts written here.
**EMBEDDING SUITE DONE (2026-08-03, run `2026-08-03_19-42-31`); verdicts below.**
Reconstruction-pathway comparison also done (`code/eval_outputs/recon_quick_20260803/`).
Lineup note: all four T4 ckpts share one frozen backbone (only head_fe differs) and
T5's backbone ≡ Feb-25 baseline, so their embedding rows are represented by tv0p1
and the baseline respectively.
- **Label regression is the decisive metric: Feb-25 SSL baseline emb R²=0.44
  (beats input baseline 0.38); ALL recon-stack backbones (tv0p1, projlin,
  mlp768, mlp2048, exp2_long) score ≈0** (−0.01…0.17, far below the 2c/3c input
  baselines too). The recon stacks share the base_libri (speech) transformer +
  apr28 FE — reconstruction-oriented training does not produce label-informative
  embeddings; SSL on NOVA data does. Which transformer dominates every
  second-order choice (TV weight, proj arch).
- Stack retrieval (sanity): proj-trained backbones best (projlin 0.114,
  mlp768 0.116 vs baseline 0.064); in_dist retrieval ≈ input level for all.
- Noise robustness: mlp2048 most robust (gaussian_std 0.90); mlp768 lowest (0.78).
- Clustering: ARI ≈ 0 for every model (NMI 0.39–0.44, baseline highest) — no
  k-means-recoverable stack structure in any embedding.
- Verdict for the program: reconstruction quality and representation quality are
  currently DECOUPLED. The promising direction is T5b-style: keep the Feb-25 SSL
  backbone (label-informative) and add reconstruction capability via decoders —
  not replace the backbone with recon-trained stacks.
**OVERNIGHT LONG ROUND DONE (2026-08-04)** — all results in
`code/eval_outputs/recon_full_{tvL,projL,3way}_20260804/` (+ overlays dirs), all
runs 10k steps / effective batch 1000 / lr 1e-4 / warmup 500 / normalize, metadata
verified in the checkpoints. Verdicts:
- **TV (final): λ_tv ≈ 0.1 wins everywhere once trained long enough.** in_dist
  fe-MSE monotone in λ (0.0367 → 0.0614); the short-round "strong TV helps OOD"
  effect washed out at 10k steps (samples median now also best at λ=0.1).
  TV = mild regularizer; use 0.1, don't go higher.
- **Projection (final): capacity ordering flips with training length —
  mlp2048L best everywhere at 10k steps** (in_dist proj-MSE 0.0451 < mlp768L
  0.0499 < linear 0.0538; −16% vs linear). Confirms mlp2048 was undertrained at
  2k. MLP > linear robust across all datasets and both schedules.
- **T5b (10k-step 3AE on frozen Feb-25): best transformer-stage reconstruction
  measured** — in_dist tr-MSE 0.0558 (contamination-corrected) vs T5 0.0880 /
  april 3AE 0.1287. Depth-inversion confirmed: on the SSL backbone deeper =
  more reconstructable. april 3AE still owns the fe stage (0.047) and OOD.
- Sanity audit: T5b's `--data_dir` glob overlapped 38/500 in_dist valid files →
  13/200 eval samples; clean-only numbers reported above, no conclusion flips.
  The 1k subset (all TV/proj rounds) has ZERO overlap with 10k valid. Known
  artifacts: `tr` column meaningless for frozen-partially-warm-started heads
  (TV/proj rows); `samples` means are outlier-driven (use medians);
  in-memory-loader normalize bug found+fixed (commit 47b18ec side).
  Future `--data_dir` runs must exclude valid-set files.

### T3. Evaluate transformer without masking (eval experiment, no training)
Find where masking is applied in the transformer forward path and run the evaluation with
masking disabled entirely. Expectation: better results. Report masked vs. unmasked
side by side; make the no-masking mode a proper CLI flag, not a code edit.
