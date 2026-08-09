# SpectralFM — Remaining Tasks

Work through sections in order within each track.
**Priority order (2026-08-03): training-improvement round T4 → T2 → T5 → T6 is the
active work (short runs on Geoffrey + full comparison eval). In the eval track E5 → E6
remain, then E2 (E1, E3, E4 are done). T1's reporting waits for the verified eval.**
Mark tasks done with a short note: what changed, how it was verified, where the outputs are.

## Training pipeline — 3AE + d2v joint round (proposed)
Five-step **strictly sequential** pipeline (Steps 0–4), visualized in the
training-plan artifact (FE / Projection / Transformer / Decoders / Loss /
Hyperparams grid + architecture diagram). Fully linear, no side branch:
**T8 → T9 → T10 → T11 → T12 (full joint runs last)**.

**Correction (2026-08-09): most of this round needs far less new code than
originally scoped.** `fairseq/examples/data2vec/models/data2vec_audio.py` on
`recon/2ae-basemerge` (uncommitted on Geoffrey — `git status` shows it
modified, latest commit `62d36d0 "Fix stagnant regression loss..."`) already
has, natively, in one `forward()` call: the masked d2v loss
(`result["losses"]["regression"]`); `fe_recon_decoder` / `trans_recon_decoder`
gated by `lambda_recon_fe` / `lambda_recon_trans`; `train_only_fe` (default
`True`, freezes everything but FE); and full per-component init/freeze
(`init_fe_ckpt` / `init_proj_ckpt` / `init_transformer_ckpt`,
`freeze_fe_v2` / `freeze_proj` / `freeze_transformer_v2`, via
`_maybe_apply_recon_components()` + a shared `recon_components.py` module also
used by `code/train_reconstruction.py`). There's even a working (uncommitted)
hydra config, `fairseq/examples/data2vec/config/audio/pretraining/recon_loss/
spectralfm_recon_loss_basemerge.yaml`, that already runs d2v + recon_fe +
recon_trans jointly with per-component LR groups.

**Correction 2 (2026-08-09): the projection decoder has already been ported
into the native model too — confirmed by user, not by direct inspection.**
`head_proj` (a `TransformerMirrorDecoder`) already exists and already works
in Pipeline C (`code/train_reconstruction.py` + `code/recon_components.py`,
exactly what T2 trained — the linear/mlp768/mlp2048 sweep, mlp2048 winning
at 10k steps per T6). Per the user (2026-08-09): a `proj_recon_decoder` /
`lambda_recon_proj` equivalent has now been merged into
`data2vec_audio.py`'s native `mask=True` forward path too, alongside
`fe_recon_decoder`/`trans_recon_decoder`, with a minor adjustment from the
merge. **Note:** the Geoffrey checkout inspected for this doc
(`/mnt5/noy/SpectralFM`, `recon/2ae-basemerge`, both worktrees) did not yet
show `proj_recon_decoder`/`lambda_recon_proj` in
`fairseq/examples/data2vec/models/data2vec_audio.py` as of this check — likely
a sync/push gap, not a correctness question. T11 and T13 (now T10 and T12,
see Correction 3) are downgraded from "new engineering" to **existing —
verify**, matching T8/T10/T12. First step on either is confirming the merged
code is present and correct on whichever checkout is actually used to
launch, not building anything new.

**Correction 3 (2026-08-09): sequence simplified per user request — old
Step 1 removed, old Step 2 restructured, and every step's loss now carries
an explicit λ_d2v weight.**
- **Removed:** the old Step 1 (frozen-backbone baseline / 3AE-decoders-only
  side probe, formerly T9) — a diagnostic that never fed into the main chain,
  judged unnecessary. Everything below is renumbered down by one: old
  T10→T9, T11→T10, T12→T11, T13→T12; "Step N" labels shift the same way
  (old Step 2→1, 3→2, 4→3, 5→4).
- **Changed:** Step 1 (T9, formerly "FE fine-tune: attach decoder") no
  longer continues T8's FE checkpoint — it now starts **both FE and
  projection from random init together**, transformer still frozen from
  base_libri. Only the FE decoder is attached (same as before); projection
  is unfrozen but has no reconstruction term of its own here — it's shaped
  purely by the d2v gradient, the same mechanism as T8's own optional
  joint-training variant.
- **New: `lambda_d2v`.** Every loss expression below now carries an explicit
  weight on the d2v term, matching how `lambda_recon_fe/proj/trans` already
  work — not an implicit weight of 1. This field doesn't exist yet: the
  current `forward()` writes `result["losses"]["regression"] = loss.sum() *
  scale` with no lambda multiplier. Add `lambda_d2v: float = field(default=
  1.0, ...)` and multiply — small, same shape of change made three times
  already for the recon terms. Needed before any step below can be launched
  with a non-default λ_d2v.

### T8. Step 0 — FE pretrain via d2v loss only, no decoder (config work, not new code — validate first)
Train the FE end to end against the **d2v self-distillation loss only** — no
decoder, no reconstruction loss. Projection and transformer participate in the
forward pass (d2v's loss reads the transformer's top-k-layer output) but are
**frozen** — only FE receives gradients.
- FE: TRAIN, random init, 5-layer conv (245→47). Projection: FROZEN, init =
  base_libri. Transformer: FROZEN, init = base_libri. No decoders built.
- **Option:** train FE + projection together from random init (transformer
  still frozen from base_libri), rather than freezing projection at
  base_libri from the start — same d2v-only loss, no decoder either way, just
  widens which components receive gradients in this step. `freeze_proj` is
  already an independent switch (see `_maybe_apply_recon_components()`), so
  this is a one-flag change, not new engineering.
- Loss: λ_d2v·d2v (`lambda_d2v` is new — see Correction 3 above).
- Hyperparams: **needs reconsideration**, not carried over from the old
  recon-only design. d2v self-distillation conventionally trains much longer
  than a reconstruction pretrain — Pipeline B's own `spectralfm_base.yaml` runs
  to `max_update: 400000`. Start with a short smoke run, then pick a real step
  budget from that precedent. Data: `single_channel_one` (999k) or
  `single_channel_all` (9.1M).
- **Not new engineering (aside from `lambda_d2v`)** — `train_only_fe=True` is
  already the default, `lambda_recon_fe`/`lambda_recon_trans` already default
  to 0.0. This is a hydra launch config (base off `spectralfm_base.yaml` or
  the existing `recon_loss/` config family), setting
  `init_proj_ckpt`/`init_transformer_ckpt` = base_libri paths and leaving
  `init_fe_ckpt` unset.
- **Real blocker: validate before relying on it.** This code path is
  uncommitted and its last commit message ("Fix stagnant regression loss")
  indicates recent, possibly unresolved debugging. First actual task: run a
  short smoke test, confirm the d2v loss curve actually decreases (not
  stagnant), then commit the model-file changes before building anything
  else on top.
- Acceptance: FE checkpoint trained via pure d2v (proj/transformer never
  updated), d2v loss curve logged and confirmed healthy. Replaces
  `apr28_fe_recon_best.pt` as the canonical FE init for every step below.
- Context: `apr28_fe_recon_best.pt`'s own saved metadata reads
  `n_samples=1000, steps=10000, lr=1e-4` — at the trainer's default effective
  batch (512) that's ~5,000 passes over the *same* 1,000 wavs. A bad
  foundation for the FE everything downstream inherits — the motivation for
  moving off the old recon-only Step 0.

### T9. Step 1 — FE + projection jointly from random init: attach FE decoder, add recon(fe) on top of d2v (config work — precedented by the existing basemerge yaml)
First link in the (now fully linear) chain — starts fresh rather than
continuing T8, training FE and projection together and attaching the FE
decoder for the first time.
- FE: TRAIN, random init, 5-layer conv (245→47). Projection: TRAIN, random
  init. Transformer: FROZEN, init = base_libri. Decoders: FE decoder TRAIN
  (random init, newly attached); projection/transformer decoders not built
  (λ=0) — projection is unfrozen but only shaped by the d2v gradient here, no
  `recon_proj` term yet (that's T10's job). Loss: λ_d2v·d2v + λ_fe·recon_fe
  [+ λ_tv·TV(recon_fe)].
- Hyperparams: 5,000–10,000 steps, lr_fe 1e-5, lr_proj 1e-5, λ sweep
  {0.1, 1, 10} for λ_fe.
- **Not new engineering (aside from `lambda_d2v`)** — same native model as
  T8, `lambda_recon_fe>0` is literally what
  `spectralfm_recon_loss_basemerge.yaml` already exercises. Set
  `freeze_fe_v2: false`, `freeze_proj: false`, `freeze_transformer_v2: true`
  — all existing independent switches. No TV term exists natively in the
  model yet (Pipeline C has it via `_tv_loss`) — port it over if wanted,
  small addition.
- Acceptance: sweep completed with logged d2v + recon_fe curves. Winning
  config's FE+projection checkpoint is what T10–T12 build on.

### T10. Step 2 — Projection fine-tune + d2v, 3 capacities (existing — verify)
- FE: FROZEN, init = **T9's winning checkpoint**. Projection: TRAIN, 3
  variants — linear / mlp768 / mlp2048, random init (T2 recipe). Transformer:
  FROZEN, init = base_libri. Decoders: projection decoder TRAIN (random init,
  ×3); FE decoder carried from T9 but frozen; transformer decoder not built
  (λ=0). Loss: λ_d2v·d2v + λ_proj·recon_proj.
- Hyperparams: 10,000 steps, lr 1e-5 (fine-tune LR), warmup 500 — not the
  original 2k steps: T6 found capacity ordering flips (mlp2048 only wins once
  trained long enough). Since T6 already crowned mlp2048 the winner, consider
  porting that config directly rather than re-sweeping all 3 capacities here.
- **Open decision: does the projection decoder start fresh or load from T2/T6?**
  Either train all 3 capacity variants from random init here (the current
  spec), or initialize from T2's already-completed linear/mlp768/mlp2048
  checkpoints (T6 already crowned mlp2048 the winner at 10k steps) and
  fine-tune further under the new d2v-joint loss instead of starting over —
  decide before launching.
- **Not new engineering (aside from `lambda_d2v`) — per user (2026-08-09),
  the projection decoder has already been merged into the native model, with
  a minor adjustment from the merge.** Freeze granularity already exists
  (`freeze_proj=False` while FE/transformer stay frozen via
  `freeze_fe_v2`/`freeze_transformer_v2`). Before launching, verify:
  - `proj_recon_decoder` (or equivalent) is instantiated in
    `data2vec_audio.py` alongside `fe_recon_decoder`/`trans_recon_decoder`,
    and a `lambda_recon_proj` config field gates it.
  - It's wired into `forward()`'s masked (`mask=True`) path and contributes to
    `result["losses"]` the same way `recon_fe`/`recon_trans` do.
  - Per-component init/freeze exists for it in `_maybe_apply_recon_components()`
    (an `init_proj_recon_decoder_ckpt` / `freeze_proj_recon_decoder` pair, or
    whatever the merge named them) and it's tagged for the composite optimizer.
  - Gradients actually reach it under `mask=True` — run (or extend)
    `recon_components.audit_gradient_flow()`-style check, since that helper as
    last inspected only tested `mask=False`.
  - Confirm which checkout (Geoffrey vs. wherever the merge landed) has the
    change — the `/mnt5/noy/SpectralFM` copy inspected 2026-08-09 didn't show
    it yet.
- Acceptance: verification above passes; all 3 capacity variants (or just
  mlp2048) trained to completion with logged d2v + recon curves. Winning
  variant is what T11–T12 build on.

### T11. Step 3 — Transformer fine-tune + d2v (config work — confirm MLP-proj checkpoint loading first)
- FE: FROZEN, init = **T9's checkpoint**. Projection: FROZEN, init =
  **T10's winning variant**. Transformer: TRAIN, init = base_libri, trained on
  NOVA data. Decoders: transformer decoder TRAIN (random init); FE/projection
  decoders carried from T9/T10 but frozen. Loss: λ_d2v·d2v + λ_trans·recon_trans.
- Hyperparams: 10,000–20,000 steps (effectively continued SSL pretraining),
  lr 1e-5 (fine-tune LR), warmup 500.
- **Not new engineering for the loss/freeze mechanics (aside from
  `lambda_d2v`)** — `freeze_transformer_v2=False` + `lambda_recon_trans>0` is
  native, same as T8/T9. One thing to confirm first: if T10's winning
  projection is an MLP variant (mlp768/mlp2048), check
  `recon_components.py`'s `load_post_extract_proj_from_ckpt` actually
  round-trips an MLP shape via `init_proj_ckpt` — T2 solved this for the
  eval-side loader; unclear whether Pipeline B's loader was updated to match.
- Why this one matters: most direct test of the T6 verdict — pair an
  SSL-informative backbone with reconstruction. Compare directly against T5b.
- Acceptance: completed checkpoint, compared against T5b.

### T12. Step 4 — Full joint fine-tune (existing — verify, depends on T10)
- FE: TRAIN, init = T9's winner. Projection: TRAIN, init = T10's winner.
  Transformer: TRAIN, init = T11's checkpoint (or base_libri, if T11
  underperforms T5b). Decoders: all three TRAIN, seeded from T9/T10/T11's
  winning configs. Loss: λ_d2v·d2v + λ_fe·recon_fe + λ_proj·recon_proj +
  λ_trans·recon_trans [+ λ_tv·TV(recon_fe)].
- Hyperparams: largest step budget of the round; lr 1e-5 (fine-tune LR); λs
  carried over from T9–T11.
- Mechanically native aside from `lambda_d2v` (`train_only_fe=False` or all
  three `freeze_*_v2=False`, `lambda_recon_fe/trans>0` already exist, and per
  T10 `lambda_recon_proj` should too now) — once T10's verification checklist
  passes, this step is pure config, no new code.
- Run only after T9, T10, T11 land in sequence — seed lambdas/inits from
  whichever config wins at each link. Hardest run to attribute if metrics
  move, since three components change at once.
- Acceptance: completed checkpoint, full comparison eval against T9–T11
  individually.


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

### T7. Audit decoder-capacity confound in the 3AE depth comparison
The 3AE setup shares one encoder trunk (FE → post_extract_proj → transformer,
`build_data2vec_audio_backbone()`) and taps it at three depths via forward hooks
on `backbone.layer_norm` / `backbone.post_extract_proj`, plus the backbone's own
output — one `backbone(...)` call per step, `code/train_reconstruction.py:957-1010`.
That single-encoder design is what makes "same weights, different depth"
comparisons valid in the first place (T5's depth-inversion finding depends on it).
But the three decoders reading those taps are *not* the same architecture:
`head_fe` is a `MirrorDecoder` (conv-transpose mirroring the FE, input 47×512)
while `head_proj`/`head_trans` are `TransformerMirrorDecoder` (a different
architecture, input 47×768). A fe-vs-trans reconstruction gap could therefore
reflect either (a) how much signal survives at that encoder depth, or (b) how
expressive that decoder architecture happens to be — currently unmeasured.
- Compare parameter counts / effective capacity of `MirrorDecoder` vs.
  `TransformerMirrorDecoder` as actually instantiated in these runs.
- Sanity check: on one existing checkpoint, swap `head_fe` for a
  `TransformerMirrorDecoder` (or vice versa) and see whether the fe/proj/trans
  MSE ordering from T5/T6 holds or flips. If it flips, part of the "depth" story
  is decoder capacity, not backbone information content.
- Acceptance: a short writeup stating whether the T5/T6 depth-ordering findings
  are robust to decoder-architecture choice, or need a caveat attached.

### T3. Evaluate transformer without masking (eval experiment, no training)
Find where masking is applied in the transformer forward path and run the evaluation with
masking disabled entirely. Expectation: better results. Report masked vs. unmasked
side by side; make the no-masking mode a proper CLI flag, not a code edit.

---

## Cleanup

### C1. Remove the legacy HuggingFace training path
`code/run_experiment.py`, `code/customize_model.py`, `code/args_parser.py`,
`code/trainer.py`, `code/evaluate.py`, `code/testing.py`, `code/main.py` — an
older HF-`transformers`-based training/eval chain that predates both the
regular backbone-training path (`spectralfm_base.yaml` + `hydra_train`) and
the autoencoder-training path (`code/train_reconstruction.py`), and isn't
part of either. No `.sh`/`.yaml` in the repo launches any of these files.
`code/compute_stats.py` is only imported by this chain — its plotting style
was already ported into `code/eval/report.py` (see the comments there), so
it has nothing left to lose functionally.

**Caveat found while merging autoencoder training into this branch:**
`code/model_loader.py` is part of this same import chain but is **not**
dead — `code/evaluation_runner.py` and other `code/eval_*.py` scripts (the
older, fairseq-dependent eval tooling that came in with the autoencoder
merge) import it for checkpoint loading and the `/storage`↔`/mnt5` path
remap. Do not remove `model_loader.py` as part of this cleanup; it would
need its own decision once `evaluation_runner.py`'s fate relative to
`code/eval/` is settled.

Also unrelated to any training path and safe to remove regardless of the
above: `code/wandb_logger.py` (an orphaned distilbert/Yelp-review tutorial
script, not imported anywhere, not SpectralFM code).

### C2. Remove redundant files from git
Sweep the repo for cruft accumulated from iteration/experimentation and
either delete or `.gitignore` it. Known candidates found this session:
- `Dockerfile.bak`, `Dockerfile_old` (repo root, untracked) — stale copies
  superseded by `Dockerfile`.
- `code/nohup.out`, `code/files_to_copy.txt`, `code/.DS_Store`,
  `.DS_Store` files generally — should be `.gitignore`d, not committed.
- `code/eval_outputs/` (currently untracked, growing every eval run) —
  decide whether run outputs belong in git at all, or should be
  `.gitignore`d with only reports/summaries checked in deliberately.
- Re-check whether the exploratory/one-off scripts that arrived via the
  autoencoder-training merge (`debug_recon_encoding.py`,
  `code/analyze_hydra_recon_checkpoint.py`, `code/plot_*.py` variants,
  `code/test_sanity_fixed_mask.py`, etc. — see the provenance note in
  `CLAUDE.md`'s Repo layout section) are still needed or were one-off
  debugging aids.
- Confirm no other now-dead config variants remain alongside
  `spectralfm_base.yaml`/`spectralfm_full_train.yaml` (four were already
  removed while merging autoencoder training into this branch, after
  confirming zero references anywhere in the tree — same check applies here).
