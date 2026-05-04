## Data layout

This directory stores modeling inputs, organized around two ideas:

- shared full-library feature tables;
- round-specific label collection and dataset assembly.

## High-level structure

- `registry.csv`
  - Canonical target metadata table.
- `round_x/`
  - Shared feature store for full-library feature tables that can be reused across rounds.
- `round_synthetic/`
  - Synthetic prototype data used before real experimental labels are available.
- `round_0/`, `round_1/`, ...
  - Real round-specific labels, datasets, and any round-specific feature recalculations.

## Core idea

At each real round `k`, models are trained on all labels collected up to that point, then used for inference over the full screening library for each target.

This means we need:

- label-bearing assay rows for compounds observed by that round;
- full-library feature tables for the target-specific inference universe.

Training rows are constructed by matching assayed compounds to the relevant full-library feature tables.

## Folder meanings

### `round_x/features/`

Use this for full-library feature tables that are stable across rounds.

These are typically features computed from frozen upstream models or static representations, for example:

- `furin_6predictor.csv`
- `furin_chemdescriptors.csv`
- `furin_plmembeddings.csv`

These tables should usually be computed for the full inference universe for a target, not just for the assayed subset.

This is the default source for both:

- `X_train`, after subsetting to compounds with labels available up to a given round;
- `X_inference`, for full-library inference.

### `round_synthetic/`

Use this for pre-experimental synthetic prototype data.

Current contents include:

- `assay_data/`
  - synthetic label-bearing tables such as `furin_synthetic.csv`
- `features/`
  - current synthetic-round feature-family tables
- `datasets/`
  - direct model-loading datasets such as `furin_synthetic_6predictor.csv`

This synthetic round is separate from `round_0` so that the first real experimental round can use `round_0` directly.

### `round_k/assay_data/`

Use this for label-bearing tables for a specific real round.

Examples:

- `furin_explabels.csv`
- `pa_ph1n1_explabels.csv`

### `round_k/features/`

Use this only for feature tables that are specific to round `k`.

These should also usually be computed for the full inference universe for that target, not only for the assayed subset.

Examples:

- features recalculated after upstream model fine-tuning;
- recalibrated predictor outputs;
- round-specific uncertainty or novelty features;
- label-history-dependent derived features.

Do not copy unchanged full-library features from `round_x/features/` into each round-specific feature folder just to create training slices.

### `round_k/datasets/`

Use this for direct model-loading datasets assembled for round `k`.

These are convenience snapshots rather than the canonical storage location for raw features.

Examples:

- `furin_explabels_6predictor.csv`
- `all_targets_explabels_6predictor.csv`

These datasets may contain labels plus one or more feature families combined into a model-ready table.

## Training and inference procedure

At round `k`:

1. Collect all label-bearing assay rows available up to that point.
2. Load the relevant full-library feature tables.
   - Use `round_x/features/` for stable feature families.
   - Use `round_k/features/` only when that round has updated or newly derived full-library features.
3. Build `X_train` by matching labeled compounds to the full-library feature tables.
4. Train the round `k` model(s).
5. Build `X_inference` from the full-library feature tables for that target.
6. Run inference across the full target library.
7. Save model outputs under `results/round_k/`.

## Practical rule

- `round_x/features` = canonical shared full-library feature store
- `round_synthetic/*` = synthetic prototype inputs before real experimental rounds begin
- `round_k/assay_data` = labels available by round `k`
- `round_k/features` = full-library features that changed at round `k`
- `round_k/datasets` = assembled train/inference-ready tables for convenience

In general, if a feature family has not changed, store it once in `round_x/features/` and subset by compound ID when assembling training data.
