## Data layout

This directory stores modeling inputs, organized around two ideas:

- shared full-library feature tables;
- round-specific label collection and dataset assembly.

## High-level structure

- `registry.csv`
  - Canonical target metadata table.
- `shared/`
  - Shared cross-round inputs, including full-library feature tables and reusable inference datasets.
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

### `shared/features/`

Use this for full-library feature tables that are stable across rounds.

These are typically features computed from frozen upstream models or other shared representations, for example:

- `furin_6predictor.csv`
- `furin_chemdescriptors.csv`
- `furin_plmembeddings.csv`

These tables should usually be computed for the full inference universe for a target, not just for the assayed subset.

File naming convention:

- `{target_id}_{feature_set}.csv`

Examples:

- `furin_6predictor.csv`
- `pa_h3n2_chemdescriptors.csv`

This is the default source for both:

- `X_train`, after subsetting to compounds with labels available up to a given round;
- `X_inference`, for full-library inference.

### `shared/datasets/`

Use this for assembled inference-ready datasets that are reusable across rounds.

These should contain no labels and should only be saved here when they are built entirely from shared feature families rather than round-specific recalculations.

Examples:

- `furin_6predictor_chemdescriptors_inference.csv`

### `round_synthetic/`

Use this for pre-experimental synthetic prototype data.

Current contents include:

- `assay_data/`
  - synthetic label-bearing tables such as `furin.csv`
- `features/`
  - round-specific feature-family tables, if any are needed for the synthetic prototype round
- `datasets/`
  - round-specific assembled training datasets and any other assembled tables that genuinely depend on this synthetic round

This synthetic round is separate from `round_0` so that the first real experimental round can use `round_0` directly.
Stable feature families such as the current `6predictor` tables should live in `shared/features/`, not here.

### `round_k/assay_data/`

Use this for label-bearing tables for a specific real round.

Examples:

- `furin.csv`
- `pa_ph1n1.csv`

### `round_k/features/`

Use this only for feature tables that are specific to round `k`.

These should also usually be computed for the full inference universe for that target, not only for the assayed subset.

These use the same file naming convention as `shared/features/`:

- `{target_id}_{feature_set}.csv`

Examples:

- features recalculated after upstream model fine-tuning;
- recalibrated predictor outputs;
- round-specific uncertainty or novelty features;
- label-history-dependent derived features.

Do not copy unchanged full-library features from `shared/features/` into each round-specific feature folder just to create training slices.

### `round_k/datasets/`

Use this for direct model-loading training datasets assembled for round `k`, or other assembled datasets that genuinely depend on round-specific labels or features.

These are convenience snapshots rather than the canonical storage location for raw features.

Examples:

- `furin_6predictor_chemdescriptors_train.csv`
- `pa_h3n2_6predictor.csv`

These datasets may contain labels plus one or more feature families combined into a model-ready table.

## Training and inference procedure

At round `k`:

1. Collect all label-bearing assay rows available up to that point.
2. Load the relevant full-library feature tables.
  - Use `shared/features/` for stable feature families.
  - Use `round_k/features/` only when that round has updated or newly derived full-library features.
3. Build `X_train` by matching labeled compounds to the full-library feature tables.
4. Train the round `k` model(s).
5. Build `X_inference` from the shared full-library feature tables for that target, unless round-specific feature recalculations are part of the inference set.
6. Run inference across the full target library.
7. Save model outputs under `results/round_k/`.

When the inference dataset is built only from shared feature families, it should be saved under `shared/datasets/`.
When an assembled dataset depends on round-specific labels or round-specific recalculated features, it should be saved under `round_k/datasets/`.

## Practical rule

- `shared/features` = canonical shared full-library feature store
- `shared/datasets` = canonical shared inference-dataset store
- `round_synthetic/*` = synthetic prototype inputs before real experimental rounds begin
- `round_k/assay_data` = labels available by round `k`
- `round_k/features` = full-library features that changed at round `k`
- `round_k/datasets` = assembled round-specific training datasets and other round-specific assembled tables

In general, if a feature family has not changed, store it once in `shared/features/` and subset by compound ID when assembling training data.
