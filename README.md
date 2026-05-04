# FluMolScreen

FluMolScreen is a modular modeling workspace for prioritizing compounds against influenza-relevant host and viral targets using a six-method screening ensemble and a learned consensus model.

## Project objectives

- Build a reusable modeling pipeline that can train on assay labels collected over successive rounds and perform inference over a larger screening library.
- Start from six predictor outputs per compound-target pair and replace the current hand-weighted consensus with supervised consensus learners.
- Support both target-specific prioritization and broader multi-strain prioritization, beginning with furin and influenza A PA.
- Keep the system extensible to additional target classes, feature families, and future rounds of real experimental data.

## Current target scope

The current prototype scope includes:

- `furin`
- `pa_ph1n1`
- `pa_h3n2`
- `pa_h5n1`

The broader project is intended to extend to additional targets such as `fasn`, `na_ph1n1`, `na_h3n2`, and `na_h5n1`.

## Modeling setup

The current consensus-learning prototype is based on six existing predictor features:

- `glidesp_pr`
- `pignet2_pr`
- `ligunity_pr`
- `boltz2_pr`
- `balm_pr`
- `mammal_pr`

These are currently treated as frozen upstream features. Models are trained on labels collected by round, then used for inference over the larger target-specific screening universe.

The repo is organized around:

- `data/`
  - modeling inputs, including shared full-library features, round-specific assay labels, and model-ready datasets
- `results/`
  - round-specific model artifacts, evaluation outputs, and inference tables

See [data/README.md](/Users/charmainechia/Documents/projects/FluMolScreen/data/README.md) for the round-based data conventions.

## Current data state

- `data/round_synthetic/` contains the current synthetic prototype data derived from the inherited labeled screening tables.
- `data/round_0/` is reserved for the first round of actual experimental labels.
- `data/round_x/features/` is the shared feature store for full-library features that remain stable across rounds.

Within each round, assay-data filenames use only the target stem, for example `furin.csv` or `pa_h3n2.csv`. The round directory itself already captures whether the labels are synthetic or experimental.

## What to build next

1. Implement schema and loader code against the current `data/` layout.
2. Build feature assembly utilities that join label tables to shared full-library feature tables.
3. Add the first furin learner and evaluation workflow.
4. Implement grouped split logic for PA so compound identities are handled correctly across strains.
5. Build the hierarchical PA learner with shared and strain-specific behavior.
6. Add broad-PA scoring and evaluation.
7. Extend the feature system so future feature families such as derived branch features, chemistry descriptors, protein embeddings, and ligand embeddings can plug in cleanly.
