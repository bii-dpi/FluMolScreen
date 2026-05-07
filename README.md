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

These are currently treated as frozen upstream percentile-rank features. Models are trained on labels collected by round, then used for inference over the larger target-specific screening universe.
The current shared feature family is `6predictor_pr`, and derived branch/disagreement features can be computed from these six percentile predictors and stored as additional shared feature tables.

Planned uncertainty estimation is based on an adaptive conformal workflow. The current recommended design is:
- outer `k = 5` fold CV for unbiased evaluation
- inner `l = 3` fold CV for model / feature / hyperparameter selection
- within each outer training fold, reserve `20%` as a calibration subset
- train an ensemble of `m = 10` models on the remaining proper-training subset
- use ensemble prediction mean and standard deviation together with conformal calibration on absolute standardized residuals to produce sample-specific prediction intervals

The first implementation target is symmetric `90%` prediction intervals for ridge and XGBoost models.

The repo is organized around:

- `data/`
  - modeling inputs, including shared full-library features, round-specific assay labels, and model-ready datasets
- `results/`
  - round-specific model artifacts, evaluation outputs, and inference tables
- `build_dataset.py`
  - assembles target-specific training and inference datasets from assay-data and feature tables
- `run_consensus_learner.py`
  - rebuilds datasets for configured feature comparisons, runs cross-validated modeling, and saves evaluation/inference outputs

See [data/README.md](/Users/charmainechia/Documents/projects/FluMolScreen/data/README.md) for the round-based data conventions.

## Current data state

- `data/round_synthetic/` contains the current synthetic prototype data derived from the inherited labeled screening tables.
- `data/round_0/` is reserved for the first round of actual experimental labels.
- `data/shared/features/` is the shared feature store for full-library features that remain stable across rounds.
- `data/shared/datasets/` is the shared store for reusable inference datasets built from shared feature families.

Within each round, assay-data filenames use only the target stem, for example `furin.csv` or `pa_h3n2.csv`. The round directory itself already captures whether the labels are synthetic or experimental.

## What to build next

1. Continue consolidating reusable full-library feature tables and inference-ready datasets under `data/shared/`.
2. Replace the current minimal random-k-fold regression scaffold with task-appropriate evaluation schemes, especially grouped compound splits for PA.
3. Add the first furin learner that serves as the real baseline consensus model rather than just a placeholder regression scaffold.
4. Implement hierarchical PA modeling with shared and strain-specific behavior.
5. Add broad-PA scoring and evaluation.
6. Extend feature assembly to support additional feature families such as derived branch features, chemistry descriptors, protein embeddings, and ligand embeddings.
