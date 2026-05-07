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

Current model comparison uses nested cross-validation:
- outer `k = 5` fold CV for unbiased evaluation of each feature-set / model ablation
- inner `l = 3` fold CV or holdout tuning for candidate-specific hyperparameter selection
- Optuna tuning within each candidate, with full fold metrics saved for every feature/model combination rather than collapsing immediately to one global winner

Current uncertainty-aware inference uses an adaptive conformal workflow on the final fitted candidate:
- reserve `20%` of labeled data as a calibration subset
- fit a bootstrap ensemble of `m = 10` models on the remaining proper-training rows
- use the ensemble mean and spread on the calibration subset to fit a symmetric conformal scaler
- save final inference outputs as `pred_mean` plus a calibrated half-width `pred_err`

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
