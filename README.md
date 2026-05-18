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

### Score functions

The current target-specific consensus learner can be written generically as:

```math
\eta_{it}^{\mathrm{generic}}
=
\alpha_t
+
\sum_{m \in \mathrm{Method}} \beta_{t,m}\, r_{it,m}
+
\sum_k \gamma_{t,k}\, d_{it,k}
+
\sum_j \theta_{t,j}\, c_{ij}.
```

Here:

- $i$ indexes compounds and `t` indexes targets
- $r_{it,m}$ are the per-method target-specific features such as `glidesp_pr`, `pignet2_pr`, `ligunity_pr`, `boltz2_pr`, `balm_pr`, and `mammal_pr`
- $d_{it,k}$ are derived target-specific features such as branch summaries, disagreement features, and the hand `1:2:3` consensus
- $c_{ij}$ are compound-only chemistry descriptors such as `logp`, `tpsa`, `qed`, and related ligand properties

When used as a hit model, the corresponding probability head is:

```math
\mathrm{logit}\left(\hat{p}_{it}^{(\tau)}\right) = \eta_{it}^{\mathrm{generic}}
```

When used as a potency model, $\eta_{it}^{generic}$ is the regression score for `pkd`.

For multi-strain target families such as PA, the planned hierarchical per-strain extension uses one pooled model with T-1 reference-coded strain indicators and strain-by-method interaction columns:

```math
\eta_{it}^{\mathrm{hier}}
=
\alpha_{\mathrm{ref}}
+
\sum_{s \in \mathcal{T}\setminus\{\mathrm{ref}\}} a_s\, I_{it}^{(s)}
+
\sum_{m \in \mathrm{Method}} \beta_m\, r_{it,m}
+
\sum_{s \in \mathcal{T}\setminus\{\mathrm{ref}\}} \sum_{m \in \mathrm{Method}} \delta_{s,m}\, I_{it}^{(s)} r_{it,m}
+
\sum_k \gamma_k\, d_{it,k}
+
\sum_j \theta_j\, c_{ij}.
```

Here:

- $\mathcal{T}$ is the set of pooled strains, for example `{pH1N1, H3N2, H5N1}` for PA
- `ref` is the chosen reference strain, for example `pH1N1`
- $I_{it}^{(s)}$ is a binary indicator that row `(i, t)` belongs to non-reference strain `s`
- $\beta_m$ is the shared target-family main effect for method `m`
- $\delta_{s,m}$ is the non-reference strain-specific adjustment for method `m`

This encoding implies effective per-strain method weights:

```math
\beta_{\mathrm{ref},m}^{\mathrm{eff}} = \beta_m
```

```math
\beta_{s,m}^{\mathrm{eff}} = \beta_m + \delta_{s,m}, \qquad s \neq \mathrm{ref}.
```

The corresponding hit-probability head is:

```math
\mathrm{logit}\left(\hat{p}_{it}^{(\tau)}\right) = \eta_{it}^{\mathrm{hier}}
```

In implementation terms, the hierarchical model keeps the existing shared feature blocks such as `6predictor_pr`, `6predictor_pr_derived`, and `chemdescriptors`, then adds a new T-1 hierarchical feature block consisting of:

- non-reference strain indicator columns such as `is_h3n2` and `is_h5n1`
- non-reference strain-by-method interaction columns such as `glidesp_pr_x_h3n2` and `glidesp_pr_x_h5n1`

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
