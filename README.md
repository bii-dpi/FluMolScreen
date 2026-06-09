# FluMolScreen

FluMolScreen is a modular modeling workspace for prioritizing compounds against influenza-relevant host and viral targets using a six-method screening ensemble and a learned consensus model.

## Project objectives

- Build a reusable modeling pipeline that can train on assay labels collected over successive rounds and perform inference over a larger screening library.
- Start from six predictor outputs per compound-target pair and replace the current hand-weighted consensus with supervised consensus learners.
- Support both target-specific prioritization and broader multi-strain prioritization across furin, FASN, influenza A PA, and influenza A NA.
- Keep the system extensible to additional target classes, feature families, and future rounds of real experimental data.

## Target scope

The target scope includes four target classes and eight constituent targets:

- `furin`
- `fasn`
- `pa_ph1n1`
- `pa_h3n2`
- `pa_h5n1`
- `na_ph1n1`
- `na_h3n2`
- `na_h5n1`

Furin and FASN use the single-target workflow. PA and NA use the pooled target-family workflow with per-strain hierarchical features.

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

## Running the learner

`run_consensus_learner.py` uses YAML run configs by default:

- shared defaults from `configs/defaults.yml`
- reusable feature-ablation presets from `configs/comparison_presets/`
- named run batches from `configs/runs/`

The merge precedence is:

1. `configs/defaults.yml`
2. run-level `settings` in the selected run YAML
3. job-level `settings` in the selected run YAML

The no-argument command defaults to `configs/runs/current.yml`, which reproduces the current pooled PA learner run:

```bash
python run_consensus_learner.py
```

To inspect the concrete jobs without fitting models, use dry-run mode:

```bash
python run_consensus_learner.py --config configs/runs/current.yml --dry-run
python run_consensus_learner.py --config configs/runs/furin_single.yml --dry-run
python run_consensus_learner.py --config configs/runs/round_synthetic.yml --dry-run
```

### Run configs

Single-target jobs use `target_id` or `targets`:

```yaml
jobs:
  - name: furin_single
    dataset_mode: single_target
    target_id: furin
    comparison_preset: single_target_standard
```

Target-family jobs use `family_key` or `families`:

```yaml
jobs:
  - name: pa_family
    dataset_mode: target_family
    family_key: pa
    comparison_preset: family_hierarchical
```

Batch jobs can expand arrays without creating one YAML file per target/preset combination:

```yaml
jobs:
  - name: single_targets
    dataset_mode: single_target
    targets: [furin, fasn]
    comparison_preset: single_target_standard

  - name: pooled_families
    dataset_mode: target_family
    families: [pa, na]
    comparison_preset: family_hierarchical
```

Each executed job saves its fully resolved config beside the results:

```text
results/<train_round_id>/configs/<output_label>_resolved_config.yml
```

### Output paths

Evaluation outputs are written under:

- `results/<round_id>/evaluation/`

Inference outputs are written under:

- `results/<round_id>/inference/`

Examples:

- single-target `furin` run:
  - `results/round_synthetic/inference/furin_6predictor_pr_ridge_inference.csv`
- pooled PA run:
  - `results/round_synthetic/inference/pa_6predictor_pr_plus_hxnx_hierarchical_tminus1_pr_ridge_inference.csv`
  - `results/round_synthetic/inference/pa_merged_inference_predictions.csv`

### Mode notes

- In `target_family` mode, `TARGET_ID` is ignored and filenames are keyed by `FAMILY_KEY`.
- In `single_target` mode, `FAMILY_KEY` is ignored and filenames are keyed by `TARGET_ID`.
- If old output files from earlier runs are still present, they are not automatically deleted.

See [data/README.md](/Users/charmainechia/Documents/projects/FluMolScreen/data/README.md) for the round-based data conventions.
