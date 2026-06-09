# FluMolScreen

FluMolScreen is a modular modeling workspace for prioritizing compounds against
influenza-relevant host and viral targets using source method predictions and
learned consensus models.

## Vocabulary

The workflow uses source-native data columns:

- `id`: compound identifier
- `target_class`: broad target group, such as `pa`, `na`, `furin`, or `fasn`
- `target`: concrete model target, such as `pa_h3n2`, `na_h5n1`, or `fasn`
- `smiles`: standardized compound SMILES

The current target classes are `furin`, `fasn`, `pa`, and `na`. PA and NA each
contain three strain targets: `ph1n1`, `h3n2`, and `h5n1`.

## Feature families

Canonical feature inputs are the six source prediction files in
`data/shared/features/` plus the compound maps in `data/shared/datasets/`.
Target-scoped feature CSVs are not required.

Generated feature families:

- `method_scores`: six raw method outputs, such as `glide-sp_score`
- `method_ranks`: per-target percentile ranks from real method outputs
- `method_rank_summary`: branch means, consensus, and disagreement features from ranks
- `glide_uncertainty`: Glide-SP geometric uncertainty, `glide-sp_glidescore_sd`
- `chemical_descriptors`: RDKit descriptors generated from `smiles`
- `target_context`: pooled target-class indicators and target-by-feature interactions

Raw method scores are the primary binding covariates. Rank-derived summaries are
computed only from `method_ranks`, not from raw scores, because the direct method
outputs are on different scales.

## Modeling workflow

`run_consensus_learner.py` loads YAML run configs, rebuilds model-ready datasets
from source files, runs cross-validated model comparisons, and saves evaluation
and inference outputs under `results/<round_id>/`.

The workflow aligns every comparison in a job to the same complete-case row
universe before making CV splits. This keeps feature-set comparisons fair even
when one comparison uses only `chemical_descriptors`.

Current model comparison uses:

- outer random k-fold CV for evaluation
- optional nested or holdout tuning
- Ridge and XGBoost model entries from `configs/defaults.yml`
- adaptive conformal inference by default

XGBoost and RDKit are optional runtime dependencies until their corresponding
model or feature family is requested. Install the project environment from
`consensus.yml` for full workflow execution.

## Running

Inspect resolved jobs without fitting models:

```bash
python run_consensus_learner.py --config configs/runs/current.yml --dry-run
python run_consensus_learner.py --config configs/runs/furin_single.yml --dry-run
python run_consensus_learner.py --config configs/runs/round_synthetic.yml --dry-run
```

Run the default PA target-class workflow:

```bash
python run_consensus_learner.py
```

## Run configs

Single-target jobs use `target` or `targets`:

```yaml
jobs:
  - name: furin_single
    dataset_mode: single_target
    target: furin
    comparison_preset: single_target_standard
```

Pooled target-class jobs use `target_class` or `target_classes`:

```yaml
jobs:
  - name: pa_current
    dataset_mode: target_class
    target_class: pa
    comparison_preset: family_hierarchical
```

Batch jobs can expand multiple targets and target classes:

```yaml
jobs:
  - name: single_targets
    dataset_mode: single_target
    targets: [furin, fasn]
    comparison_preset: single_target_standard

  - name: pooled_target_classes
    dataset_mode: target_class
    target_classes: [pa, na]
    comparison_preset: family_hierarchical
```

Each executed job saves its resolved config beside outputs:

```text
results/<train_round_id>/configs/<output_label>_resolved_config.yml
```

## Data

See [data/README.md](data/README.md) for source file conventions and round-based
label storage.
