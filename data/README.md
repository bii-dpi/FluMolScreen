## Data layout

This directory stores source-native modeling inputs. The canonical column names
are:

- `id`: compound identifier
- `target_class`: broad target group, such as `pa`, `na`, `furin`, or `fasn`
- `target`: concrete model target, such as `pa_h3n2`, `na_h5n1`, or `fasn`
- `smiles`: standardized compound SMILES

## Canonical sources

### `registry.csv`

Canonical target metadata with `target,target_class,strain`.

### `shared/features/`

This folder should contain only source prediction files:

- `glide-sp_predictions.csv`
- `pignet2_predictions.csv`
- `ligunity_predictions.csv`
- `boltz-2_predictions.csv`
- `balm_predictions.csv`
- `mammal_predictions.csv`

Each file uses `id,target,prediction`. `glide-sp_predictions.csv` also carries
`glidescore_sd`, which is exposed as the `glide_uncertainty` feature family.

### `shared/datasets/`

This folder holds target-class compound maps:

- `compounds_furin_standardized.csv`
- `compounds_fasn_standardized.csv`
- `compounds_pa_standardized.csv`
- `compounds_na_standardized.csv`

Each compound map uses `id,smiles,source`. The `source` column is retained in the
raw map but is not used as a model feature.

### `round_synthetic/assay_data/`

Synthetic label-bearing assay tables. These are the only synthetic modeling
inputs. They use:

```text
id,target,target_class,strain,smiles,label_pkd,label_source,round_id
```

The assay tables do not contain method outputs; all features are derived from
the shared prediction and compound CSVs.

## Generated feature families

Feature families are generated on demand rather than stored as target-scoped
CSV artifacts:

- `method_scores`: six raw method outputs, such as `glide-sp_score`
- `method_ranks`: per-target percentile ranks from real method outputs
- `method_rank_summary`: branch means, consensus, and disagreement features from ranks
- `glide_uncertainty`: `glide-sp_glidescore_sd`
- `chemical_descriptors`: RDKit descriptors generated from `smiles`
- `target_context`: pooled target-class indicators and target-by-feature interactions

Method-derived feature families use complete cases across the six methods. The
learner additionally aligns all comparisons in a run to a shared row universe so
feature-set comparisons are made on the same compounds.

## Round-specific folders

At each real round `k`, labels should be placed under:

```text
data/round_k/assay_data/
```

The workflow trains on labels from the requested round and builds inference
tables from `shared/features/` and `shared/datasets/`. Round-specific feature
folders should be used only for genuinely round-specific recalculations, not for
copies of unchanged shared predictions.
