"""Schema constants used by FluMolScreen loaders."""

ASSAY_DATA_REQUIRED_COLUMNS = [
    "id",
    "target",
    "target_class",
    "strain",
    "smiles",
    "label_pkd",
    "label_source",
    "round_id",
]

DATASET_REQUIRED_COLUMNS = [
    "id",
    "target",
    "smiles",
    "label_pkd",
]
