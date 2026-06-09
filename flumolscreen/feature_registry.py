"""Canonical feature registry and method metadata for FluMolScreen."""

from __future__ import annotations

from flumolscreen.features.chemical_descriptors import DESCRIPTOR_COLUMNS

METHODS = [
    "glide-sp",
    "pignet2",
    "ligunity",
    "boltz-2",
    "balm",
    "mammal",
]

METHOD_PREDICTION_FILES = {
    method: f"{method}_predictions.csv"
    for method in METHODS
}

METHOD_SCORE_COLUMNS = [f"{method}_score" for method in METHODS]
METHOD_RANK_COLUMNS = [f"{method}_rank" for method in METHODS]

GLIDE_UNCERTAINTY_COLUMN = "glide-sp_glidescore_sd"

BRANCH_METHODS = {
    "sequence": ["balm", "mammal"],
    "pose": ["glide-sp", "pignet2"],
    "structure": ["ligunity", "boltz-2"],
}

METHOD_RANK_SUMMARY_COLUMNS = [
    "sequence_rank_mean",
    "pose_rank_mean",
    "structure_rank_mean",
    "consensus_123_rank",
    "method_rank_sd",
    "method_rank_range",
    "branch_rank_sd",
    "structure_minus_pose_rank",
    "sequence_minus_structure_rank",
]

FEATURE_REGISTRY = {
    "method_scores": {
        "join_keys": ["id", "target"],
        "default_columns": METHOD_SCORE_COLUMNS,
    },
    "method_ranks": {
        "join_keys": ["id", "target"],
        "default_columns": METHOD_RANK_COLUMNS,
    },
    "method_rank_summary": {
        "join_keys": ["id", "target"],
        "default_columns": METHOD_RANK_SUMMARY_COLUMNS,
    },
    "glide_uncertainty": {
        "join_keys": ["id", "target"],
        "default_columns": [GLIDE_UNCERTAINTY_COLUMN],
    },
    "chemical_descriptors": {
        "join_keys": ["id", "target"],
        "default_columns": DESCRIPTOR_COLUMNS,
    },
    "target_context": {
        "join_keys": ["id", "target"],
        "default_columns": [],
    },
}
