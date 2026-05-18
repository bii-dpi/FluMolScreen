"""Canonical feature registry and family metadata for FluMolScreen."""

from __future__ import annotations

HIERARCHICAL_TARGET_FAMILY_REGISTRY = {
    "pa": {
        "target_stem": "pa_",
        "target_suffixes": ["ph1n1", "h3n2", "h5n1"],
        "reference_target_id_suffix": "ph1n1",
    },
    "na": {
        "target_stem": "na_",
        "target_suffixes": ["ph1n1", "h3n2", "h5n1"],
        "reference_target_id_suffix": "ph1n1",
    },
}

FEATURE_REGISTRY = {
    "6predictor_pr": {
        "join_keys": ["compound_id", "target_id"],
        "default_columns": [
            "glidesp_pr",
            "pignet2_pr",
            "ligunity_pr",
            "boltz2_pr",
            "balm_pr",
            "mammal_pr",
        ],
    },
    "6predictor_pr_derived": {
        "join_keys": ["compound_id", "target_id"],
        "default_columns": [
            "branch_sequence_pr",
            "branch_pose_pr",
            "branch_structure_pr",
            "consensus_123_pr",
            "disagreement_sd_pr",
            "disagreement_range_pr",
            "disagreement_branch_sd_pr",
            "structure_minus_pose_pr",
            "sequence_minus_structure_pr",
        ],
    },
    "6predictor_sc": {
        "join_keys": ["compound_id", "target_id"],
        "default_columns": [
            "glidesp_sc",
            "pignet2_sc",
            "ligunity_sc",
            "boltz2_sc",
            "balm_sc",
            "mammal_sc",
        ],
    },
    "6predictor_sc_derived": {
        "join_keys": ["compound_id", "target_id"],
        "default_columns": [
            "branch_sequence_sc",
            "branch_pose_sc",
            "branch_structure_sc",
            "consensus_123_sc",
            "disagreement_sd_sc",
            "disagreement_range_sc",
            "disagreement_branch_sd_sc",
            "structure_minus_pose_sc",
            "sequence_minus_structure_sc",
        ],
    },
    "chemdescriptors": {
        "join_keys": ["compound_id"],
        "default_columns": [
            "exact_mol_wt",
            "logp",
            "tpsa",
            "hbd",
            "hba",
            "formal_charge",
            "rotatable_bonds",
            "ring_count",
            "aromatic_ring_count",
            "heavy_atom_count",
            "fraction_csp3",
            "structural_alert_count",
            "qed",
        ],
    },
    "hxnx_hierarchical_tminus1_pr": {
        "join_keys": ["compound_id", "target_id"],
        "default_columns": [
            "is_h3n2",
            "is_h5n1",
            "glidesp_pr_x_h3n2",
            "glidesp_pr_x_h5n1",
            "pignet2_pr_x_h3n2",
            "pignet2_pr_x_h5n1",
            "ligunity_pr_x_h3n2",
            "ligunity_pr_x_h5n1",
            "boltz2_pr_x_h3n2",
            "boltz2_pr_x_h5n1",
            "balm_pr_x_h3n2",
            "balm_pr_x_h5n1",
            "mammal_pr_x_h3n2",
            "mammal_pr_x_h5n1",
        ],
    },
}


def resolve_hierarchical_target_ids(family_key: str) -> list[str]:
    """Return the canonical target ids for a hierarchical target family."""
    if family_key not in HIERARCHICAL_TARGET_FAMILY_REGISTRY:
        raise ValueError(f"Unknown hierarchical family key: {family_key}")

    family_spec = HIERARCHICAL_TARGET_FAMILY_REGISTRY[family_key]
    target_stem = family_spec["target_stem"]
    return [
        f"{target_stem}{target_suffix}"
        for target_suffix in family_spec["target_suffixes"]
    ]


def resolve_hierarchical_reference_target_id(family_key: str) -> str:
    """Return the canonical reference target id for a hierarchical family."""
    if family_key not in HIERARCHICAL_TARGET_FAMILY_REGISTRY:
        raise ValueError(f"Unknown hierarchical family key: {family_key}")

    family_spec = HIERARCHICAL_TARGET_FAMILY_REGISTRY[family_key]
    return f"{family_spec['target_stem']}{family_spec['reference_target_id_suffix']}"


def resolve_hierarchical_target_id_to_label(
    family_key: str,
    target_ids: list[str] | None = None,
) -> dict[str, str]:
    """Return a canonical target_id -> short label mapping for a family."""
    if family_key not in HIERARCHICAL_TARGET_FAMILY_REGISTRY:
        raise ValueError(f"Unknown hierarchical family key: {family_key}")

    family_spec = HIERARCHICAL_TARGET_FAMILY_REGISTRY[family_key]
    target_stem = family_spec["target_stem"]
    resolved_target_ids = (
        resolve_hierarchical_target_ids(family_key)
        if target_ids is None
        else target_ids
    )
    return {
        target_id: target_id.removeprefix(target_stem)
        for target_id in resolved_target_ids
    }
