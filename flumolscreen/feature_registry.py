"""Canonical feature registry for FluMolScreen."""

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
}
