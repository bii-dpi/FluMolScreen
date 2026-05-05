"""Canonical feature registry for FluMolScreen."""

FEATURE_REGISTRY = {
    "6predictor": {
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
    "6predictor_derived": {
        "join_keys": ["compound_id", "target_id"],
        "default_columns": [
            "branch_sequence",
            "branch_pose",
            "branch_structure",
            "consensus_123",
            "disagreement_sd",
            "disagreement_range",
            "disagreement_branch_sd",
            "structure_minus_pose",
            "sequence_minus_structure",
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
