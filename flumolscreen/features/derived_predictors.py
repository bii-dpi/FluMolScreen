"""Derived feature extraction from six-predictor feature tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PREDICTOR_COLUMNS = [
    "glidesp_pr",
    "pignet2_pr",
    "ligunity_pr",
    "boltz2_pr",
    "balm_pr",
    "mammal_pr",
]


def build_derived_6predictor_features(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["compound_id", "target_id", *PREDICTOR_COLUMNS]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"6predictor table is missing required column(s): {missing_columns}"
        )

    out = df.loc[:, ["compound_id", "target_id"]].copy()
    out["branch_sequence"] = (df["balm_pr"] + df["mammal_pr"]) / 2.0
    out["branch_pose"] = (df["glidesp_pr"] + df["pignet2_pr"]) / 2.0
    out["branch_structure"] = (df["ligunity_pr"] + df["boltz2_pr"]) / 2.0
    out["consensus_123"] = (
        2.0 * df["glidesp_pr"]
        + 2.0 * df["pignet2_pr"]
        + 3.0 * df["ligunity_pr"]
        + 3.0 * df["boltz2_pr"]
        + 1.0 * df["balm_pr"]
        + 1.0 * df["mammal_pr"]
    ) / 12.0

    predictor_frame = df.loc[:, PREDICTOR_COLUMNS]
    out["disagreement_sd"] = predictor_frame.std(axis=1, ddof=1)
    out["disagreement_range"] = predictor_frame.max(axis=1) - predictor_frame.min(axis=1)

    branch_frame = out.loc[:, ["branch_sequence", "branch_pose", "branch_structure"]]
    out["disagreement_branch_sd"] = branch_frame.std(axis=1, ddof=1)
    out["structure_minus_pose"] = out["branch_structure"] - out["branch_pose"]
    out["sequence_minus_structure"] = out["branch_sequence"] - out["branch_structure"]
    return out


def write_derived_6predictor_features(
    input_path: Path | str,
    output_path: Path | str,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = pd.read_csv(input_path)
    derived_df = build_derived_6predictor_features(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    derived_df.to_csv(output_path, index=False)
    return output_path
