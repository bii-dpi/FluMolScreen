"""Derived summary features from real method percentile ranks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flumolscreen.feature_registry import (
    BRANCH_METHODS,
    METHOD_RANK_COLUMNS,
    METHOD_RANK_SUMMARY_COLUMNS,
)

KEY_COLUMNS = ["id", "target"]


def validate_method_rank_columns(df: pd.DataFrame) -> None:
    """Validate that a method-rank table has the required base columns."""
    required_columns = [*KEY_COLUMNS, *METHOD_RANK_COLUMNS]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"method_ranks table is missing required column(s): {missing_columns}"
        )


def build_method_rank_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build branch, consensus, and disagreement features from method ranks."""
    validate_method_rank_columns(df)
    out = df.loc[:, KEY_COLUMNS].copy()

    branch_columns = []
    for branch_name, methods in BRANCH_METHODS.items():
        branch_column = f"{branch_name}_rank_mean"
        out[branch_column] = df.loc[
            :,
            [f"{method}_rank" for method in methods],
        ].mean(axis=1)
        branch_columns.append(branch_column)

    out["consensus_123_rank"] = (
        2.0 * df["glide-sp_rank"]
        + 2.0 * df["pignet2_rank"]
        + 3.0 * df["ligunity_rank"]
        + 3.0 * df["boltz-2_rank"]
        + df["balm_rank"]
        + df["mammal_rank"]
    ) / 12.0
    out["method_rank_sd"] = df.loc[:, METHOD_RANK_COLUMNS].std(axis=1, ddof=1)
    out["method_rank_range"] = (
        df.loc[:, METHOD_RANK_COLUMNS].max(axis=1)
        - df.loc[:, METHOD_RANK_COLUMNS].min(axis=1)
    )
    out["branch_rank_sd"] = out.loc[:, branch_columns].std(axis=1, ddof=1)
    out["structure_minus_pose_rank"] = (
        out["structure_rank_mean"] - out["pose_rank_mean"]
    )
    out["sequence_minus_structure_rank"] = (
        out["sequence_rank_mean"] - out["structure_rank_mean"]
    )
    return out.loc[:, [*KEY_COLUMNS, *METHOD_RANK_SUMMARY_COLUMNS]]


def write_method_rank_summary_features(
    input_path: Path | str,
    output_path: Path | str,
) -> Path:
    """Read method ranks, derive summaries, and write them to disk."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    input_df = pd.read_csv(input_path)
    derived_df = build_method_rank_summary_features(input_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    derived_df.to_csv(output_path, index=False)
    return output_path
