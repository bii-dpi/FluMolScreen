"""Derived feature extraction from six-predictor feature tables.

This module derives the same branch-level and disagreement features from either:
- ``6predictor_pr`` percentile-rank inputs
- ``6predictor_sc`` normalized score inputs

The output columns carry the same ``_pr`` or ``_sc`` suffix so the two derived
feature families can coexist without column-name collisions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PREDICTOR_METHODS = [
    "glidesp",
    "pignet2",
    "ligunity",
    "boltz2",
    "balm",
    "mammal",
]

BRANCH_NAMES = [
    "branch_sequence",
    "branch_pose",
    "branch_structure",
]

DERIVED_FEATURE_NAMES = [
    *BRANCH_NAMES,
    "consensus_123",
    "disagreement_sd",
    "disagreement_range",
    "disagreement_branch_sd",
    "structure_minus_pose",
    "sequence_minus_structure",
]


def get_predictor_columns(value_suffix: str) -> list[str]:
    """Return the six base predictor columns for a given value suffix."""
    return [f"{method}_{value_suffix}" for method in PREDICTOR_METHODS]


def get_branch_columns(value_suffix: str) -> list[str]:
    """Return the three branch-summary feature columns for a given value suffix."""
    return [f"{name}_{value_suffix}" for name in BRANCH_NAMES]


def get_derived_columns(value_suffix: str) -> list[str]:
    """Return all derived feature columns for a given value suffix."""
    return [f"{name}_{value_suffix}" for name in DERIVED_FEATURE_NAMES]


def validate_6predictor_columns(df: pd.DataFrame, value_suffix: str) -> None:
    """Validate that a six-predictor table has the required base columns."""
    required_columns = ["compound_id", "target_id", *get_predictor_columns(value_suffix)]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"6predictor_{value_suffix} table is missing required column(s): "
            f"{missing_columns}"
        )


def initialize_derived_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create the output frame carrying only join keys before derived features are added."""
    return df.loc[:, ["compound_id", "target_id"]].copy()


def compute_branch_features(df: pd.DataFrame, value_suffix: str) -> pd.DataFrame:
    """Compute the three branch-level summary features.

    Branch definitions follow the current project grouping:
    - sequence: BALM + MAMMAL
    - pose: Glide-SP + PIGNet2
    - structure: LigUnity + Boltz-2

    Each branch feature is the mean within that branch, using either percentile
    ranks or normalized scores depending on ``value_suffix``.
    """
    branch_df = pd.DataFrame(index=df.index)
    branch_df[f"branch_sequence_{value_suffix}"] = (
        df[f"balm_{value_suffix}"] + df[f"mammal_{value_suffix}"]
    ) / 2.0
    branch_df[f"branch_pose_{value_suffix}"] = (
        df[f"glidesp_{value_suffix}"] + df[f"pignet2_{value_suffix}"]
    ) / 2.0
    branch_df[f"branch_structure_{value_suffix}"] = (
        df[f"ligunity_{value_suffix}"] + df[f"boltz2_{value_suffix}"]
    ) / 2.0
    return branch_df


def compute_consensus_123(df: pd.DataFrame, value_suffix: str) -> pd.Series:
    """Compute the hand-weighted 1:2:3 baseline consensus.

    The weights reproduce the current project heuristic:
    - BALM, MAMMAL: 1
    - Glide-SP, PIGNet2: 2
    - LigUnity, Boltz-2: 3
    """
    return (
        2.0 * df[f"glidesp_{value_suffix}"]
        + 2.0 * df[f"pignet2_{value_suffix}"]
        + 3.0 * df[f"ligunity_{value_suffix}"]
        + 3.0 * df[f"boltz2_{value_suffix}"]
        + 1.0 * df[f"balm_{value_suffix}"]
        + 1.0 * df[f"mammal_{value_suffix}"]
    ) / 12.0


def compute_disagreement_features(
    df: pd.DataFrame,
    branch_df: pd.DataFrame,
    value_suffix: str,
) -> pd.DataFrame:
    """Compute disagreement and branch-difference features.

    These features expose both:
    - how much the six methods disagree overall
    - which branches are more favorable relative to others
    """
    disagreement_df = pd.DataFrame(index=df.index)

    predictor_frame = df.loc[:, get_predictor_columns(value_suffix)]
    branch_columns = get_branch_columns(value_suffix)

    # Overall disagreement across all six methods.
    disagreement_df[f"disagreement_sd_{value_suffix}"] = predictor_frame.std(
        axis=1,
        ddof=1,
    )
    disagreement_df[f"disagreement_range_{value_suffix}"] = (
        predictor_frame.max(axis=1) - predictor_frame.min(axis=1)
    )

    # Branch-level disagreement collapses the six methods into the three
    # branch summaries first, then measures disagreement across branches.
    disagreement_df[f"disagreement_branch_sd_{value_suffix}"] = branch_df.loc[
        :, branch_columns
    ].std(axis=1, ddof=1)

    # Signed branch differences preserve which branch is more favorable, not
    # just whether disagreement exists.
    disagreement_df[f"structure_minus_pose_{value_suffix}"] = (
        branch_df[f"branch_structure_{value_suffix}"]
        - branch_df[f"branch_pose_{value_suffix}"]
    )
    disagreement_df[f"sequence_minus_structure_{value_suffix}"] = (
        branch_df[f"branch_sequence_{value_suffix}"]
        - branch_df[f"branch_structure_{value_suffix}"]
    )
    return disagreement_df


def build_derived_6predictor_features(
    df: pd.DataFrame,
    value_suffix: str,
) -> pd.DataFrame:
    """Build the full derived feature family for ``6predictor_pr`` or ``6predictor_sc``."""
    validate_6predictor_columns(df, value_suffix)

    out = initialize_derived_feature_frame(df)
    branch_df = compute_branch_features(df, value_suffix)
    disagreement_df = compute_disagreement_features(df, branch_df, value_suffix)

    out = pd.concat(
        [
            out,
            branch_df,
            compute_consensus_123(df, value_suffix).rename(
                f"consensus_123_{value_suffix}"
            ),
            disagreement_df,
        ],
        axis=1,
    )
    return out.loc[:, ["compound_id", "target_id", *get_derived_columns(value_suffix)]]


def build_derived_6predictor_pr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full derived feature family for ``6predictor_pr`` inputs."""
    return build_derived_6predictor_features(df, value_suffix="pr")


def build_derived_6predictor_sc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full derived feature family for ``6predictor_sc`` inputs."""
    return build_derived_6predictor_features(df, value_suffix="sc")


def write_derived_6predictor_features(
    input_path: Path | str,
    output_path: Path | str,
    value_suffix: str,
) -> Path:
    """Read a six-predictor CSV, derive the feature block, and write it to disk."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    input_df = pd.read_csv(input_path)
    derived_df = build_derived_6predictor_features(input_df, value_suffix=value_suffix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    derived_df.to_csv(output_path, index=False)
    return output_path


def write_derived_6predictor_pr_features(
    input_path: Path | str,
    output_path: Path | str,
) -> Path:
    """Read a ``6predictor_pr`` CSV, derive the feature block, and write it to disk."""
    return write_derived_6predictor_features(
        input_path=input_path,
        output_path=output_path,
        value_suffix="pr",
    )


def write_derived_6predictor_sc_features(
    input_path: Path | str,
    output_path: Path | str,
) -> Path:
    """Read a ``6predictor_sc`` CSV, derive the feature block, and write it to disk."""
    return write_derived_6predictor_features(
        input_path=input_path,
        output_path=output_path,
        value_suffix="sc",
    )
