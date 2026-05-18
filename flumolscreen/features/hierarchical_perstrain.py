"""Hierarchical per-strain feature builders for pooled multi-target datasets.

This module builds T-1 reference-coded indicator and interaction feature blocks
for pooled datasets where each row corresponds to one compound-target pair.

The public functions are column-agnostic: callers pass whichever base feature
columns should receive per-strain interaction expansion. Today that will usually
be the six `_pr` method columns, but the same functions can be reused later for
`_sc` or other per-target signal columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flumolscreen.feature_registry import FEATURE_REGISTRY

DEFAULT_METHOD_COLUMNS_PR = FEATURE_REGISTRY["6predictor_pr"]["default_columns"]
DEFAULT_METHOD_COLUMNS_SC = FEATURE_REGISTRY["6predictor_sc"]["default_columns"]

REQUIRED_KEY_COLUMNS = ["compound_id", "target_id"]

__all__ = [
    "DEFAULT_METHOD_COLUMNS_PR",
    "DEFAULT_METHOD_COLUMNS_SC",
    "build_tminus1_indicator_columns",
    "build_tminus1_method_interactions",
    "build_hierarchical_perstrain_features",
    "write_hierarchical_perstrain_features",
]


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise when a dataframe is missing required columns."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataframe is missing required column(s): {missing_columns}")


def _validate_reference_target_id(df: pd.DataFrame, reference_target_id: str) -> None:
    """Raise when the requested reference target does not exist in the dataframe."""
    available_target_ids = df["target_id"].drop_duplicates().tolist()
    if reference_target_id not in set(available_target_ids):
        raise ValueError(
            "reference_target_id must exist in the dataframe. "
            f"Got {reference_target_id!r}; available target_ids are {available_target_ids}."
        )


def _get_non_reference_target_ids(
    df: pd.DataFrame,
    reference_target_id: str,
) -> list[str]:
    """Return pooled target ids excluding the chosen reference target."""
    target_ids = df["target_id"].drop_duplicates().tolist()
    return [target_id for target_id in target_ids if target_id != reference_target_id]


def _get_indicator_column_name(target_label: str) -> str:
    """Return the canonical indicator column name for a non-reference target."""
    return f"is_{target_label}"


def build_tminus1_indicator_columns(
    df: pd.DataFrame,
    reference_target_id: str,
    target_id_to_label: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build non-reference target indicator columns for a pooled per-strain dataset.

    Parameters
    ----------
    df
        Pooled dataframe containing one row per compound-target pair. Must include
        ``compound_id`` and ``target_id``.
    reference_target_id
        Target id to use as the reference level. No indicator column is created
        for this target.
    target_id_to_label
        Required mapping from target_id to short label used in output column
        names.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``compound_id``, ``target_id``, and one indicator column
        per non-reference target.
    """
    _validate_required_columns(df, REQUIRED_KEY_COLUMNS)
    _validate_reference_target_id(df, reference_target_id)
    if target_id_to_label is None:
        raise ValueError("target_id_to_label must be provided for hierarchical expansion")

    indicator_df = df.loc[:, REQUIRED_KEY_COLUMNS].copy()
    non_reference_target_ids = _get_non_reference_target_ids(df, reference_target_id)

    for target_id in non_reference_target_ids:
        if target_id not in target_id_to_label:
            raise ValueError(f"Missing label mapping for target_id: {target_id}")
        indicator_column = _get_indicator_column_name(target_id_to_label[target_id])
        indicator_df[indicator_column] = (df["target_id"] == target_id).astype(int)

    return indicator_df


def build_tminus1_method_interactions(
    df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    method_columns: list[str],
    target_id_to_label: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build non-reference strain-by-feature interaction columns.

    Parameters
    ----------
    df
        Pooled dataframe containing ``compound_id``, ``target_id``, and the
        base feature columns to expand.
    indicator_df
        Output of :func:`build_tminus1_indicator_columns`.
    method_columns
        Base feature columns to expand into interaction columns.
    target_id_to_label
        Required mapping from target_id to short label used in output names.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``compound_id``, ``target_id``, and interaction columns
        only.
    """
    _validate_required_columns(df, [*REQUIRED_KEY_COLUMNS, *method_columns])
    _validate_required_columns(indicator_df, REQUIRED_KEY_COLUMNS)
    if target_id_to_label is None:
        raise ValueError("target_id_to_label must be provided for hierarchical expansion")

    if indicator_df.shape[0] != df.shape[0]:
        raise ValueError(
            "indicator_df must contain the same number of rows as df so row-wise "
            "feature expansion is well-defined."
        )

    if not indicator_df.loc[:, REQUIRED_KEY_COLUMNS].equals(df.loc[:, REQUIRED_KEY_COLUMNS]):
        raise ValueError(
            "indicator_df keys must align exactly with df keys in the same row order."
        )

    interaction_df = df.loc[:, REQUIRED_KEY_COLUMNS].copy()
    target_ids = df["target_id"].drop_duplicates().tolist()
    non_reference_target_ids = [
        target_id for target_id in target_ids if target_id in target_id_to_label
    ]

    indicator_columns = [
        column for column in indicator_df.columns if column not in REQUIRED_KEY_COLUMNS
    ]
    label_to_indicator = {column.removeprefix("is_"): column for column in indicator_columns}

    for target_id in non_reference_target_ids:
        target_label = target_id_to_label[target_id]
        indicator_column = label_to_indicator.get(target_label)
        if indicator_column is None:
            continue
        for method_column in method_columns:
            interaction_column = f"{method_column}_x_{target_label}"
            interaction_df[interaction_column] = (
                df[method_column] * indicator_df[indicator_column]
            )

    return interaction_df


def build_hierarchical_perstrain_features(
    df: pd.DataFrame,
    reference_target_id: str,
    feature_columns: list[str] | None = None,
    target_id_to_label: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the full T-1 hierarchical per-strain feature family.

    Parameters
    ----------
    df
        Pooled dataframe containing one row per compound-target pair and the base
        feature columns to expand.
    reference_target_id
        Reference target id used for T-1 encoding.
    feature_columns
        Base feature columns to expand. Defaults to ``DEFAULT_METHOD_COLUMNS_PR``.
    target_id_to_label
        Required mapping from target_id to short label used in output names.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``compound_id``, ``target_id``, non-reference indicator
        columns, and non-reference interaction columns.
    """
    feature_columns = feature_columns or DEFAULT_METHOD_COLUMNS_PR
    indicator_df = build_tminus1_indicator_columns(
        df=df,
        reference_target_id=reference_target_id,
        target_id_to_label=target_id_to_label,
    )
    interaction_df = build_tminus1_method_interactions(
        df=df,
        indicator_df=indicator_df,
        method_columns=feature_columns,
        target_id_to_label=target_id_to_label,
    )
    return indicator_df.merge(interaction_df, on=REQUIRED_KEY_COLUMNS, how="left")


def write_hierarchical_perstrain_features(
    input_path: Path | str,
    output_path: Path | str,
    reference_target_id: str,
    feature_columns: list[str] | None = None,
    target_id_to_label: dict[str, str] | None = None,
) -> Path:
    """Read a pooled CSV, build the hierarchical feature family, and write it.

    Parameters
    ----------
    input_path
        CSV path containing pooled rows and base feature columns.
    output_path
        CSV path to write the hierarchical feature family.
    reference_target_id
        Reference target id used for T-1 encoding.
    feature_columns
        Base feature columns to expand. Defaults to ``DEFAULT_METHOD_COLUMNS_PR``.
    target_id_to_label
        Required mapping from target_id to short label used in output names.

    Returns
    -------
    Path
        Resolved output path written to disk.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    input_df = pd.read_csv(input_path)
    feature_df = build_hierarchical_perstrain_features(
        df=input_df,
        reference_target_id=reference_target_id,
        feature_columns=feature_columns,
        target_id_to_label=target_id_to_label,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    return output_path
