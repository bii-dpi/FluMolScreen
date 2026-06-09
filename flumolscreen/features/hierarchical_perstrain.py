"""Target-context feature builders for pooled target-class datasets."""

from __future__ import annotations

import pandas as pd

REQUIRED_KEY_COLUMNS = ["id", "target"]

__all__ = [
    "build_tminus1_indicator_columns",
    "build_tminus1_method_interactions",
    "build_target_context_features",
]


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataframe is missing required column(s): {missing_columns}")


def _validate_reference_target(df: pd.DataFrame, reference_target: str) -> None:
    available_targets = df["target"].drop_duplicates().tolist()
    if reference_target not in set(available_targets):
        raise ValueError(
            "reference_target must exist in the dataframe. "
            f"Got {reference_target!r}; available targets are {available_targets}."
        )


def _get_non_reference_targets(
    df: pd.DataFrame,
    reference_target: str,
) -> list[str]:
    targets = df["target"].drop_duplicates().tolist()
    return [target for target in targets if target != reference_target]


def _get_indicator_column_name(target_label: str) -> str:
    return f"is_{target_label}"


def _validate_target_to_label(
    df: pd.DataFrame,
    target_to_label: dict[str, str],
) -> None:
    pooled_targets = df["target"].drop_duplicates().tolist()
    missing_targets = [
        target for target in pooled_targets if target not in target_to_label
    ]
    if missing_targets:
        raise ValueError(f"target_to_label is missing targets: {missing_targets}")

    labels = [target_to_label[target] for target in pooled_targets]
    if len(set(labels)) != len(labels):
        raise ValueError("target_to_label must map targets to unique short labels.")


def build_tminus1_indicator_columns(
    df: pd.DataFrame,
    reference_target: str,
    target_to_label: dict[str, str],
) -> pd.DataFrame:
    """Build non-reference target indicator columns."""
    _validate_required_columns(df, REQUIRED_KEY_COLUMNS)
    _validate_reference_target(df, reference_target)
    _validate_target_to_label(df, target_to_label)

    indicator_df = df.loc[:, REQUIRED_KEY_COLUMNS].copy()
    for target in _get_non_reference_targets(df, reference_target):
        indicator_column = _get_indicator_column_name(target_to_label[target])
        indicator_df[indicator_column] = (df["target"] == target).astype(int)
    return indicator_df


def build_tminus1_method_interactions(
    df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    method_columns: list[str],
    target_to_label: dict[str, str],
) -> pd.DataFrame:
    """Build non-reference target-by-feature interaction columns."""
    _validate_required_columns(df, [*REQUIRED_KEY_COLUMNS, *method_columns])
    _validate_required_columns(indicator_df, REQUIRED_KEY_COLUMNS)
    _validate_target_to_label(df, target_to_label)

    if indicator_df.shape[0] != df.shape[0]:
        raise ValueError("indicator_df must contain the same number of rows as df")
    if not indicator_df.loc[:, REQUIRED_KEY_COLUMNS].equals(df.loc[:, REQUIRED_KEY_COLUMNS]):
        raise ValueError("indicator_df keys must align exactly with df")

    interaction_df = df.loc[:, REQUIRED_KEY_COLUMNS].copy()
    indicator_columns = [
        column for column in indicator_df.columns if column not in REQUIRED_KEY_COLUMNS
    ]
    label_to_indicator = {
        column.removeprefix("is_"): column for column in indicator_columns
    }

    for target, target_label in target_to_label.items():
        indicator_column = label_to_indicator.get(target_label)
        if indicator_column is None:
            continue
        for method_column in method_columns:
            interaction_df[f"{method_column}_x_{target_label}"] = (
                df[method_column] * indicator_df[indicator_column]
            )
    return interaction_df


def build_target_context_features(
    df: pd.DataFrame,
    reference_target: str,
    feature_columns: list[str],
    target_to_label: dict[str, str],
) -> pd.DataFrame:
    """Build T-1 target indicators and target-by-feature interactions."""
    indicator_df = build_tminus1_indicator_columns(
        df=df,
        reference_target=reference_target,
        target_to_label=target_to_label,
    )
    interaction_df = build_tminus1_method_interactions(
        df=df,
        indicator_df=indicator_df,
        method_columns=feature_columns,
        target_to_label=target_to_label,
    )
    return indicator_df.merge(interaction_df, on=REQUIRED_KEY_COLUMNS, how="left")
