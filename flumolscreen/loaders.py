"""Data loading utilities for FluMolScreen."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flumolscreen.features.source_features import build_feature_table
from flumolscreen.schema import ASSAY_DATA_REQUIRED_COLUMNS, DATASET_REQUIRED_COLUMNS
from flumolscreen.target_registry import TARGET_REGISTRY


def _normalize_data_dir(data_dir: Path | str) -> Path:
    return Path(data_dir)


def _select_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    selected_columns: list[str] | None,
    table_name: str,
) -> pd.DataFrame:
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"{table_name} is missing required column(s): {missing_required}"
        )

    if selected_columns is None:
        return df.loc[:, required_columns].copy()

    requested = list(dict.fromkeys(required_columns + selected_columns))
    missing_requested = [col for col in requested if col not in df.columns]
    if missing_requested:
        raise ValueError(
            f"{table_name} is missing requested column(s): {missing_requested}"
        )
    return df.loc[:, requested].copy()


def load_assay_data(
    data_dir: Path | str,
    round_id: str,
    target: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load label-bearing assay data for one target."""
    if target not in TARGET_REGISTRY:
        raise ValueError(f"Unknown target: {target}")

    path = _normalize_data_dir(data_dir) / round_id / "assay_data" / f"{target}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Assay data file not found: {path}")

    df = pd.read_csv(path)
    return _select_columns(
        df=df,
        required_columns=ASSAY_DATA_REQUIRED_COLUMNS,
        selected_columns=columns,
        table_name=str(path),
    )


def load_shared_feature_table(
    data_dir: Path | str,
    target: str,
    feature_set: str,
    columns: list[str] | None = None,
    base_feature_set: str | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a shared source-native feature table for one target."""
    return build_feature_table(
        data_dir=data_dir,
        target=target,
        feature_set=feature_set,
        columns=columns,
        base_feature_set=base_feature_set,
        feature_columns=feature_columns,
    )


def load_feature_table(
    data_dir: Path | str,
    target: str,
    feature_set: str,
    round_id: str | None = None,
    source: str = "auto",
    columns: list[str] | None = None,
    base_feature_set: str | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load or build one feature table.

    Source-native features are shared across rounds; ``round_id`` is accepted for
    call-site compatibility but is not used.
    """
    if source not in {"auto", "shared"}:
        raise ValueError("source-native feature sets support only 'auto' or 'shared'")

    return load_shared_feature_table(
        data_dir=data_dir,
        target=target,
        feature_set=feature_set,
        columns=columns,
        base_feature_set=base_feature_set,
        feature_columns=feature_columns,
    )


def load_shared_dataset(
    data_dir: Path | str,
    dataset_name: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    path = _normalize_data_dir(data_dir) / "shared" / "datasets" / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Shared dataset file not found: {path}")

    df = pd.read_csv(path)
    return _select_columns(
        df=df,
        required_columns=DATASET_REQUIRED_COLUMNS,
        selected_columns=columns,
        table_name=str(path),
    )
