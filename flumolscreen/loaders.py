"""Data loading utilities for FluMolScreen."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flumolscreen.feature_registry import FEATURE_REGISTRY
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
        return df.copy()

    requested = list(dict.fromkeys(required_columns + selected_columns))
    missing_requested = [col for col in requested if col not in df.columns]
    if missing_requested:
        raise ValueError(
            f"{table_name} is missing requested column(s): {missing_requested}"
        )

    return df.loc[:, requested].copy()


def _feature_file_name(target_id: str, feature_set: str) -> str:
    return f"{target_id}_{feature_set}.csv"


def _feature_table_columns(feature_set: str, columns: list[str] | None) -> list[str]:
    feature_spec = FEATURE_REGISTRY[feature_set]
    join_keys = feature_spec["join_keys"]
    default_columns = feature_spec["default_columns"]
    requested = default_columns if columns is None else columns
    return list(dict.fromkeys(join_keys + requested))


def load_target_registry() -> dict:
    return TARGET_REGISTRY


def load_feature_registry() -> dict:
    return FEATURE_REGISTRY


def load_assay_data(
    data_dir: Path | str,
    round_id: str,
    target_id: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if target_id not in TARGET_REGISTRY:
        raise ValueError(f"Unknown target_id: {target_id}")

    path = _normalize_data_dir(data_dir) / round_id / "assay_data" / f"{target_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Assay data file not found: {path}")

    df = pd.read_csv(path)
    return _select_columns(
        df=df,
        required_columns=ASSAY_DATA_REQUIRED_COLUMNS,
        selected_columns=columns,
        table_name=str(path),
    )


def load_static_feature_table(
    data_dir: Path | str,
    target_id: str,
    feature_set: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if target_id not in TARGET_REGISTRY:
        raise ValueError(f"Unknown target_id: {target_id}")
    if feature_set not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    path = _normalize_data_dir(data_dir) / "static_features" / _feature_file_name(
        target_id, feature_set
    )
    if not path.exists():
        raise FileNotFoundError(f"Static feature file not found: {path}")

    df = pd.read_csv(path)
    required_columns = _feature_table_columns(feature_set, columns)
    return _select_columns(
        df=df,
        required_columns=required_columns,
        selected_columns=columns,
        table_name=str(path),
    )


def load_round_feature_table(
    data_dir: Path | str,
    round_id: str,
    target_id: str,
    feature_set: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if target_id not in TARGET_REGISTRY:
        raise ValueError(f"Unknown target_id: {target_id}")
    if feature_set not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    path = (
        _normalize_data_dir(data_dir)
        / round_id
        / "features"
        / _feature_file_name(target_id, feature_set)
    )
    if not path.exists():
        raise FileNotFoundError(f"Round-specific feature file not found: {path}")

    df = pd.read_csv(path)
    required_columns = _feature_table_columns(feature_set, columns)
    return _select_columns(
        df=df,
        required_columns=required_columns,
        selected_columns=columns,
        table_name=str(path),
    )


def load_feature_table(
    data_dir: Path | str,
    target_id: str,
    feature_set: str,
    round_id: str | None = None,
    source: str = "auto",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if source not in {"auto", "static", "round"}:
        raise ValueError("source must be one of: 'auto', 'static', 'round'")

    if source == "static":
        return load_static_feature_table(data_dir, target_id, feature_set, columns)

    if source == "round":
        if round_id is None:
            raise ValueError("round_id is required when source='round'")
        return load_round_feature_table(
            data_dir, round_id, target_id, feature_set, columns
        )

    if round_id is not None:
        try:
            return load_round_feature_table(
                data_dir, round_id, target_id, feature_set, columns
            )
        except FileNotFoundError:
            pass

    return load_static_feature_table(data_dir, target_id, feature_set, columns)


def load_dataset(
    data_dir: Path | str,
    round_id: str,
    dataset_name: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    path = _normalize_data_dir(data_dir) / round_id / "datasets" / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    return _select_columns(
        df=df,
        required_columns=DATASET_REQUIRED_COLUMNS,
        selected_columns=columns,
        table_name=str(path),
    )
