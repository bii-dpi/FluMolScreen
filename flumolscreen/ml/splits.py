"""Dataset split utilities for model evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, train_test_split

__all__ = [
    "make_bootstrap_sample_indices",
    "make_group_kfold_splits",
    "make_random_holdout_split",
    "make_holdout_splits_by_column",
    "make_random_kfold_splits",
    "make_splits",
]


def make_random_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create random K-fold train/test splits."""
    splitter = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    indices = np.arange(len(df))
    return list(splitter.split(indices))


def make_group_kfold_splits(
    df: pd.DataFrame,
    group_column: str,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create GroupKFold train/test splits."""
    if group_column not in df.columns:
        raise ValueError(f"Group column not found: {group_column}")

    splitter = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(df))
    groups = df[group_column].to_numpy()
    return list(splitter.split(indices, groups=groups))


def make_random_holdout_split(
    df: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_state: int = 43,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create one random train/validation split for holdout tuning."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=random_state,
        shuffle=True,
    )
    return [(np.sort(train_idx), np.sort(val_idx))]


def make_bootstrap_sample_indices(
    df: pd.DataFrame,
    n_bootstrap: int,
    random_state: int = 42,
) -> list[np.ndarray]:
    """Create bootstrap training-index samples for ensemble fitting."""
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if df.empty:
        raise ValueError("Cannot bootstrap an empty dataframe")

    # Sample row indices with replacement so each model sees a reweighted dataset.
    rng = np.random.default_rng(random_state)
    n_rows = len(df)
    return [
        rng.choice(n_rows, size=n_rows, replace=True)
        for _ in range(n_bootstrap)
    ]


def make_holdout_splits_by_column(
    df: pd.DataFrame,
    holdout_column: str,
    holdout_values: Iterable,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create a single train/test split by holding out specified values."""
    if holdout_column not in df.columns:
        raise ValueError(f"Holdout column not found: {holdout_column}")

    holdout_values = set(holdout_values)
    test_mask = df[holdout_column].isin(holdout_values).to_numpy()
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    return [(train_idx, test_idx)]


def make_splits(
    df: pd.DataFrame,
    split_type: str,
    split_params: dict | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Dispatch to the configured split strategy."""
    split_params = split_params or {}

    if split_type == "random_kfold":
        return make_random_kfold_splits(df=df, **split_params)
    if split_type == "group_kfold":
        return make_group_kfold_splits(df=df, **split_params)
    if split_type == "random_holdout":
        return make_random_holdout_split(df=df, **split_params)
    if split_type == "holdout_by_column":
        return make_holdout_splits_by_column(df=df, **split_params)

    raise ValueError(f"Unsupported split_type: {split_type}")
