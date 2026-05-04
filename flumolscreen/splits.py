"""Dataset split utilities for model evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


def make_random_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
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
    if group_column not in df.columns:
        raise ValueError(f"Group column not found: {group_column}")

    splitter = GroupKFold(n_splits=n_splits)
    indices = np.arange(len(df))
    groups = df[group_column].to_numpy()
    return list(splitter.split(indices, groups=groups))


def make_holdout_splits_by_column(
    df: pd.DataFrame,
    holdout_column: str,
    holdout_values: Iterable,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if holdout_column not in df.columns:
        raise ValueError(f"Holdout column not found: {holdout_column}")

    holdout_values = set(holdout_values)
    test_mask = df[holdout_column].isin(holdout_values).to_numpy()
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    return [(train_idx, test_idx)]
