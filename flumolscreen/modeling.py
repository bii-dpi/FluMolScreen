"""Minimal modeling utilities for FluMolScreen."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge


NON_FEATURE_COLUMNS = {
    "compound_id",
    "target_id",
    "target_class",
    "strain",
    "isomeric_smiles",
    "label_source",
    "round_id",
}


def select_model_feature_columns(
    df: pd.DataFrame,
    label_column: str = "label_pkd",
) -> list[str]:
    return [
        col
        for col in df.columns
        if col != label_column and col not in NON_FEATURE_COLUMNS
    ]


def fit_regression_model(
    training_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = "label_pkd",
    alpha: float = 1.0,
) -> tuple[Ridge, list[str]]:
    if label_column not in training_df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    selected_feature_columns = (
        select_model_feature_columns(training_df, label_column)
        if feature_columns is None
        else feature_columns
    )
    X = training_df.loc[:, selected_feature_columns]
    y = training_df[label_column]

    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model, selected_feature_columns


def predict_regression_model(
    model: Ridge,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    predictions = model.predict(df.loc[:, feature_columns])
    return pd.Series(predictions, index=df.index, name="prediction")
