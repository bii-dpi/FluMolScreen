"""Minimal evaluation utilities for FluMolScreen."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from flumolscreen.modeling import fit_regression_model, predict_regression_model


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def run_split_evaluation(
    training_df: pd.DataFrame,
    splits: list[tuple],
    feature_columns: list[str] | None = None,
    label_column: str = "label_pkd",
    alpha: float = 1.0,
) -> pd.DataFrame:
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_df = training_df.iloc[train_idx]
        test_df = training_df.iloc[test_idx]

        model, used_feature_columns = fit_regression_model(
            training_df=train_df,
            feature_columns=feature_columns,
            label_column=label_column,
            alpha=alpha,
        )
        predictions = predict_regression_model(
            model=model,
            df=test_df,
            feature_columns=used_feature_columns,
        )
        metrics = compute_regression_metrics(
            y_true=test_df[label_column],
            y_pred=predictions,
        )
        rows.append({"fold": fold_idx, **metrics})

    return pd.DataFrame(rows)
