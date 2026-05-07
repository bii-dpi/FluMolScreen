"""Evaluation utilities for FluMolScreen."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from flumolscreen.ml.utils import make_regression_model, select_model_feature_columns

__all__ = [
    "compute_regression_metrics",
    "fit_regression_model",
    "predict_regression_model",
    "run_split_evaluation",
]


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Compute regression metrics for one fold."""
    # Use rank correlation alongside error metrics for screening relevance.
    spearman = y_true.corr(y_pred, method="spearman")
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearman": float(spearman) if pd.notna(spearman) else float("nan"),
    }


def fit_regression_model(
    training_df: pd.DataFrame,
    model_type: str,
    feature_columns: list[str] | None = None,
    label_column: str = "label_pkd",
    model_params: dict[str, Any] | None = None,
):
    """Fit a regression model and return it with the feature columns used."""
    if label_column not in training_df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    # Resolve the feature matrix once so train/test paths share the same columns.
    used_feature_columns = feature_columns or select_model_feature_columns(
        training_df,
        label_column,
    )
    X = training_df.loc[:, used_feature_columns]
    y = training_df[label_column]

    # Build the configured learner and fit it on the labeled rows.
    model = make_regression_model(model_type, model_params=model_params)
    model.fit(X, y)
    return model, used_feature_columns


def predict_regression_model(
    model,
    df: pd.DataFrame,
    feature_columns: list[str],
    return_uncertainty: bool = False,
) -> pd.Series | pd.DataFrame:
    """Generate predictions for a dataframe using the selected feature columns."""
    # Reuse the fitted feature list so prediction matrices stay aligned.
    X = df.loc[:, feature_columns]
    if not return_uncertainty:
        predictions = model.predict(X)
        return pd.Series(predictions, index=df.index, name="prediction")

    # Prefer model-native predictive uncertainty when the estimator exposes it.
    try:
        prediction_mean, prediction_err = model.predict(X, return_std=True)
    except TypeError:
        prediction_mean = model.predict(X)
        prediction_err = None

    prediction_df = pd.DataFrame(
        {"prediction_mean": prediction_mean},
        index=df.index,
    )
    if prediction_err is not None:
        prediction_df["prediction_err"] = prediction_err
    return prediction_df


def run_split_evaluation(
    training_df: pd.DataFrame,
    splits: list[tuple],
    model_type: str,
    feature_columns: list[str] | None = None,
    label_column: str = "label_pkd",
    model_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one model across all provided splits.

    Returns:
    - fold-level metrics dataframe
    - mean summary dataframe with one row
    """
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Materialize the fold-specific train/test partition.
        train_df = training_df.iloc[train_idx]
        test_df = training_df.iloc[test_idx]

        model, used_feature_columns = fit_regression_model(
            training_df=train_df,
            model_type=model_type,
            feature_columns=feature_columns,
            label_column=label_column,
            model_params=model_params,
        )
        # Score the held-out fold with the fitted model.
        predictions = predict_regression_model(
            model=model,
            df=test_df,
            feature_columns=used_feature_columns,
        )
        metrics = compute_regression_metrics(
            y_true=test_df[label_column],
            y_pred=predictions,
        )
        rows.append(
            {
                "fold": fold_idx,
                "n_train": len(train_df),
                "n_test": len(test_df),
                **metrics,
            }
        )

    # Return both fold-level results and a simple average summary.
    fold_df = pd.DataFrame(rows)
    summary_df = (
        fold_df.drop(columns=["fold"])
        .mean(numeric_only=True)
        .to_frame()
        .T
    )
    summary_df.insert(0, "summary", "mean_across_folds")
    return fold_df, summary_df
