"""Evaluation utilities for FluMolScreen."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from flumolscreen.ml.utils import make_regression_pipeline, select_model_feature_columns

__all__ = [
    "compute_binary_hit_labels",
    "compute_enrichment_factor",
    "compute_precision_at_n",
    "compute_regression_metrics",
    "fit_regression_model",
    "predict_regression_model",
    "run_split_evaluation",
]


def _build_ranking_df(
    y_true_binary: pd.Series,
    y_score: pd.Series,
) -> pd.DataFrame:
    """Return one score-sorted ranking dataframe for hit-based metrics."""
    if len(y_true_binary) != len(y_score):
        raise ValueError("y_true_binary and y_score must be the same length")
    return pd.DataFrame({"y_true_binary": y_true_binary, "y_score": y_score}).sort_values(
        "y_score",
        ascending=False,
    )


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    hit_threshold_pkd: float | None = None,
    enrichment_top_fractions: list[float] | None = None,
    precision_at_n_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute regression metrics for one fold."""
    # Use rank correlation alongside error metrics for screening relevance.
    spearman = y_true.corr(y_pred, method="spearman")
    metrics = {
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearman": float(spearman) if pd.notna(spearman) else float("nan"),
    }
    if hit_threshold_pkd is None:
        return metrics

    y_hit = compute_binary_hit_labels(y_true=y_true, hit_threshold_pkd=hit_threshold_pkd)
    for top_fraction in enrichment_top_fractions or []:
        metrics[_format_enrichment_metric_name(top_fraction)] = compute_enrichment_factor(
            y_true_binary=y_hit,
            y_score=y_pred,
            top_fraction=top_fraction,
        )
    for n in precision_at_n_values or []:
        metrics[f"precision_at_{int(n)}"] = compute_precision_at_n(
            y_true_binary=y_hit,
            y_score=y_pred,
            n=int(n),
        )
    return metrics


def compute_binary_hit_labels(
    y_true: pd.Series,
    hit_threshold_pkd: float,
) -> pd.Series:
    """Return a binary hit label derived from a potency threshold."""
    return (y_true >= hit_threshold_pkd).astype(int)


def _resolve_top_k(n_rows: int, top_fraction: float) -> int:
    """Return the number of rows corresponding to a top-fraction cutoff."""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be between 0 and 1")
    return max(1, int(math.ceil(n_rows * top_fraction)))


def _format_enrichment_metric_name(top_fraction: float) -> str:
    """Return a stable metric name like ef_1pct or ef_5pct."""
    pct_value = top_fraction * 100.0
    if float(pct_value).is_integer():
        pct_label = str(int(pct_value))
    else:
        pct_label = str(pct_value).replace(".", "p")
    return f"ef_{pct_label}pct"


def compute_enrichment_factor(
    y_true_binary: pd.Series,
    y_score: pd.Series,
    top_fraction: float,
) -> float:
    """Compute enrichment factor in the top-ranked fraction of a test set."""
    if len(y_true_binary) == 0:
        return float("nan")

    n_rows = len(y_true_binary)
    n_hits = int(y_true_binary.sum())
    if n_hits == 0:
        return float("nan")

    top_k = _resolve_top_k(n_rows=n_rows, top_fraction=top_fraction)
    ranking_df = _build_ranking_df(y_true_binary=y_true_binary, y_score=y_score)
    top_hits = int(ranking_df.head(top_k)["y_true_binary"].sum())
    observed_hit_rate = top_hits / top_k
    baseline_hit_rate = n_hits / n_rows
    return observed_hit_rate / baseline_hit_rate


def compute_precision_at_n(
    y_true_binary: pd.Series,
    y_score: pd.Series,
    n: int,
) -> float:
    """Compute precision among the top-N ranked rows in a test set."""
    if n <= 0:
        raise ValueError("n must be positive")
    if len(y_true_binary) == 0:
        return float("nan")

    top_k = min(int(n), len(y_true_binary))
    ranking_df = _build_ranking_df(y_true_binary=y_true_binary, y_score=y_score)
    top_hits = int(ranking_df.head(top_k)["y_true_binary"].sum())
    return top_hits / top_k


def fit_regression_model(
    training_df: pd.DataFrame,
    model_type: str,
    feature_columns: list[str] | None = None,
    label_column: str = "label_pkd",
    model_params: dict[str, Any] | None = None,
    standardize_features: bool = False,
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
    model = make_regression_pipeline(
        model_type=model_type,
        model_params=model_params,
        standardize_features=standardize_features,
    )
    model.fit(X, y)
    return model, used_feature_columns


def predict_regression_model(
    model,
    df: pd.DataFrame,
    feature_columns: list[str],
    return_uncertainty: bool = False,
) -> pd.Series | pd.DataFrame:
    """Generate predictions for a dataframe using the selected feature columns."""
    def _predict_with_pipeline_native_uncertainty(model_pipeline: Pipeline, X: pd.DataFrame):
        """Try predictive uncertainty on the final estimator after preprocessing."""
        transformed_X = X
        estimator = model_pipeline.steps[-1][1]
        for _, step in model_pipeline.steps[:-1]:
            transformed_X = step.transform(transformed_X)
        return estimator.predict(transformed_X, return_std=True)

    # Reuse the fitted feature list so prediction matrices stay aligned.
    X = df.loc[:, feature_columns]
    if not return_uncertainty:
        predictions = model.predict(X)
        return pd.Series(predictions, index=df.index, name="prediction")

    # Prefer model-native predictive uncertainty when the estimator exposes it.
    try:
        prediction_mean, prediction_err = model.predict(X, return_std=True)
    except TypeError:
        if isinstance(model, Pipeline):
            try:
                prediction_mean, prediction_err = _predict_with_pipeline_native_uncertainty(
                    model_pipeline=model,
                    X=X,
                )
            except TypeError:
                prediction_mean = model.predict(X)
                prediction_err = None
        else:
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
    standardize_features: bool = False,
    hit_threshold_pkd: float | None = None,
    enrichment_top_fractions: list[float] | None = None,
    precision_at_n_values: list[int] | None = None,
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
            standardize_features=standardize_features,
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
            hit_threshold_pkd=hit_threshold_pkd,
            enrichment_top_fractions=enrichment_top_fractions,
            precision_at_n_values=precision_at_n_values,
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
