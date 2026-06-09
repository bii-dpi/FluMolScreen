"""Final-model inference helpers for FluMolScreen ML workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flumolscreen.console import print_progress
from flumolscreen.ml.conformal import (
    compute_symmetric_conformal_half_width,
    compute_absolute_standardized_residuals,
    fit_symmetric_conformal_scaler,
)
from flumolscreen.ml.evaluation import fit_regression_model, predict_regression_model
from flumolscreen.ml.splits import (
    make_bootstrap_sample_indices,
    make_random_holdout_split,
)
from flumolscreen.ml.utils import (
    annotate_inference_overlap,
    inference_file_name,
    select_model_feature_columns,
)

__all__ = [
    "fit_bootstrap_ensemble",
    "fit_final_candidate_and_save_inference",
    "predict_bootstrap_ensemble",
]


def _build_inference_output_path(
    inference_dir: Path,
    target: str,
    candidate: dict,
) -> Path:
    """Return the standard output path for one candidate inference file."""
    # Keep inference filenames consistent across point and uncertainty-aware modes.
    return inference_dir / inference_file_name(
        target=target,
        comparison_name=candidate["comparison_name"],
        model_type=candidate["model_type"],
    )


def _print_inference_progress(tag: str, fields: list[tuple[str, object]]) -> None:
    """Print one inference progress line with flushing for long-running jobs."""
    print_progress(tag, fields)


def _summarize_calibration_errors(
    calibration_df: pd.DataFrame,
    calibration_predictions: pd.DataFrame,
    interval_coverage: float,
) -> tuple[pd.Series, pd.Series]:
    """Return raw and standardized calibration errors from ensemble-mean predictions."""
    # Compute raw absolute errors against the ensemble mean prediction.
    calibration_absolute_errors = (
        calibration_df["label_pkd"] - calibration_predictions["prediction_mean"]
    ).abs()

    # Normalize those errors by ensemble spread for conformal calibration.
    z_scores = compute_absolute_standardized_residuals(
        y_true=calibration_df["label_pkd"],
        prediction_mean=calibration_predictions["prediction_mean"],
        prediction_std=calibration_predictions["prediction_std"],
    )
    _print_inference_progress(
        "[inference-progress]",
        [
            ("phase", "calibration_error_stats"),
            ("raw_abs_err_median", f"{calibration_absolute_errors.median():.4f}"),
            (
                "raw_abs_err_q",
                f"{calibration_absolute_errors.quantile(interval_coverage):.4f}",
            ),
            ("standardized_err_median", f"{z_scores.median():.4f}"),
        ],
    )
    return calibration_absolute_errors, z_scores


def _build_uncertainty_inference_df(
    inference_source_df: pd.DataFrame,
    prediction_mean: pd.Series,
    prediction_err: pd.Series,
) -> pd.DataFrame:
    """Build the compact uncertainty-aware inference payload."""
    # Keep only stable IDs plus the calibrated prediction center and half-width.
    inference_df = inference_source_df.loc[:, ["id", "target"]].copy()
    inference_df["pred_mean"] = prediction_mean.round(4)
    inference_df["pred_err"] = prediction_err.round(4)
    return inference_df


def fit_bootstrap_ensemble(
    training_df: pd.DataFrame,
    model_type: str,
    model_params: dict | None,
    n_bootstrap: int,
    random_state: int = 42,
    feature_columns: list[str] | None = None,
    label_column: str = "label_pkd",
    standardize_features: bool = False,
) -> tuple[list, list[str]]:
    """Fit a bootstrap ensemble and return the models with their shared features."""
    # Resolve one shared feature list so every ensemble member uses the same inputs.
    used_feature_columns = feature_columns or select_model_feature_columns(
        training_df,
        label_column=label_column,
    )
    bootstrap_indices = make_bootstrap_sample_indices(
        df=training_df,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    # Fit one model per bootstrap-resampled training table.
    models = []
    for sample_idx in bootstrap_indices:
        bootstrap_df = training_df.iloc[sample_idx]
        model, _ = fit_regression_model(
            training_df=bootstrap_df,
            model_type=model_type,
            feature_columns=used_feature_columns,
            label_column=label_column,
            model_params=model_params,
            standardize_features=standardize_features,
        )
        models.append(model)

    return models, used_feature_columns


def predict_bootstrap_ensemble(
    models: list,
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Predict with each ensemble member and summarize the ensemble outputs."""
    if not models:
        raise ValueError("models must contain at least one fitted ensemble member")

    # Collect one prediction vector per model on the same target rows.
    prediction_df = pd.DataFrame(
        {
            f"prediction_model_{idx}": predict_regression_model(
                model=model,
                df=df,
                feature_columns=feature_columns,
            )
            for idx, model in enumerate(models)
        },
        index=df.index,
    )

    # Summarize ensemble center and spread for downstream conformal calibration.
    prediction_df["prediction_mean"] = prediction_df.mean(axis=1)
    prediction_df["prediction_std"] = prediction_df.std(axis=1, ddof=1).fillna(0.0)
    return prediction_df


def _fit_and_save_point_inference(
    inference_dir: Path,
    target: str,
    candidate: dict,
    tuned_model_params: dict | None,
) -> Path:
    """Fit one final model on all labeled data and save point predictions."""
    # Fit the final candidate on all labeled rows for the simplest inference path.
    model, feature_columns = fit_regression_model(
        training_df=candidate["training_df"],
        model_type=candidate["model_type"],
        model_params=tuned_model_params,
        standardize_features=candidate.get("standardize_features", False),
    )

    # Predict the full inference table, requesting model-native uncertainty when available.
    predictions = predict_regression_model(
        model=model,
        df=candidate["inference_df"],
        feature_columns=feature_columns,
        return_uncertainty=True,
    ).round(4)

    # Save either plain point predictions or GP-style mean/error outputs.
    inference_df = candidate["inference_df"].loc[:, ["id", "target"]].copy()
    if "prediction_err" in predictions.columns:
        inference_df["pred_mean"] = predictions["prediction_mean"]
        inference_df["pred_err"] = predictions["prediction_err"]
    else:
        inference_df["prediction"] = predictions["prediction_mean"]
    inference_df = annotate_inference_overlap(
        inference_df=inference_df,
        experimental_df=candidate["training_df"],
    )
    output_path = _build_inference_output_path(
        inference_dir=inference_dir,
        target=target,
        candidate=candidate,
    )
    inference_df.to_csv(output_path, index=False)
    return output_path


def _fit_and_save_adaptive_conformal_inference(
    inference_dir: Path,
    target: str,
    candidate: dict,
    tuned_model_params: dict | None,
    calibration_fraction: float,
    n_bootstrap: int,
    interval_coverage: float,
    random_state: int,
) -> Path:
    """Fit a bootstrap ensemble, calibrate it, and save interval-valued inference."""
    comparison_name = candidate["comparison_name"]
    model_type = candidate["model_type"]

    # Report the candidate and settings before the uncertainty pipeline begins.
    _print_inference_progress(
        "[inference-start]",
        [
            ("comparison", comparison_name),
            ("model", model_type),
            ("mode", "adaptive_conformal"),
            ("coverage", f"{interval_coverage:.2f}"),
            ("n_bootstrap", n_bootstrap),
            ("calibration_fraction", f"{calibration_fraction:.2f}"),
        ],
    )

    # Split labeled data into proper-train and calibration subsets.
    training_df = candidate["training_df"].reset_index(drop=True)
    (proper_train_idx, calibration_idx) = make_random_holdout_split(
        df=training_df,
        validation_fraction=calibration_fraction,
        random_state=random_state,
    )[0]
    proper_train_df = training_df.iloc[proper_train_idx].reset_index(drop=True)
    calibration_df = training_df.iloc[calibration_idx].reset_index(drop=True)
    _print_inference_progress(
        "[inference-progress]",
        [
            ("phase", "split_labeled_data"),
            ("n_proper_train", len(proper_train_df)),
            ("n_calibration", len(calibration_df)),
        ],
    )

    # Fit the bootstrap ensemble on proper-train only.
    _print_inference_progress(
        "[inference-progress]",
        [
            ("phase", "fit_bootstrap_ensemble"),
            ("split", "proper_train"),
            ("n_bootstrap", n_bootstrap),
        ],
    )
    models, feature_columns = fit_bootstrap_ensemble(
        training_df=proper_train_df,
        model_type=model_type,
        model_params=tuned_model_params,
        n_bootstrap=n_bootstrap,
        random_state=random_state + 1,
        standardize_features=candidate.get("standardize_features", False),
    )

    # Predict calibration rows and estimate the conformal multiplier q.
    _print_inference_progress(
        "[inference-progress]",
        [("phase", "predict_calibration_and_fit_conformal_scaler")],
    )
    calibration_predictions = predict_bootstrap_ensemble(
        models=models,
        df=calibration_df,
        feature_columns=feature_columns,
    )
    _, z_scores = _summarize_calibration_errors(
        calibration_df=calibration_df,
        calibration_predictions=calibration_predictions,
        interval_coverage=interval_coverage,
    )
    q = fit_symmetric_conformal_scaler(
        z_scores=z_scores,
        interval_coverage=interval_coverage,
    )
    _print_inference_progress(
        "[inference-progress]",
        [
            ("phase", "selected_conformal_scaler"),
            ("standardized_err_q", f"{q:.4f}"),
        ],
    )

    # Predict inference rows with the same ensemble and convert spread into error bars.
    _print_inference_progress(
        "[inference-progress]",
        [("phase", "predict_inference_and_convert_spread")],
    )
    inference_predictions = predict_bootstrap_ensemble(
        models=models,
        df=candidate["inference_df"],
        feature_columns=feature_columns,
    )
    prediction_err = compute_symmetric_conformal_half_width(
        prediction_std=inference_predictions["prediction_std"],
        q=q,
    )

    # Save the compact uncertainty-aware payload needed for ranking and triage.
    inference_df = _build_uncertainty_inference_df(
        inference_source_df=candidate["inference_df"],
        prediction_mean=inference_predictions["prediction_mean"],
        prediction_err=prediction_err,
    )
    inference_df = annotate_inference_overlap(
        inference_df=inference_df,
        experimental_df=candidate["training_df"],
    )
    output_path = _build_inference_output_path(
        inference_dir=inference_dir,
        target=target,
        candidate=candidate,
    )
    inference_df.to_csv(output_path, index=False)
    _print_inference_progress(
        "[inference-done]",
        [
            ("saved", output_path),
            ("interval", "prediction_mean +/- prediction_err"),
            ("standardized_err_q", f"{q:.4f}"),
            (
                "prediction_err_range",
                f"[{prediction_err.min():.4f}, {prediction_err.max():.4f}]",
            ),
        ],
    )
    return output_path


def fit_final_candidate_and_save_inference(
    inference_dir: Path,
    target: str,
    candidate: dict,
    tuned_model_params: dict | None,
    inference_mode: str = "point",
    calibration_fraction: float = 0.2,
    n_bootstrap: int = 10,
    interval_coverage: float = 0.9,
    random_state: int = 42,
) -> Path:
    """Fit and save either point or adaptive-conformal inference for one candidate."""
    # Dispatch to the requested inference strategy without changing the public API.
    if inference_mode == "point":
        return _fit_and_save_point_inference(
            inference_dir=inference_dir,
            target=target,
            candidate=candidate,
            tuned_model_params=tuned_model_params,
        )
    if inference_mode == "adaptive_conformal":
        return _fit_and_save_adaptive_conformal_inference(
            inference_dir=inference_dir,
            target=target,
            candidate=candidate,
            tuned_model_params=tuned_model_params,
            calibration_fraction=calibration_fraction,
            n_bootstrap=n_bootstrap,
            interval_coverage=interval_coverage,
            random_state=random_state,
        )

    raise ValueError(f"Unsupported inference_mode: {inference_mode}")
