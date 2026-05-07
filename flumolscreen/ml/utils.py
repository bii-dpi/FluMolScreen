"""General ML utility helpers for FluMolScreen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "NON_FEATURE_COLUMNS",
    "DISPLAY_TUNING_MODES",
    "evaluation_base_name",
    "format_model_params",
    "inference_file_name",
    "make_regression_model",
    "prepare_result_dirs",
    "round_metrics",
    "select_model_feature_columns",
]

NON_FEATURE_COLUMNS = {
    "compound_id",
    "target_id",
    "isomeric_smiles",
    "label_pkd",
}

DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "gaussian_process": {
        "alpha": 1e-6,
        "normalize_y": True,
        "random_state": 42,
    },
    "ridge": {"alpha": 1.0},
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "objective": "reg:squarederror",
    },
}

DISPLAY_TUNING_MODES = {
    None: "no_tuning",
    "holdout": "holdout_tuning",
    "nested": "nested_tuning",
}


def format_model_params(model_params: dict[str, Any] | None) -> str:
    """Format model parameters into a compact, readable string for saved outputs."""
    if not model_params:
        return "{}"

    # Round float-like values to a readable precision while leaving other types alone.
    def _format_value(value: Any) -> Any:
        if isinstance(value, float):
            return float(f"{value:.3g}")
        return value

    formatted_items = (
        f"{key!r}: {_format_value(value)!r}"
        for key, value in sorted(model_params.items())
    )
    return "{" + ", ".join(formatted_items) + "}"


def prepare_result_dirs(results_dir: str, round_id: str) -> dict[str, Path]:
    """Create and return result directories for a learner run."""
    # Keep result output locations consistent across workflows.
    round_results_dir = Path(results_dir) / round_id
    directories = {
        "evaluation": round_results_dir / "evaluation",
        "inference": round_results_dir / "inference",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def inference_file_name(target_id: str, comparison_name: str, model_type: str) -> str:
    """Build the inference filename for one comparison/model pair."""
    # Encode the target, feature comparison, and learner in one stable filename.
    return f"{target_id}_{comparison_name}_{model_type}_inference.csv"


def evaluation_base_name(
    target_id: str,
    outer_split_type: str,
    tuning_mode: str | None,
) -> str:
    """Build the base filename stem for evaluation outputs."""
    # Encode the outer CV scaffold and, when relevant, the tuning strategy.
    base_name = f"{target_id}_{outer_split_type}_cv"
    if tuning_mode is None:
        return base_name
    return f"{base_name}_{DISPLAY_TUNING_MODES.get(tuning_mode, tuning_mode)}"


def round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Round common metric columns to 4 decimals before saving."""
    # Keep saved metrics concise and consistent across result tables.
    out = df.copy()
    for column in ("rmse", "mae", "r2", "spearman", "tuning_score"):
        if column in out.columns:
            out[column] = out[column].round(4)
    return out


def select_model_feature_columns(
    df: pd.DataFrame,
    label_column: str = "label_pkd",
) -> list[str]:
    """Return the columns to use as model inputs."""
    # Treat IDs/metadata as non-features and keep only modeling columns.
    return [
        col
        for col in df.columns
        if col != label_column and col not in NON_FEATURE_COLUMNS
    ]


def make_regression_model(
    model_type: str,
    model_params: dict[str, Any] | None = None,
):
    """Instantiate a supported regression model."""
    # Normalize aliases and layer custom parameters over defaults.
    model_type = {
        "gp": "gaussian_process",
        "xgb": "xgboost",
    }.get(model_type, model_type)
    resolved_params = {**DEFAULT_MODEL_PARAMS.get(model_type, {}), **(model_params or {})}

    if model_type == "gaussian_process":
        return GaussianProcessRegressor(**resolved_params)
    if model_type == "ridge":
        return Ridge(**resolved_params)
    if model_type == "xgboost":
        return XGBRegressor(**resolved_params)

    raise ValueError(f"Unsupported model_type: {model_type}")
