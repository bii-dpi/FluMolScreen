"""General ML utility helpers for FluMolScreen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "NON_FEATURE_COLUMNS",
    "DISPLAY_TUNING_MODES",
    "OVERLAP_COLUMN",
    "OVERLAP_KEY_COLUMNS",
    "annotate_inference_overlap",
    "build_merged_inference_path",
    "evaluation_base_name",
    "format_model_params",
    "inference_file_name",
    "make_regression_pipeline",
    "make_regression_model",
    "merge_inference_predictions",
    "prepare_result_dirs",
    "round_metrics",
    "select_model_feature_columns",
]

NON_FEATURE_COLUMNS = {
    "id",
    "target",
    "target_class",
    "strain",
    "smiles",
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
OVERLAP_COLUMN = "in_experimental_data"
OVERLAP_KEY_COLUMNS = ["id", "target"]


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


def annotate_inference_overlap(
    inference_df: pd.DataFrame,
    experimental_df: pd.DataFrame,
    key_columns: list[str] | None = None,
    overlap_column: str = OVERLAP_COLUMN,
) -> pd.DataFrame:
    """Add a 1/0 column indicating whether inference rows appear in labeled data."""
    key_columns = key_columns or OVERLAP_KEY_COLUMNS

    # Reduce the labeled table to unique key pairs used to define row overlap.
    experimental_keys = (
        experimental_df.loc[:, key_columns]
        .drop_duplicates()
        .assign(**{overlap_column: True})
    )

    # Left-join overlap flags onto the inference table and fill missing rows as unseen.
    annotated_df = inference_df.merge(
        experimental_keys,
        on=key_columns,
        how="left",
    )
    annotated_df[overlap_column] = annotated_df[overlap_column].notna().astype(int)

    # Place the overlap flag immediately after the target/compound key metadata.
    ordered_columns = [
        *key_columns,
        overlap_column,
        *[
            column
            for column in annotated_df.columns
            if column not in {*key_columns, overlap_column}
        ],
    ]
    return annotated_df.loc[:, ordered_columns]


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


def inference_file_name(target: str, comparison_name: str, model_type: str) -> str:
    """Build the inference filename for one comparison/model pair."""
    # Encode the target, feature comparison, and learner in one stable filename.
    return f"{target}_{comparison_name}_{model_type}_inference.csv"


def build_merged_inference_path(
    inference_dir: Path,
    target: str,
) -> Path:
    """Return the standard output path for the merged inference comparison table."""
    return inference_dir / f"{target}_merged_inference_predictions.csv"


def merge_inference_predictions(
    inference_tables: dict[str, pd.DataFrame],
    key_columns: list[str] | None = None,
    overlap_column: str = OVERLAP_COLUMN,
) -> pd.DataFrame:
    """Merge per-candidate inference tables into one wide comparison dataframe."""
    if not inference_tables:
        raise ValueError("inference_tables must contain at least one table")

    def _rename_prediction_columns(df: pd.DataFrame, identifier: str) -> pd.DataFrame:
        """Rename one candidate's prediction columns into the wide-table convention."""
        return df.rename(
            columns={
                "prediction": f"pred_{identifier}",
                "pred_mean": f"pred_mean_{identifier}",
                "pred_err": f"pred_err_{identifier}",
                "prediction_mean": f"pred_mean_{identifier}",
                "prediction_err": f"pred_err_{identifier}",
            }
        )

    def _select_value_columns(df: pd.DataFrame, identifier: str) -> list[str]:
        """Return the model-specific prediction columns after renaming."""
        prefixes = (
            f"pred_{identifier}",
            f"pred_mean_{identifier}",
            f"pred_err_{identifier}",
        )
        return [column for column in df.columns if column.startswith(prefixes)]

    key_columns = key_columns or OVERLAP_KEY_COLUMNS
    iterator = iter(inference_tables.items())
    first_identifier, first_df = next(iterator)

    # Start from the shared metadata columns carried by every inference output.
    merged_df = first_df.loc[:, [*key_columns, overlap_column]].copy()

    # Add the first candidate's prediction columns using the requested model identifier.
    renamed_first_df = _rename_prediction_columns(first_df, first_identifier)
    first_value_columns = _select_value_columns(renamed_first_df, first_identifier)
    merged_df = merged_df.merge(
        renamed_first_df.loc[:, [*key_columns, *first_value_columns]],
        on=key_columns,
        how="left",
    )

    for identifier, df in iterator:
        # Rename each candidate's prediction columns into a model-specific wide format.
        renamed_df = _rename_prediction_columns(df, identifier)
        value_columns = _select_value_columns(renamed_df, identifier)
        merged_df = merged_df.merge(
            renamed_df.loc[:, [*key_columns, *value_columns]],
            on=key_columns,
            how="left",
        )

    return merged_df


def evaluation_base_name(
    target: str,
    outer_split_type: str,
    tuning_mode: str | None,
) -> str:
    """Build the base filename stem for evaluation outputs."""
    # Encode the outer CV scaffold and, when relevant, the tuning strategy.
    base_name = f"{target}_{outer_split_type}_cv"
    if tuning_mode is None:
        return base_name
    return f"{base_name}_{DISPLAY_TUNING_MODES.get(tuning_mode, tuning_mode)}"


def round_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Round common metric columns to 4 decimals before saving."""
    # Keep saved metrics concise and consistent across result tables.
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
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
        try:
            from xgboost import XGBRegressor
        except ImportError as error:
            raise ImportError(
                "xgboost is required when model_type='xgboost'. "
                "Install the project environment from consensus.yml first."
            ) from error
        return XGBRegressor(**resolved_params)

    raise ValueError(f"Unsupported model_type: {model_type}")


def make_regression_pipeline(
    model_type: str,
    model_params: dict[str, Any] | None = None,
    standardize_features: bool = False,
):
    """Instantiate a supported regression pipeline with optional feature scaling."""
    model = make_regression_model(model_type=model_type, model_params=model_params)
    if not standardize_features:
        return model
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
