"""Evaluation model-comparison visualization diagnostics."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any

import numpy as np
import pandas as pd

from flumolscreen.visualization.chemical_space import format_target_class_label

DEFAULT_METRIC = "spearman"

SUMMARY_SUFFIX = "_summary.csv"
FOLD_METRICS_SUFFIX = "_fold_metrics.csv"
TRACE_SUFFIX = "_trace.csv"

KNOWN_SPLIT_TYPES = (
    "random_kfold",
    "group_kfold",
    "random_holdout",
    "holdout_by_column",
)
KNOWN_TUNING_LABELS = (
    "nested_tuning",
    "holdout_tuning",
)

MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge",
    "xgboost": "XGBoost",
}
FEATURE_SET_DISPLAY_NAMES = {
    "method_scores": "Scores",
    "method_ranks": "Ranks",
    "method_rank_summary": "Rank summary",
    "rank_summary": "Rank summary",
    "glide_uncertainty": "Glide uncertainty",
    "chemical_descriptors": "Chem. descriptors",
    "target_context": "Target context",
}
METRIC_DISPLAY_NAMES = {
    "spearman": r"Spearman $r_s$",
    "r2": r"$R^2$",
    "rmse": "RMSE",
    "mae": "MAE",
}
MAXIMIZE_METRICS = {"spearman", "r2"}
MODEL_TUNING_PARAMETER_AXES = {
    "ridge": [
        ("feature_set_label", "Feature set", "categorical", None),
        ("param_log10_alpha", "log10(alpha)", "numeric", "log10"),
    ],
    "xgboost": [
        ("feature_set_label", "Feature set", "categorical", None),
        ("param_log10_learning_rate", "log10(learning rate)", "numeric", "log10"),
        ("param_max_depth", "Max depth", "numeric", "integer"),
        ("param_n_estimators", "N estimators", "numeric", "integer"),
        ("param_subsample", "Subsample", "numeric", "float"),
    ],
}
INNER_CV_LANDSCAPE_BIN_COUNT = 8
MODEL_TUNING_LANDSCAPE_PARAMETERS = {
    "ridge": [
        {
            "parameter": "log10_alpha",
            "column": "param_log10_alpha",
            "label": "log10(alpha)",
            "binning": "continuous",
        },
    ],
    "xgboost": [
        {
            "parameter": "log10_learning_rate",
            "column": "param_log10_learning_rate",
            "label": "log10(learning rate)",
            "binning": "continuous",
        },
        {
            "parameter": "max_depth",
            "column": "param_max_depth",
            "label": "Max depth",
            "binning": "exact",
        },
        {
            "parameter": "n_estimators",
            "column": "param_n_estimators",
            "label": "N estimators",
            "binning": "exact",
        },
        {
            "parameter": "subsample",
            "column": "param_subsample",
            "label": "Subsample",
            "binning": "continuous",
        },
    ],
}


@dataclass(frozen=True)
class EvaluationRun:
    """Paths for one complete evaluation-output triple."""

    dataset_label: str
    base_name: str
    summary_path: Path
    fold_metrics_path: Path
    trace_path: Path


def resolve_output_dir(results_dir: Path | str, round_id: str) -> Path:
    """Return the standard output directory for evaluation diagnostics."""
    return Path(results_dir) / round_id / "visualizations" / "evaluation"


def infer_dataset_label_from_base_name(base_name: str) -> str:
    """Infer the dataset label from an evaluation filename stem."""
    suffixes = []
    for split_type in KNOWN_SPLIT_TYPES:
        suffixes.append(f"_{split_type}_cv")
        suffixes.extend(
            f"_{split_type}_cv_{tuning_label}"
            for tuning_label in KNOWN_TUNING_LABELS
        )

    for suffix in sorted(suffixes, key=len, reverse=True):
        if base_name.endswith(suffix):
            return base_name[: -len(suffix)]
    return base_name


def discover_evaluation_runs(
    results_dir: Path | str,
    round_id: str,
    dataset_labels: list[str] | None = None,
) -> list[EvaluationRun]:
    """Discover complete summary/fold/trace evaluation output triples."""
    evaluation_dir = Path(results_dir) / round_id / "evaluation"
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"Evaluation directory not found: {evaluation_dir}")

    requested_labels = set(dataset_labels or [])
    runs: list[EvaluationRun] = []
    missing_paths: list[Path] = []

    for summary_path in sorted(evaluation_dir.glob(f"*{SUMMARY_SUFFIX}")):
        base_name = summary_path.name[: -len(SUMMARY_SUFFIX)]
        dataset_label = infer_dataset_label_from_base_name(base_name)
        if requested_labels and dataset_label not in requested_labels:
            continue

        fold_metrics_path = evaluation_dir / f"{base_name}{FOLD_METRICS_SUFFIX}"
        trace_path = evaluation_dir / f"{base_name}{TRACE_SUFFIX}"
        run_missing_paths = []
        if not fold_metrics_path.exists():
            run_missing_paths.append(fold_metrics_path)
        if not trace_path.exists():
            run_missing_paths.append(trace_path)
        if run_missing_paths:
            missing_paths.extend(run_missing_paths)
            continue

        runs.append(
            EvaluationRun(
                dataset_label=dataset_label,
                base_name=base_name,
                summary_path=summary_path,
                fold_metrics_path=fold_metrics_path,
                trace_path=trace_path,
            )
        )

    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Incomplete evaluation output triple(s); missing: " f"{missing_text}"
        )
    if not runs:
        label_text = (
            f" for dataset label(s) {sorted(requested_labels)}"
            if requested_labels
            else ""
        )
        raise FileNotFoundError(
            f"No complete evaluation outputs found in {evaluation_dir}{label_text}."
        )
    return runs


def _read_run_table(path: Path, run: EvaluationRun) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    out.insert(0, "base_name", run.base_name)
    out.insert(0, "dataset_label", run.dataset_label)
    return out


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{table_name} is missing required column(s): {missing_columns}")


def load_evaluation_tables(runs: list[EvaluationRun]) -> dict[str, pd.DataFrame]:
    """Load discovered evaluation output triples into labeled dataframes."""
    if not runs:
        raise ValueError("runs must contain at least one evaluation run")

    summary_df = pd.concat(
        [_read_run_table(run.summary_path, run) for run in runs],
        ignore_index=True,
    )
    fold_metrics_df = pd.concat(
        [_read_run_table(run.fold_metrics_path, run) for run in runs],
        ignore_index=True,
    )
    trace_df = pd.concat(
        [_read_run_table(run.trace_path, run) for run in runs],
        ignore_index=True,
    )

    candidate_columns = ["comparison_name", "model_type", "p"]
    _validate_required_columns(summary_df, candidate_columns, "summary_df")
    _validate_required_columns(
        fold_metrics_df,
        [*candidate_columns, "outer_fold"],
        "fold_metrics_df",
    )
    _validate_required_columns(trace_df, candidate_columns, "trace_df")
    return {
        "summary_df": summary_df,
        "fold_metrics_df": fold_metrics_df,
        "trace_df": trace_df,
    }


def _candidate_label(df: pd.DataFrame) -> pd.Series:
    return df["model_type"].astype(str) + " | " + df["comparison_name"].astype(str)


def format_dataset_label(dataset_label: str) -> str:
    """Return a display label for a dataset or target class."""
    return format_target_class_label(str(dataset_label))


def format_metric_label(metric: str) -> str:
    """Return a display label for an evaluation metric."""
    metric_key = str(metric)
    if metric_key in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[metric_key]
    if metric_key.startswith("ef_") and metric_key.endswith("pct"):
        percent_label = metric_key.removeprefix("ef_").removesuffix("pct").replace("p", ".")
        return f"EF {percent_label}%"
    if metric_key.startswith("precision_at_"):
        return f"Precision@{metric_key.removeprefix('precision_at_')}"
    return metric_key.replace("_", " ").title()


def format_model_label(model_type: str) -> str:
    """Return a display label for a model family."""
    model_key = str(model_type)
    return MODEL_DISPLAY_NAMES.get(model_key, model_key.replace("_", " ").title())


def format_feature_set_label(comparison_name: str) -> str:
    """Return a compact display label for a feature-set comparison."""
    feature_parts = str(comparison_name).split("_plus_")
    display_parts = [
        FEATURE_SET_DISPLAY_NAMES.get(part, part.replace("_", " ").title())
        for part in feature_parts
    ]
    return " + ".join(display_parts)


def format_candidate_label(candidate_label: str) -> str:
    """Return a display label for a model and feature-set candidate."""
    model_type, separator, comparison_name = str(candidate_label).partition(" | ")
    if not separator:
        return str(candidate_label)
    return (
        f"{format_model_label(model_type)} | "
        f"{format_feature_set_label(comparison_name)}"
    )


def format_legend_candidate_label(
    candidate_label: str,
    code_label: str,
    width: int = 82,
) -> str:
    """Return a wrapped legend entry for one candidate code."""
    prefix = f"{code_label}: "
    return textwrap.fill(
        f"{prefix}{format_candidate_label(candidate_label)}",
        width=width,
        subsequent_indent=" " * len(prefix),
        break_long_words=False,
        break_on_hyphens=False,
    )


def build_model_family_feature_summary_table(
    fold_metrics_df: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Aggregate fold metrics by dataset, model family, and feature subset."""
    required_columns = [
        "dataset_label",
        "comparison_name",
        "model_type",
        "p",
        metric,
    ]
    _validate_required_columns(fold_metrics_df, required_columns, "fold_metrics_df")

    value_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    out = (
        fold_metrics_df.groupby(
            ["dataset_label", "model_type", "comparison_name", "p"],
            as_index=False,
        )
        .agg(
            **{
                value_col: (metric, "mean"),
                std_col: (metric, "std"),
                "n_folds": (metric, "count"),
            }
        )
        .sort_values(["dataset_label", "model_type", "p", "comparison_name"])
        .reset_index(drop=True)
    )
    out["row_label"] = out["dataset_label"] + " | " + out["model_type"]
    return out


def select_fold_winners(
    fold_metrics_df: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Select one deterministic winning candidate per dataset and outer fold."""
    required_columns = [
        "dataset_label",
        "outer_fold",
        "comparison_name",
        "model_type",
        "p",
        metric,
    ]
    _validate_required_columns(fold_metrics_df, required_columns, "fold_metrics_df")

    candidate_df = fold_metrics_df.loc[fold_metrics_df[metric].notna()].copy()
    if candidate_df.empty:
        raise ValueError(f"No non-null values found for metric: {metric}")

    # Tie-break toward simpler configurations, then stable lexical identifiers.
    sorted_df = candidate_df.sort_values(
        ["dataset_label", "outer_fold", metric, "p", "model_type", "comparison_name"],
        ascending=[True, True, not higher_is_better, True, True, True],
        kind="mergesort",
    )
    winners_df = (
        sorted_df.groupby(["dataset_label", "outer_fold"], sort=False)
        .head(1)
        .copy()
        .sort_values(["dataset_label", "outer_fold"])
        .reset_index(drop=True)
    )
    winners_df["winner_label"] = _candidate_label(winners_df)
    return winners_df


def metric_higher_is_better(metric: str) -> bool:
    """Return whether larger values are better for an evaluation metric."""
    metric_key = str(metric)
    return (
        metric_key in MAXIMIZE_METRICS
        or metric_key.startswith("ef_")
        or metric_key.startswith("precision_at_")
    )


def parse_model_params(value: Any) -> dict[str, Any]:
    """Safely parse one formatted model-parameter dictionary value."""
    if isinstance(value, dict):
        return dict(value)
    if pd.isna(value):
        return {}

    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Could not parse model_params value: {value!r}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"model_params must parse to a dict, got: {type(parsed).__name__}")
    return parsed


def expand_model_params(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Return a trace table with typed parameter columns expanded from model_params."""
    _validate_required_columns(trace_df, ["model_params"], "trace_df")
    out = trace_df.copy()
    parsed_params = out["model_params"].map(parse_model_params)
    param_names = sorted({name for params in parsed_params for name in params})

    for param_name in param_names:
        column = f"param_{param_name}"
        out[column] = [params.get(param_name) for params in parsed_params]
        numeric_values = pd.to_numeric(out[column], errors="coerce")
        if numeric_values.notna().sum() == out[column].notna().sum():
            out[column] = numeric_values

    for param_name in ["alpha", "learning_rate"]:
        column = f"param_{param_name}"
        if column not in out.columns:
            continue
        numeric_values = pd.to_numeric(out[column], errors="coerce")
        out[f"param_log10_{param_name}"] = np.where(
            numeric_values > 0,
            np.log10(numeric_values),
            np.nan,
        )

    out["param_signature"] = parsed_params.map(
        lambda params: ";".join(
            f"{key}={params[key]}" for key in sorted(params)
        )
    )
    return out


def select_best_tuning_trials(
    trace_df: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Select the best tuning trial for each candidate on each outer fold."""
    required_columns = [
        "dataset_label",
        "selection_scope",
        "outer_fold",
        "comparison_name",
        "model_type",
        "p",
        "tuning_metric",
        "trial_number",
        "tuning_score",
        "model_params",
    ]
    _validate_required_columns(trace_df, required_columns, "trace_df")

    out = expand_model_params(trace_df)
    out = out.loc[
        out["outer_fold"].notna()
        & out["selection_scope"].astype(str).str.startswith("outer_fold_")
        & out["tuning_metric"].eq(metric)
        & out["tuning_score"].notna()
    ].copy()
    if out.empty:
        raise ValueError(f"No outer-fold tuning trace rows found for metric: {metric}")

    out["outer_fold"] = out["outer_fold"].astype(int)
    out["trial_number"] = out["trial_number"].astype(int)
    sort_columns = [
        "dataset_label",
        "model_type",
        "comparison_name",
        "outer_fold",
        "tuning_score",
        "trial_number",
        "param_signature",
    ]
    sorted_df = out.sort_values(
        sort_columns,
        ascending=[
            True,
            True,
            True,
            True,
            not metric_higher_is_better(metric),
            True,
            True,
        ],
        kind="mergesort",
    )
    group_columns = [
        "dataset_label",
        "model_type",
        "comparison_name",
        "outer_fold",
    ]
    return (
        sorted_df.groupby(group_columns, sort=False)
        .head(1)
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def merge_selected_tuning_trials_with_outer_metrics(
    selected_trials_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Attach held-out outer-fold metrics to selected tuning trial rows."""
    required_selected_columns = [
        "dataset_label",
        "comparison_name",
        "model_type",
        "p",
        "outer_fold",
        "tuning_score",
    ]
    required_fold_columns = [
        "dataset_label",
        "comparison_name",
        "model_type",
        "p",
        "outer_fold",
        metric,
    ]
    _validate_required_columns(
        selected_trials_df,
        required_selected_columns,
        "selected_trials_df",
    )
    _validate_required_columns(fold_metrics_df, required_fold_columns, "fold_metrics_df")

    selected_df = selected_trials_df.copy()
    fold_df = fold_metrics_df.copy()
    selected_df["outer_fold"] = selected_df["outer_fold"].astype(int)
    fold_df["outer_fold"] = fold_df["outer_fold"].astype(int)

    merge_columns = [
        column
        for column in [
            "dataset_label",
            "comparison_name",
            "model_type",
            "p",
            "outer_fold",
            "tuning_mode",
            "tuning_metric",
        ]
        if column in selected_df.columns and column in fold_df.columns
    ]
    metric_column = f"outer_fold_{metric}"
    merged_df = selected_df.merge(
        fold_df.loc[:, [*merge_columns, metric]].rename(columns={metric: metric_column}),
        on=merge_columns,
        how="left",
        validate="many_to_one",
    )
    if merged_df[metric_column].isna().any():
        missing = merged_df.loc[
            merged_df[metric_column].isna(),
            ["dataset_label", "comparison_name", "model_type", "outer_fold"],
        ].drop_duplicates()
        raise ValueError(
            "Missing outer-fold metric rows for selected tuning trials: "
            f"{missing.to_dict(orient='records')}"
        )
    merged_df["feature_set_label"] = merged_df["comparison_name"].map(
        format_feature_set_label
    )
    merged_df["target_class_label"] = merged_df["dataset_label"].map(format_dataset_label)
    return merged_df


def build_selected_tuning_outer_metric_trials(
    trace_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Build selected best-trial rows with typed params and outer-fold outcomes."""
    selected_trials_df = select_best_tuning_trials(trace_df=trace_df, metric=metric)
    return merge_selected_tuning_trials_with_outer_metrics(
        selected_trials_df=selected_trials_df,
        fold_metrics_df=fold_metrics_df,
        metric=metric,
    )


def build_inner_cv_tuning_trials(
    trace_df: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
) -> pd.DataFrame:
    """Build all outer-fold tuning trial rows with typed params and inner-CV scores."""
    required_columns = [
        "dataset_label",
        "selection_scope",
        "outer_fold",
        "comparison_name",
        "model_type",
        "p",
        "tuning_metric",
        "trial_number",
        "tuning_score",
        "best_tuning_score_so_far",
        "model_params",
    ]
    _validate_required_columns(trace_df, required_columns, "trace_df")

    out = expand_model_params(trace_df)
    out = out.loc[
        out["outer_fold"].notna()
        & out["selection_scope"].astype(str).str.startswith("outer_fold_")
        & out["tuning_metric"].eq(metric)
        & out["tuning_score"].notna()
    ].copy()
    if out.empty:
        raise ValueError(f"No outer-fold inner-CV tuning rows found for metric: {metric}")

    out["outer_fold"] = out["outer_fold"].astype(int)
    out["trial_number"] = out["trial_number"].astype(int)
    out["feature_set_label"] = out["comparison_name"].map(format_feature_set_label)
    out["target_class_label"] = out["dataset_label"].map(format_dataset_label)
    return (
        out.sort_values(
            [
                "dataset_label",
                "model_type",
                "comparison_name",
                "outer_fold",
                "trial_number",
                "param_signature",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _ensure_parent_dir(output_path: Path | str) -> Path:
    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def format_feature_axis_label(comparison_name: str, width: int = 42) -> str:
    """Return a wrapped feature-set label for static plot axes."""
    return textwrap.fill(
        format_feature_set_label(comparison_name),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def build_feature_set_order_by_dataset(feature_summary_df: pd.DataFrame) -> dict[str, list[str]]:
    """Return per-dataset feature-set order by feature count and display label."""
    _validate_required_columns(
        feature_summary_df,
        ["dataset_label", "comparison_name", "p"],
        "feature_summary_df",
    )
    order_df = (
        feature_summary_df.loc[:, ["dataset_label", "comparison_name", "p"]]
        .drop_duplicates()
        .assign(
            feature_set_label=lambda df: df["comparison_name"].map(
                format_feature_set_label
            )
        )
        .sort_values(
            ["dataset_label", "p", "feature_set_label", "comparison_name"],
            kind="mergesort",
        )
    )
    return {
        str(dataset_label): dataset_df["comparison_name"].tolist()
        for dataset_label, dataset_df in order_df.groupby("dataset_label", sort=False)
    }


def _format_parallel_tick(value: float, tick_format: str | None) -> str:
    if pd.isna(value):
        return ""
    if tick_format == "integer":
        return f"{int(round(value))}"
    if tick_format == "log10":
        return f"{value:.1f}"
    if tick_format == "metric":
        return f"{value:.2f}"
    return f"{value:.2g}"


def _numeric_axis_limits(values: pd.Series) -> tuple[float, float]:
    finite_values = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    if finite_values.empty:
        return (-0.5, 0.5)

    lower = float(finite_values.min())
    upper = float(finite_values.max())
    if lower == upper:
        padding = max(abs(lower) * 0.08, 0.05)
    else:
        padding = max((upper - lower) * 0.08, 0.05)
    return lower - padding, upper + padding


def _scale_numeric_value(value: Any, limits: tuple[float, float]) -> float:
    lower, upper = limits
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value) or lower == upper:
        return 0.5
    return float((numeric_value - lower) / (upper - lower))


def _build_static_parallel_axis_specs(
    plot_df: pd.DataFrame,
    model_type: str,
    outcome_column: str,
    outcome_label: str,
) -> list[dict[str, Any]]:
    parameter_axes = MODEL_TUNING_PARAMETER_AXES.get(model_type)
    if parameter_axes is None:
        raise ValueError(f"No hyperparameter PCP axes configured for model: {model_type}")

    specs = []
    axes = [*parameter_axes, (outcome_column, outcome_label, "numeric", "metric")]
    for column, label, kind, tick_format in axes:
        if column not in plot_df.columns:
            raise ValueError(
                "Tuning trial table is missing required PCP column: "
                f"{column}"
            )
        if kind == "categorical":
            categories = (
                plot_df.loc[:, ["comparison_name", "p", column]]
                .drop_duplicates()
                .sort_values(["p", column, "comparison_name"], kind="mergesort")
                [column]
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            tick_label_by_category = {
                category: f"F{idx + 1}" for idx, category in enumerate(categories)
            }
            if len(categories) == 1:
                category_position = {categories[0]: 0.5}
            else:
                category_position = {
                    category: idx / (len(categories) - 1)
                    for idx, category in enumerate(categories)
                }
            specs.append(
                {
                    "column": column,
                    "label": label,
                    "kind": kind,
                    "categories": categories,
                    "category_position": category_position,
                    "tick_label_by_category": tick_label_by_category,
                }
            )
        else:
            specs.append(
                {
                    "column": column,
                    "label": label,
                    "kind": kind,
                    "limits": _numeric_axis_limits(plot_df[column]),
                    "tick_format": tick_format,
                }
            )
    return specs


def _scale_parallel_axis_value(value: Any, spec: dict[str, Any]) -> float:
    if spec["kind"] == "categorical":
        return float(spec["category_position"].get(str(value), 0.5))
    return _scale_numeric_value(value=value, limits=spec["limits"])


def _draw_parallel_axis(ax, axis_idx: int, spec: dict[str, Any]) -> None:
    ax.vlines(axis_idx, 0.0, 1.0, color="#5c5c5c", linewidth=0.9, zorder=1)
    ax.text(
        axis_idx,
        1.065,
        spec["label"],
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    if spec["kind"] == "categorical":
        for category in spec["categories"]:
            y_pos = spec["category_position"][category]
            ax.plot(
                [axis_idx - 0.025, axis_idx + 0.025],
                [y_pos, y_pos],
                color="#777777",
                linewidth=0.7,
            )
            ax.text(
                axis_idx - 0.045,
                y_pos,
                spec["tick_label_by_category"].get(category, category),
                ha="right",
                va="center",
                fontsize=7.5,
            )
        return

    lower, upper = spec["limits"]
    tick_values = np.linspace(lower, upper, num=4)
    for tick_value in tick_values:
        y_pos = _scale_numeric_value(tick_value, spec["limits"])
        ax.plot(
            [axis_idx - 0.025, axis_idx + 0.025],
            [y_pos, y_pos],
            color="#777777",
            linewidth=0.7,
        )
        ax.text(
            axis_idx - 0.04,
            y_pos,
            _format_parallel_tick(tick_value, spec.get("tick_format")),
            ha="right",
            va="center",
            fontsize=7,
        )


def _parallel_color_limits(values: pd.Series) -> tuple[float, float]:
    lower, upper = _numeric_axis_limits(values)
    if lower < -1.0 and upper <= 1.05:
        lower = -1.0
    if upper > 1.0 and lower >= -1.05:
        upper = 1.0
    return lower, upper


def _feature_code_key_text(axis_specs: list[dict[str, Any]]) -> str:
    feature_spec = next(
        (spec for spec in axis_specs if spec["kind"] == "categorical"),
        None,
    )
    if feature_spec is None:
        return ""

    entries = [
        f"{feature_spec['tick_label_by_category'][category]}={category}"
        for category in feature_spec["categories"]
    ]
    return textwrap.fill(
        "Feature-set codes: " + "; ".join(entries),
        width=180,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _write_tuning_parallel_coordinates_plot(
    trials_df: pd.DataFrame,
    output_path: Path | str,
    model_type: str,
    outcome_column: str,
    outcome_label: str,
    title: str,
    table_name: str,
    line_alpha: float,
    line_width: float,
) -> Path:
    """Write a static faceted PCP for tuning trials."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "matplotlib is required to write hyperparameter tuning PCPs."
        ) from error

    output_path = _ensure_parent_dir(output_path)
    required_columns = [
        "dataset_label",
        "comparison_name",
        "feature_set_label",
        "model_type",
        "outer_fold",
        "trial_number",
        outcome_column,
    ]
    _validate_required_columns(trials_df, required_columns, table_name)

    plot_df = trials_df.loc[
        trials_df["model_type"].astype(str).eq(model_type)
    ].copy()
    if plot_df.empty:
        raise ValueError(f"No tuning trials found for model: {model_type}")

    axis_specs = _build_static_parallel_axis_specs(
        plot_df=plot_df,
        model_type=model_type,
        outcome_column=outcome_column,
        outcome_label=outcome_label,
    )
    dataset_order = sorted(
        plot_df["dataset_label"].astype(str).unique(),
        key=format_dataset_label,
    )
    color_limits = _parallel_color_limits(plot_df[outcome_column])
    color_norm = plt.Normalize(vmin=color_limits[0], vmax=color_limits[1])
    cmap = plt.get_cmap("viridis")

    fig_width = 15.0 if model_type == "xgboost" else 10.5
    fig_height = max(8.0, 2.65 * len(dataset_order))
    fig, axes = plt.subplots(
        len(dataset_order),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    x_positions = np.arange(len(axis_specs))

    for ax, dataset_label in zip(axes, dataset_order):
        dataset_df = plot_df.loc[
            plot_df["dataset_label"].astype(str).eq(dataset_label)
        ].copy()
        dataset_df = dataset_df.sort_values(
            ["comparison_name", "outer_fold", "trial_number"],
            kind="mergesort",
        )

        for _, row in dataset_df.iterrows():
            y_values = [
                _scale_parallel_axis_value(row[spec["column"]], spec)
                for spec in axis_specs
            ]
            color_value = pd.to_numeric(
                pd.Series([row[outcome_column]]),
                errors="coerce",
            ).iloc[0]
            line_color = (
                cmap(color_norm(color_value))
                if pd.notna(color_value)
                else "#999999"
            )
            ax.plot(
                x_positions,
                y_values,
                color=line_color,
                linewidth=line_width,
                alpha=line_alpha,
                zorder=2,
            )

        for axis_idx, spec in enumerate(axis_specs):
            _draw_parallel_axis(ax, axis_idx=axis_idx, spec=spec)

        ax.set_xlim(-0.86, len(axis_specs) - 0.5)
        ax.set_ylim(-0.06, 1.10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(format_dataset_label(dataset_label), loc="left", fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=color_norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.84, pad=0.045)
    cbar.set_label(outcome_label)
    fig.suptitle(title, y=0.992)
    feature_key_text = _feature_code_key_text(axis_specs)
    if feature_key_text:
        fig.text(0.02, 0.018, feature_key_text, ha="left", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.12, right=0.76, top=0.94, bottom=0.105, hspace=0.54)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_selected_tuning_outer_metric_parallel_coordinates_plot(
    selected_trials_df: pd.DataFrame,
    output_path: Path | str,
    model_type: str,
    metric: str = DEFAULT_METRIC,
) -> Path:
    """Write a static faceted PCP for selected params and outer-fold outcomes."""
    outcome_label = f"Outer fold {format_metric_label(metric)}"
    return _write_tuning_parallel_coordinates_plot(
        trials_df=selected_trials_df,
        output_path=output_path,
        model_type=model_type,
        outcome_column=f"outer_fold_{metric}",
        outcome_label=outcome_label,
        title=(
            f"{format_model_label(model_type)} selected tuning trials by "
            f"{outcome_label}"
        ),
        table_name="selected_trials_df",
        line_alpha=0.54,
        line_width=1.0,
    )


def write_inner_cv_tuning_score_parallel_coordinates_plot(
    inner_cv_trials_df: pd.DataFrame,
    output_path: Path | str,
    model_type: str,
    metric: str = DEFAULT_METRIC,
) -> Path:
    """Write a static faceted PCP for all inner-CV tuning trial scores."""
    outcome_label = f"Inner-CV tuning {format_metric_label(metric)}"
    return _write_tuning_parallel_coordinates_plot(
        trials_df=inner_cv_trials_df,
        output_path=output_path,
        model_type=model_type,
        outcome_column="tuning_score",
        outcome_label=outcome_label,
        title=f"{format_model_label(model_type)} inner-CV tuning scores",
        table_name="inner_cv_trials_df",
        line_alpha=0.28,
        line_width=0.75,
    )


def _coerce_landscape_numeric_values(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _format_landscape_bin_number(value: float) -> str:
    if pd.isna(value):
        return ""
    numeric_value = float(value)
    rounded_value = round(numeric_value)
    if np.isclose(numeric_value, rounded_value, atol=1e-9):
        return f"{int(rounded_value)}"
    return f"{numeric_value:.3g}"


def _build_exact_landscape_bins(values: pd.Series) -> pd.DataFrame:
    numeric_values = _coerce_landscape_numeric_values(values)
    return pd.DataFrame(
        {
            "parameter_bin": numeric_values.map(_format_landscape_bin_number),
            "parameter_bin_sort": numeric_values,
            "parameter_bin_lower": numeric_values,
            "parameter_bin_upper": numeric_values,
        },
        index=values.index,
    )


def _build_equal_width_landscape_bins(
    values: pd.Series,
    bin_count: int = INNER_CV_LANDSCAPE_BIN_COUNT,
) -> pd.DataFrame:
    numeric_values = _coerce_landscape_numeric_values(values)
    bin_df = pd.DataFrame(
        {
            "parameter_bin": pd.NA,
            "parameter_bin_sort": np.nan,
            "parameter_bin_lower": np.nan,
            "parameter_bin_upper": np.nan,
        },
        index=values.index,
    )

    finite_values = numeric_values.dropna()
    if finite_values.empty:
        return bin_df

    lower = float(finite_values.min())
    upper = float(finite_values.max())
    if lower == upper:
        padding = max(abs(lower) * 0.04, 0.05)
        lower -= padding
        upper += padding

    edges = np.linspace(lower, upper, num=bin_count + 1)
    labels = [
        (
            f"{_format_landscape_bin_number(edges[idx])} to "
            f"{_format_landscape_bin_number(edges[idx + 1])}"
        )
        for idx in range(bin_count)
    ]
    mids = (edges[:-1] + edges[1:]) / 2.0

    valid_values = numeric_values.dropna()
    bin_positions = np.searchsorted(
        edges,
        valid_values.to_numpy(dtype=float),
        side="right",
    ) - 1
    bin_positions = np.clip(bin_positions, 0, bin_count - 1)
    bin_position_series = pd.Series(bin_positions, index=valid_values.index)

    bin_df.loc[valid_values.index, "parameter_bin"] = bin_position_series.map(
        lambda idx: labels[int(idx)]
    )
    bin_df.loc[valid_values.index, "parameter_bin_sort"] = bin_position_series.map(
        lambda idx: float(mids[int(idx)])
    )
    bin_df.loc[valid_values.index, "parameter_bin_lower"] = bin_position_series.map(
        lambda idx: float(edges[int(idx)])
    )
    bin_df.loc[valid_values.index, "parameter_bin_upper"] = bin_position_series.map(
        lambda idx: float(edges[int(idx) + 1])
    )
    return bin_df


def _build_landscape_parameter_bins(
    values: pd.Series,
    binning: str,
    bin_count: int = INNER_CV_LANDSCAPE_BIN_COUNT,
) -> pd.DataFrame:
    if binning == "exact":
        return _build_exact_landscape_bins(values)
    if binning == "continuous":
        return _build_equal_width_landscape_bins(values, bin_count=bin_count)
    raise ValueError(f"Unknown landscape binning mode: {binning}")


def _landscape_parameter_order() -> dict[str, int]:
    parameters = [
        spec["parameter"]
        for specs in MODEL_TUNING_LANDSCAPE_PARAMETERS.values()
        for spec in specs
    ]
    return {parameter: idx for idx, parameter in enumerate(parameters)}


def build_inner_cv_tuning_score_landscape(
    inner_cv_trials_df: pd.DataFrame,
    bin_count: int = INNER_CV_LANDSCAPE_BIN_COUNT,
) -> pd.DataFrame:
    """Aggregate inner-CV tuning trials into model-parameter score landscapes."""
    if bin_count < 1:
        raise ValueError("bin_count must be at least 1")

    required_columns = [
        "dataset_label",
        "comparison_name",
        "feature_set_label",
        "model_type",
        "p",
        "tuning_score",
    ]
    _validate_required_columns(
        inner_cv_trials_df,
        required_columns,
        "inner_cv_trials_df",
    )

    trials_df = inner_cv_trials_df.copy()
    if "target_class_label" not in trials_df.columns:
        trials_df["target_class_label"] = trials_df["dataset_label"].map(
            format_dataset_label
        )
    trials_df["tuning_score"] = pd.to_numeric(
        trials_df["tuning_score"],
        errors="coerce",
    )

    landscape_frames: list[pd.DataFrame] = []
    for model_type, parameter_specs in MODEL_TUNING_LANDSCAPE_PARAMETERS.items():
        model_df = trials_df.loc[
            trials_df["model_type"].astype(str).eq(model_type)
        ].copy()
        if model_df.empty:
            continue

        for spec in parameter_specs:
            column = spec["column"]
            if column not in model_df.columns:
                raise ValueError(
                    "Inner-CV tuning trial table is missing required landscape "
                    f"column for {model_type}: {column}"
                )

            parameter_values = _coerce_landscape_numeric_values(model_df[column])
            parameter_df = model_df.loc[
                model_df["tuning_score"].notna() & parameter_values.notna()
            ].copy()
            if parameter_df.empty:
                continue

            bin_df = _build_landscape_parameter_bins(
                parameter_df[column],
                binning=spec["binning"],
                bin_count=bin_count,
            )
            parameter_df = pd.concat([parameter_df, bin_df], axis=1)
            parameter_df["parameter"] = spec["parameter"]
            parameter_df["parameter_label"] = spec["label"]

            group_columns = [
                "dataset_label",
                "target_class_label",
                "model_type",
                "parameter",
                "parameter_label",
                "comparison_name",
                "feature_set_label",
                "p",
                "parameter_bin",
                "parameter_bin_sort",
                "parameter_bin_lower",
                "parameter_bin_upper",
            ]
            grouped_df = (
                parameter_df.groupby(group_columns, dropna=False, as_index=False)
                .agg(
                    median_tuning_score=("tuning_score", "median"),
                    trial_count=("tuning_score", "size"),
                )
            )
            landscape_frames.append(grouped_df)

    if not landscape_frames:
        raise ValueError("No configured model-parameter inner-CV landscape rows found.")

    landscape_df = pd.concat(landscape_frames, ignore_index=True)
    parameter_rank = _landscape_parameter_order()
    landscape_df["_parameter_rank"] = landscape_df["parameter"].map(parameter_rank)
    return (
        landscape_df.sort_values(
            [
                "dataset_label",
                "model_type",
                "_parameter_rank",
                "p",
                "feature_set_label",
                "comparison_name",
                "parameter_bin_sort",
            ],
            kind="mergesort",
        )
        .drop(columns="_parameter_rank")
        .reset_index(drop=True)
    )


def _landscape_feature_order(landscape_df: pd.DataFrame) -> list[str]:
    return (
        landscape_df.loc[:, ["comparison_name", "feature_set_label", "p"]]
        .drop_duplicates()
        .sort_values(["p", "feature_set_label", "comparison_name"], kind="mergesort")
        ["comparison_name"]
        .tolist()
    )


def _landscape_bin_order(landscape_df: pd.DataFrame) -> list[str]:
    return (
        landscape_df.loc[:, ["parameter_bin", "parameter_bin_sort"]]
        .drop_duplicates()
        .sort_values(["parameter_bin_sort", "parameter_bin"], kind="mergesort")
        ["parameter_bin"]
        .astype(str)
        .tolist()
    )


def _landscape_parameters_for_model(
    model_type: str,
    landscape_df: pd.DataFrame,
) -> list[str]:
    configured_order = [
        spec["parameter"]
        for spec in MODEL_TUNING_LANDSCAPE_PARAMETERS.get(model_type, [])
    ]
    observed_parameters = set(landscape_df["parameter"].astype(str))
    ordered_parameters = [
        parameter for parameter in configured_order if parameter in observed_parameters
    ]
    return [
        *ordered_parameters,
        *sorted(observed_parameters.difference(ordered_parameters)),
    ]


def write_inner_cv_tuning_score_landscape_plot(
    landscape_df: pd.DataFrame,
    output_path: Path | str,
    model_type: str,
    metric: str = DEFAULT_METRIC,
) -> Path:
    """Write a static faceted heatmap of aggregated inner-CV tuning scores."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "matplotlib is required to write inner-CV score landscape plots."
        ) from error

    output_path = _ensure_parent_dir(output_path)
    required_columns = [
        "dataset_label",
        "model_type",
        "parameter",
        "parameter_label",
        "comparison_name",
        "feature_set_label",
        "p",
        "parameter_bin",
        "parameter_bin_sort",
        "median_tuning_score",
        "trial_count",
    ]
    _validate_required_columns(landscape_df, required_columns, "landscape_df")

    plot_df = landscape_df.loc[
        landscape_df["model_type"].astype(str).eq(model_type)
    ].copy()
    if plot_df.empty:
        raise ValueError(f"No inner-CV score landscape rows found for model: {model_type}")

    plot_df["median_tuning_score"] = pd.to_numeric(
        plot_df["median_tuning_score"],
        errors="coerce",
    )
    plot_df = plot_df.loc[plot_df["median_tuning_score"].notna()].copy()
    if plot_df.empty:
        raise ValueError(
            f"No finite inner-CV score landscape values found for model: {model_type}"
        )

    dataset_order = sorted(
        plot_df["dataset_label"].astype(str).unique(),
        key=format_dataset_label,
    )
    parameter_order = _landscape_parameters_for_model(model_type, plot_df)
    color_limits = _parallel_color_limits(plot_df["median_tuning_score"])
    color_norm = plt.Normalize(vmin=color_limits[0], vmax=color_limits[1])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f2f2f2")

    max_feature_count = max(
        len(_landscape_feature_order(plot_df.loc[plot_df["dataset_label"].eq(dataset)]))
        for dataset in dataset_order
    )
    panel_height = max(2.7, 0.22 * max_feature_count + 1.25)
    fig_width = max(8.8, 3.35 * len(parameter_order) + 3.2)
    fig_height = max(5.8, panel_height * len(dataset_order) + 2.0)
    fig, axes = plt.subplots(
        len(dataset_order),
        len(parameter_order),
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for row_idx, dataset_label in enumerate(dataset_order):
        dataset_df = plot_df.loc[
            plot_df["dataset_label"].astype(str).eq(dataset_label)
        ].copy()
        feature_order = _landscape_feature_order(dataset_df)
        feature_labels = (
            dataset_df.loc[:, ["comparison_name", "feature_set_label"]]
            .drop_duplicates()
            .set_index("comparison_name")["feature_set_label"]
            .to_dict()
        )

        for col_idx, parameter in enumerate(parameter_order):
            ax = axes[row_idx, col_idx]
            parameter_df = dataset_df.loc[
                dataset_df["parameter"].astype(str).eq(parameter)
            ].copy()
            if parameter_df.empty:
                ax.axis("off")
                continue

            bin_order = _landscape_bin_order(parameter_df)
            score_matrix = pd.DataFrame(
                np.nan,
                index=feature_order,
                columns=bin_order,
            )
            for _, row in parameter_df.iterrows():
                score_matrix.loc[
                    row["comparison_name"],
                    str(row["parameter_bin"]),
                ] = row["median_tuning_score"]

            ax.imshow(
                np.ma.masked_invalid(score_matrix.to_numpy(dtype=float)),
                aspect="auto",
                cmap=cmap,
                norm=color_norm,
            )
            ax.set_title(
                (
                    f"{format_dataset_label(dataset_label)} | "
                    f"{parameter_df['parameter_label'].iloc[0]}"
                ),
                loc="left",
                fontsize=9,
                fontweight="bold",
            )
            ax.set_xticks(range(len(bin_order)), bin_order, rotation=45, ha="right")
            if col_idx == 0:
                ax.set_yticks(
                    range(len(feature_order)),
                    [
                        feature_labels.get(comparison_name, comparison_name)
                        for comparison_name in feature_order
                    ],
                )
            else:
                ax.set_yticks(range(len(feature_order)), [])

            ax.set_xticks(np.arange(-0.5, len(bin_order), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(feature_order), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.55)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=color_norm, cmap=cmap)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.30, 0.060, 0.52, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"Median inner-CV tuning {format_metric_label(metric)}")
    cbar.ax.xaxis.set_label_position("top")
    fig.suptitle(
        f"{format_model_label(model_type)} inner-CV tuning score landscape",
        y=0.992,
    )
    fig.subplots_adjust(
        left=0.34 if len(parameter_order) > 1 else 0.42,
        right=0.98,
        top=0.94,
        bottom=0.18,
        hspace=0.62,
        wspace=0.26,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_metric_axis_limits(
    feature_summary_df: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
) -> tuple[float, float]:
    """Return shared x-axis limits for the dot plot metric and fold SD."""
    value_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    _validate_required_columns(feature_summary_df, [value_col, std_col], "feature_summary_df")

    values = pd.to_numeric(feature_summary_df[value_col], errors="coerce")
    spreads = pd.to_numeric(feature_summary_df[std_col], errors="coerce").fillna(0.0)
    lower_values = values - spreads
    upper_values = values + spreads
    finite_values = pd.concat([lower_values, upper_values]).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    if finite_values.empty:
        return (-0.5, 0.5)

    lower = min(float(finite_values.min()), 0.0)
    upper = max(float(finite_values.max()), 0.0)
    if lower == upper:
        padding = max(abs(lower) * 0.08, 0.05)
    else:
        padding = max((upper - lower) * 0.08, 0.03)
    lower -= padding
    upper += padding

    if metric == "spearman":
        lower = max(-1.0, lower)
        upper = min(1.0, upper)
    return lower, upper


def _model_style(model_type: str) -> dict[str, str]:
    styles = {
        "ridge": {"color": "#3b6ea8", "marker": "o"},
        "xgboost": {"color": "#d55e00", "marker": "s"},
    }
    fallback_styles = [
        {"color": "#4c956c", "marker": "^"},
        {"color": "#7b2cbf", "marker": "D"},
        {"color": "#444444", "marker": "P"},
    ]
    if model_type in styles:
        return styles[model_type]
    fallback_idx = sum(ord(char) for char in model_type) % len(fallback_styles)
    return fallback_styles[fallback_idx]


def _ordered_model_types(feature_summary_df: pd.DataFrame) -> list[str]:
    model_types = sorted(
        feature_summary_df["model_type"].astype(str).unique(),
        key=lambda model_type: format_model_label(model_type),
    )
    preferred_order = ["ridge", "xgboost"]
    return [
        *[model_type for model_type in preferred_order if model_type in model_types],
        *[model_type for model_type in model_types if model_type not in preferred_order],
    ]


def write_model_family_feature_dotplot_static_plot(
    feature_summary_df: pd.DataFrame,
    output_path: Path | str,
    metric: str = DEFAULT_METRIC,
) -> Path:
    """Write a static faceted dot plot for model-family feature comparisons."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise ImportError(
            "matplotlib is required to write evaluation feature-comparison plots."
        ) from error

    output_path = _ensure_parent_dir(output_path)
    value_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    _validate_required_columns(
        feature_summary_df,
        [
            "dataset_label",
            "model_type",
            "row_label",
            "comparison_name",
            "p",
            value_col,
            std_col,
        ],
        "feature_summary_df",
    )

    plot_df = feature_summary_df.copy()
    plot_df["dataset_label"] = plot_df["dataset_label"].astype(str)
    plot_df["model_type"] = plot_df["model_type"].astype(str)
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df[std_col] = pd.to_numeric(plot_df[std_col], errors="coerce").fillna(0.0)
    plot_df = plot_df.loc[plot_df[value_col].notna()].copy()
    if plot_df.empty:
        raise ValueError(f"No non-null values found for metric: {metric}")

    feature_order_by_dataset = build_feature_set_order_by_dataset(plot_df)
    dataset_order = sorted(
        feature_order_by_dataset,
        key=lambda dataset_label: format_dataset_label(dataset_label),
    )
    model_order = _ordered_model_types(plot_df)
    x_limits = build_metric_axis_limits(plot_df, metric=metric)

    panel_heights = [
        max(2.5, 0.38 * len(feature_order_by_dataset[dataset_label]) + 0.9)
        for dataset_label in dataset_order
    ]
    fig_width = 13.5
    fig_height = max(6.0, sum(panel_heights) + 1.2)
    fig, axes = plt.subplots(
        len(dataset_order),
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    if len(model_order) > 1:
        offsets = np.linspace(-0.13, 0.13, num=len(model_order))
    else:
        offsets = np.array([0.0])
    offset_by_model = dict(zip(model_order, offsets))

    for ax, dataset_label in zip(axes, dataset_order):
        dataset_df = plot_df.loc[plot_df["dataset_label"].eq(dataset_label)].copy()
        feature_order = feature_order_by_dataset[dataset_label]
        feature_position = {
            comparison_name: idx for idx, comparison_name in enumerate(feature_order)
        }

        ax.axvline(0.0, color="#808080", linewidth=0.8, linestyle="--", zorder=0)
        for model_type in model_order:
            model_df = dataset_df.loc[dataset_df["model_type"].eq(model_type)].copy()
            if model_df.empty:
                continue
            model_df["y_position"] = model_df["comparison_name"].map(feature_position)
            model_df = model_df.sort_values("y_position")
            style = _model_style(model_type)
            ax.errorbar(
                model_df[value_col],
                model_df["y_position"] + offset_by_model[model_type],
                xerr=model_df[std_col],
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                markersize=6,
                elinewidth=1.1,
                capsize=3,
                linestyle="none",
                alpha=0.95,
                zorder=3,
            )

        ax.set_yticks(
            range(len(feature_order)),
            [format_feature_axis_label(feature_set) for feature_set in feature_order],
        )
        ax.set_ylim(-0.6, len(feature_order) - 0.4)
        ax.invert_yaxis()
        ax.set_xlim(*x_limits)
        ax.set_title(format_dataset_label(dataset_label), loc="left", fontweight="bold")
        ax.grid(axis="x", color="#d0d0d0", linewidth=0.6, alpha=0.7)
        ax.set_ylabel("")
        if hasattr(ax, "tick_params"):
            ax.tick_params(axis="y", labelsize=8.5)
            ax.tick_params(axis="x", labelsize=9)
        if hasattr(ax, "spines"):
            for spine_name in ["top", "right"]:
                ax.spines[spine_name].set_visible(False)

    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[-1].set_xlabel(format_metric_label(metric))
    fig.suptitle(
        f"Model-family feature comparison by {format_metric_label(metric)}",
        y=0.985,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=_model_style(model_type)["marker"],
            color=_model_style(model_type)["color"],
            linestyle="none",
            markersize=7,
            label=format_model_label(model_type),
        )
        for model_type in model_order
    ]
    fig.legend(
        handles=legend_handles,
        title="Model family",
        loc="upper center",
        bbox_to_anchor=(0.62, 0.965),
        ncol=max(1, len(legend_handles)),
        frameon=False,
    )
    fig.subplots_adjust(left=0.36, right=0.98, top=0.93, bottom=0.055, hspace=0.58)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_fold_winner_stability_static_plot(
    winners_df: pd.DataFrame,
    output_path: Path | str,
    metric: str = DEFAULT_METRIC,
) -> Path:
    """Write a static plot showing the winning candidate in each outer fold."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as error:
        raise ImportError(
            "matplotlib is required to write fold-winner stability plots."
        ) from error

    output_path = _ensure_parent_dir(output_path)
    _validate_required_columns(
        winners_df,
        ["dataset_label", "outer_fold", "winner_label", metric],
        "winners_df",
    )

    row_order = sorted(winners_df["dataset_label"].astype(str).unique())
    fold_order = sorted(winners_df["outer_fold"].unique())
    winner_order = winners_df["winner_label"].value_counts().index.tolist()
    winner_code_by_label = {
        winner_label: idx for idx, winner_label in enumerate(winner_order)
    }

    code_matrix = pd.DataFrame(np.nan, index=row_order, columns=fold_order)
    metric_matrix = pd.DataFrame(np.nan, index=row_order, columns=fold_order)
    for _, row in winners_df.iterrows():
        dataset_label = str(row["dataset_label"])
        outer_fold = row["outer_fold"]
        code_matrix.loc[dataset_label, outer_fold] = winner_code_by_label[
            row["winner_label"]
        ]
        metric_matrix.loc[dataset_label, outer_fold] = row[metric]

    fig_width = max(7.0, 0.9 * max(1, len(fold_order)) + 4.0)
    fig_height = max(3.5, 0.6 * max(1, len(row_order)) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = plt.get_cmap("tab20", max(1, len(winner_order))).copy()
    cmap.set_bad("#f2f2f2")
    masked_codes = np.ma.masked_invalid(code_matrix.to_numpy(dtype=float))
    ax.imshow(
        masked_codes,
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=max(0.5, len(winner_order) - 0.5),
    )

    for row_idx, dataset_label in enumerate(row_order):
        for col_idx, outer_fold in enumerate(fold_order):
            code_value = code_matrix.loc[dataset_label, outer_fold]
            metric_value = metric_matrix.loc[dataset_label, outer_fold]
            if pd.isna(code_value):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"C{int(code_value) + 1}\n{metric_value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    ax.set_xticks(range(len(fold_order)), [str(fold) for fold in fold_order])
    ax.set_yticks(range(len(row_order)), [format_dataset_label(row) for row in row_order])
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("")
    ax.set_title(f"Fold winner stability by {format_metric_label(metric)}")

    legend_handles = [
        Patch(
            facecolor=cmap(idx),
            edgecolor="none",
            label=format_legend_candidate_label(
                candidate_label=label,
                code_label=f"C{idx + 1}",
            ),
        )
        for idx, label in enumerate(winner_order)
    ]
    ax.legend(
        handles=legend_handles,
        title="Winning candidate",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.10, right=0.62, top=0.86, bottom=0.16)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _split_tuning_trials_by_model(trials_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return model-specific tuning trial tables for configured PCP model types."""
    return {
        str(model_type): model_df.copy().dropna(axis=1, how="all")
        for model_type, model_df in trials_df.groupby("model_type", sort=True)
        if str(model_type) in MODEL_TUNING_PARAMETER_AXES and not model_df.empty
    }


def create_evaluation_diagnostics(
    results_dir: Path | str = "results",
    round_id: str = "round_synthetic",
    dataset_labels: list[str] | None = None,
    metric: str = DEFAULT_METRIC,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Create evaluation model-comparison diagnostics for one result round."""
    runs = discover_evaluation_runs(
        results_dir=results_dir,
        round_id=round_id,
        dataset_labels=dataset_labels,
    )
    tables = load_evaluation_tables(runs)
    feature_summary_df = build_model_family_feature_summary_table(
        fold_metrics_df=tables["fold_metrics_df"],
        metric=metric,
    )
    winners_df = select_fold_winners(
        fold_metrics_df=tables["fold_metrics_df"],
        metric=metric,
    )
    selected_outer_trials_df = build_selected_tuning_outer_metric_trials(
        trace_df=tables["trace_df"],
        fold_metrics_df=tables["fold_metrics_df"],
        metric=metric,
    )
    inner_cv_trials_df = build_inner_cv_tuning_trials(
        trace_df=tables["trace_df"],
        metric=metric,
    )
    inner_cv_landscape_df = build_inner_cv_tuning_score_landscape(inner_cv_trials_df)
    selected_outer_trials_by_model = _split_tuning_trials_by_model(
        selected_outer_trials_df
    )
    inner_cv_trials_by_model = _split_tuning_trials_by_model(inner_cv_trials_df)
    inner_cv_landscape_by_model = {
        str(model_type): model_df.copy().dropna(axis=1, how="all")
        for model_type, model_df in inner_cv_landscape_df.groupby(
            "model_type",
            sort=True,
        )
        if str(model_type) in MODEL_TUNING_LANDSCAPE_PARAMETERS
        and not model_df.empty
    }

    output_dir = resolve_output_dir(results_dir=results_dir, round_id=round_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete_path in [
        output_dir / "model_comparison_candidates.csv",
        output_dir / "model_comparison_parallel_coordinates.html",
        output_dir / "model_family_feature_heatmap.csv",
        output_dir / "model_family_feature_heatmap.png",
        output_dir / "ridge_hyperparameter_tuning_selected_trials.csv",
        output_dir / "xgboost_hyperparameter_tuning_selected_trials.csv",
        output_dir / "ridge_hyperparameter_tuning_stability_parallel_coordinates.png",
        output_dir / "xgboost_hyperparameter_tuning_stability_parallel_coordinates.png",
        output_dir / "ridge_hyperparameter_tuning_stability_parallel_coordinates.html",
        output_dir / "xgboost_hyperparameter_tuning_stability_parallel_coordinates.html",
    ]:
        if obsolete_path.exists():
            obsolete_path.unlink()
    for obsolete_path in output_dir.glob(
        "*_inner_cv_tuning_score_parallel_coordinates.png"
    ):
        obsolete_path.unlink()

    metric_slug = str(metric).replace(" ", "_").replace("/", "_")
    data_paths = {
        "tuning_trace_csv": output_dir / "model_comparison_tuning_trace.csv",
        "feature_summary_csv": output_dir / "model_family_feature_summary.csv",
        "fold_winners_csv": output_dir / "fold_winner_stability.csv",
    }
    for model_type in selected_outer_trials_by_model:
        data_paths[f"{model_type}_selected_tuning_outer_{metric_slug}_trials_csv"] = (
            output_dir / f"{model_type}_selected_tuning_outer_{metric_slug}_trials.csv"
        )
    for model_type in inner_cv_trials_by_model:
        data_paths[f"{model_type}_inner_cv_tuning_trials_csv"] = (
            output_dir / f"{model_type}_inner_cv_tuning_trials.csv"
        )
    for model_type in inner_cv_landscape_by_model:
        data_paths[f"{model_type}_inner_cv_tuning_score_landscape_csv"] = (
            output_dir / f"{model_type}_inner_cv_tuning_score_landscape.csv"
        )

    plot_paths = {
        "feature_dotplot_png": output_dir / "model_family_feature_dotplot.png",
        "fold_winner_stability_png": output_dir / "fold_winner_stability.png",
    }
    for model_type in selected_outer_trials_by_model:
        plot_paths[
            f"{model_type}_selected_tuning_outer_{metric_slug}_parallel_coordinates_png"
        ] = (
            output_dir
            / f"{model_type}_selected_tuning_outer_{metric_slug}_parallel_coordinates.png"
        )
    for model_type in inner_cv_trials_by_model:
        if model_type not in inner_cv_landscape_by_model:
            continue
        plot_paths[f"{model_type}_inner_cv_tuning_score_landscape_png"] = (
            output_dir / f"{model_type}_inner_cv_tuning_score_landscape.png"
        )

    tables["trace_df"].to_csv(data_paths["tuning_trace_csv"], index=False)
    feature_summary_df.to_csv(data_paths["feature_summary_csv"], index=False)
    winners_df.to_csv(data_paths["fold_winners_csv"], index=False)
    for model_type, model_df in selected_outer_trials_by_model.items():
        model_df.to_csv(
            data_paths[f"{model_type}_selected_tuning_outer_{metric_slug}_trials_csv"],
            index=False,
        )
    for model_type, model_df in inner_cv_trials_by_model.items():
        model_df.to_csv(
            data_paths[f"{model_type}_inner_cv_tuning_trials_csv"],
            index=False,
        )
    for model_type, model_df in inner_cv_landscape_by_model.items():
        model_df.to_csv(
            data_paths[f"{model_type}_inner_cv_tuning_score_landscape_csv"],
            index=False,
        )

    if write_plots:
        for model_type, model_df in selected_outer_trials_by_model.items():
            write_selected_tuning_outer_metric_parallel_coordinates_plot(
                selected_trials_df=model_df,
                output_path=plot_paths[
                    f"{model_type}_selected_tuning_outer_{metric_slug}_parallel_coordinates_png"
                ],
                model_type=model_type,
                metric=metric,
            )
        for model_type, model_df in inner_cv_landscape_by_model.items():
            write_inner_cv_tuning_score_landscape_plot(
                landscape_df=model_df,
                output_path=plot_paths[
                    f"{model_type}_inner_cv_tuning_score_landscape_png"
                ],
                model_type=model_type,
                metric=metric,
            )
        write_model_family_feature_dotplot_static_plot(
            feature_summary_df=feature_summary_df,
            output_path=plot_paths["feature_dotplot_png"],
            metric=metric,
        )
        write_fold_winner_stability_static_plot(
            winners_df=winners_df,
            output_path=plot_paths["fold_winner_stability_png"],
            metric=metric,
        )

    return {
        "output_dir": output_dir,
        "runs": runs,
        "data_paths": data_paths,
        "plot_paths": plot_paths,
    }
