"""Selected-model behavior diagnostics built from held-out OOF predictions."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from flumolscreen.visualization.chemical_space import (
    format_target_class_label,
    format_target_label,
)

SELECTED_MANIFEST_SUFFIX = "_selected_model.yml"
SELECTED_SUMMARY_SUFFIX = "_selected_model_summary.csv"
SELECTED_OOF_SUFFIX = "_selected_oof_predictions.csv"

DEFAULT_DISAGREEMENT_COLUMN = "branch_rank_sd"
DISAGREEMENT_COLUMNS = (
    "branch_rank_sd",
    "method_rank_sd",
    "method_rank_range",
    "structure_minus_pose_rank",
    "sequence_minus_structure_rank",
)
DISAGREEMENT_DISPLAY_NAMES = {
    "branch_rank_sd": "Branch percentile-rank SD",
    "method_rank_sd": "Method percentile-rank SD",
    "method_rank_range": "Method percentile-rank range",
    "structure_minus_pose_rank": "Structure minus pose percentile rank",
    "sequence_minus_structure_rank": "Sequence minus structure percentile rank",
}
PLOTLY_TARGET_COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
)
REQUIRED_OOF_COLUMNS = (
    "dataset_label",
    "id",
    "target",
    "label_pkd",
    "pred_mean",
)
OPTIONAL_NUMERIC_COLUMNS = (
    "pred_err",
    "pred_lower",
    "pred_upper",
    "residual",
    "abs_error",
    "covered",
    "pred_score_decile",
    "target_pred_score_decile",
    "scaffold_library_count",
    "scaffold_assayed_count",
    "scaffold_coverage_fraction",
    *DISAGREEMENT_COLUMNS,
)
PREDICTION_SUMMARY_COLUMNS = (
    "summary_level",
    "dataset_label",
    "dataset_display",
    "target",
    "target_display",
    "n",
    "spearman",
    "pearson",
    "rmse",
    "mae",
    "mean_residual",
    "median_abs_error",
    "coverage",
)


@dataclass(frozen=True)
class SelectedModelRun:
    """Paths for one complete selected-model artifact set."""

    dataset_label: str
    selected_dir: Path
    manifest_path: Path
    summary_path: Path
    oof_predictions_path: Path


def resolve_output_dir(results_dir: Path | str, round_id: str) -> Path:
    """Return the standard output directory for selected-model diagnostics."""
    return Path(results_dir) / round_id / "visualizations" / "selected_models"


def _selected_dir(results_dir: Path | str, round_id: str) -> Path:
    return Path(results_dir) / round_id / "selected_models"


def _selected_outputs_error(selected_dir: Path) -> str:
    return (
        f"Selected OOF prediction files were not found in {selected_dir}. "
        "Rerun the learner with write_selected_outputs: true and "
        "write_selected_oof_diagnostics: true."
    )


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: tuple[str, ...] | list[str],
    table_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{table_name} is missing required column(s): {missing_columns}")


def _ensure_parent_dir(output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def format_dataset_label(dataset_label: str) -> str:
    """Return a display label for a selected-model dataset label."""
    return format_target_class_label(str(dataset_label))


def discover_selected_model_runs(
    results_dir: Path | str = "results",
    round_id: str = "round_synthetic",
    dataset_labels: list[str] | tuple[str, ...] | None = None,
) -> list[SelectedModelRun]:
    """Discover complete selected-model artifact sets for one result round."""
    selected_dir = _selected_dir(results_dir=results_dir, round_id=round_id)
    if not selected_dir.is_dir():
        raise FileNotFoundError(_selected_outputs_error(selected_dir))

    requested_labels = {str(label) for label in dataset_labels or []}
    oof_paths = sorted(selected_dir.glob(f"*{SELECTED_OOF_SUFFIX}"))
    if not oof_paths:
        raise FileNotFoundError(_selected_outputs_error(selected_dir))

    runs: list[SelectedModelRun] = []
    missing_paths: list[Path] = []
    discovered_labels: set[str] = set()
    for oof_predictions_path in oof_paths:
        dataset_label = oof_predictions_path.name[: -len(SELECTED_OOF_SUFFIX)]
        discovered_labels.add(dataset_label)
        if requested_labels and dataset_label not in requested_labels:
            continue

        manifest_path = selected_dir / f"{dataset_label}{SELECTED_MANIFEST_SUFFIX}"
        summary_path = selected_dir / f"{dataset_label}{SELECTED_SUMMARY_SUFFIX}"
        for path in [manifest_path, summary_path]:
            if not path.exists():
                missing_paths.append(path)
        if missing_paths and (manifest_path in missing_paths or summary_path in missing_paths):
            continue

        runs.append(
            SelectedModelRun(
                dataset_label=dataset_label,
                selected_dir=selected_dir,
                manifest_path=manifest_path,
                summary_path=summary_path,
                oof_predictions_path=oof_predictions_path,
            )
        )

    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Incomplete selected-model artifact set(s); missing: " f"{missing_text}"
        )
    if not runs:
        if requested_labels:
            missing_labels = sorted(requested_labels.difference(discovered_labels))
            raise FileNotFoundError(
                f"No selected OOF prediction files found for dataset label(s) "
                f"{sorted(requested_labels)} in {selected_dir}. Missing requested "
                f"label(s): {missing_labels}. Rerun the learner with "
                "write_selected_outputs: true and write_selected_oof_diagnostics: true."
            )
        raise FileNotFoundError(_selected_outputs_error(selected_dir))
    return runs


def load_selected_manifests(runs: list[SelectedModelRun]) -> dict[str, dict[str, Any]]:
    """Load selected-model YAML manifests keyed by dataset label."""
    manifests: dict[str, dict[str, Any]] = {}
    for run in runs:
        with run.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        manifests[run.dataset_label] = manifest
    return manifests


def load_selected_summaries(runs: list[SelectedModelRun]) -> pd.DataFrame:
    """Load selected-model summary tables into one labeled dataframe."""
    summary_frames = []
    for run in runs:
        summary_df = pd.read_csv(run.summary_path)
        if "dataset_label" not in summary_df.columns:
            summary_df.insert(0, "dataset_label", run.dataset_label)
        summary_df.insert(0, "selected_summary_path", str(run.summary_path))
        summary_frames.append(summary_df)
    if not summary_frames:
        raise ValueError("runs must contain at least one selected-model run")
    return pd.concat(summary_frames, ignore_index=True)


def load_selected_oof_predictions(runs: list[SelectedModelRun]) -> pd.DataFrame:
    """Load selected OOF predictions into one dataframe and validate the schema."""
    oof_frames = []
    for run in runs:
        oof_df = pd.read_csv(run.oof_predictions_path)
        if "dataset_label" not in oof_df.columns:
            oof_df.insert(0, "dataset_label", run.dataset_label)
        oof_df.insert(0, "selected_oof_predictions_path", str(run.oof_predictions_path))
        _validate_required_columns(
            oof_df,
            REQUIRED_OOF_COLUMNS,
            table_name=str(run.oof_predictions_path),
        )
        oof_frames.append(oof_df)
    if not oof_frames:
        raise ValueError("runs must contain at least one selected-model run")
    return pd.concat(oof_frames, ignore_index=True)


def available_disagreement_columns(df: pd.DataFrame) -> list[str]:
    """Return known disagreement columns present with at least one non-null value."""
    available = []
    for column in DISAGREEMENT_COLUMNS:
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").notna().any():
            available.append(column)
    return available


def _coerce_numeric_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def prepare_selected_oof_predictions(
    oof_df: pd.DataFrame,
    disagreement_column: str = DEFAULT_DISAGREEMENT_COLUMN,
) -> pd.DataFrame:
    """Standardize selected OOF predictions for summaries and plotting."""
    _validate_required_columns(oof_df, REQUIRED_OOF_COLUMNS, table_name="oof_df")
    if disagreement_column not in oof_df.columns:
        available = available_disagreement_columns(oof_df)
        raise ValueError(
            f"Requested disagreement column '{disagreement_column}' is missing. "
            f"Available disagreement column(s): {available}"
        )

    out = oof_df.copy()
    out["dataset_label"] = out["dataset_label"].astype(str)
    out["target"] = out["target"].astype(str)
    out = _coerce_numeric_columns(
        out,
        ("label_pkd", "pred_mean", *OPTIONAL_NUMERIC_COLUMNS),
    )
    if out[disagreement_column].notna().sum() == 0:
        available = available_disagreement_columns(out)
        raise ValueError(
            f"Requested disagreement column '{disagreement_column}' has no "
            f"non-null values. Available disagreement column(s): {available}"
        )

    out["residual"] = out["label_pkd"] - out["pred_mean"]
    out["abs_error"] = out["residual"].abs()
    out["selected_disagreement_column"] = disagreement_column
    out["selected_disagreement"] = out[disagreement_column]
    out["selected_disagreement_bin"] = (
        out.groupby("dataset_label", group_keys=False)["selected_disagreement"]
        .apply(lambda values: _assign_disagreement_bins(values, max_bins=10))
        .astype("Int64")
    )
    out["dataset_display"] = out["dataset_label"].map(format_dataset_label)
    out["target_display"] = out["target"].map(format_target_label)
    if "target_class" in out.columns:
        out["target_class_display"] = out["target_class"].map(format_dataset_label)
    if "strain" in out.columns:
        out["strain"] = out["strain"].astype(str)
    return out


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    pair_df = pd.DataFrame({"x": x, "y": y}).apply(pd.to_numeric, errors="coerce")
    pair_df = pair_df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair_df) < 2:
        return float("nan")
    if pair_df["x"].nunique() < 2 or pair_df["y"].nunique() < 2:
        return float("nan")
    corr = pair_df["x"].corr(pair_df["y"], method=method)
    return float(corr) if pd.notna(corr) else float("nan")


def _prediction_metrics(group_df: pd.DataFrame) -> dict[str, float | int]:
    metric_df = group_df.loc[:, ["label_pkd", "pred_mean", "residual", "abs_error"]].copy()
    for column in metric_df.columns:
        metric_df[column] = pd.to_numeric(metric_df[column], errors="coerce")
    valid_pair = metric_df[["label_pkd", "pred_mean"]].dropna()
    residuals = metric_df["residual"].dropna()
    abs_errors = metric_df["abs_error"].dropna()
    coverage = (
        pd.to_numeric(group_df["covered"], errors="coerce").dropna()
        if "covered" in group_df.columns
        else pd.Series(dtype=float)
    )
    return {
        "n": int(len(valid_pair)),
        "spearman": _safe_corr(group_df["label_pkd"], group_df["pred_mean"], "spearman"),
        "pearson": _safe_corr(group_df["label_pkd"], group_df["pred_mean"], "pearson"),
        "rmse": (
            float(np.sqrt(np.mean(np.square(residuals))))
            if len(residuals) > 0
            else float("nan")
        ),
        "mae": float(abs_errors.mean()) if len(abs_errors) > 0 else float("nan"),
        "mean_residual": (
            float(residuals.mean()) if len(residuals) > 0 else float("nan")
        ),
        "median_abs_error": (
            float(abs_errors.median()) if len(abs_errors) > 0 else float("nan")
        ),
        "coverage": float(coverage.mean()) if len(coverage) > 0 else float("nan"),
    }


def build_prediction_summary(oof_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-dataset and per-target selected OOF prediction metrics."""
    _validate_required_columns(
        oof_df,
        ["dataset_label", "target", "label_pkd", "pred_mean", "residual", "abs_error"],
        table_name="oof_df",
    )

    rows: list[dict[str, Any]] = []
    dataset_order = sorted(oof_df["dataset_label"].astype(str).unique(), key=format_dataset_label)
    for dataset_label in dataset_order:
        dataset_df = oof_df.loc[oof_df["dataset_label"].astype(str).eq(dataset_label)]
        rows.append(
            {
                "summary_level": "dataset",
                "dataset_label": dataset_label,
                "dataset_display": format_dataset_label(dataset_label),
                "target": "all",
                "target_display": "All targets",
                **_prediction_metrics(dataset_df),
            }
        )

        target_order = sorted(
            dataset_df["target"].astype(str).unique(),
            key=format_target_label,
        )
        for target in target_order:
            target_df = dataset_df.loc[dataset_df["target"].astype(str).eq(target)]
            rows.append(
                {
                    "summary_level": "target",
                    "dataset_label": dataset_label,
                    "dataset_display": format_dataset_label(dataset_label),
                    "target": target,
                    "target_display": format_target_label(target),
                    **_prediction_metrics(target_df),
                }
            )

    return pd.DataFrame(rows, columns=PREDICTION_SUMMARY_COLUMNS)


def _assign_disagreement_bins(values: pd.Series, max_bins: int) -> pd.Series:
    bins = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid_values = pd.to_numeric(values, errors="coerce").dropna()
    if valid_values.empty:
        return bins
    n_bins = min(max_bins, len(valid_values), valid_values.nunique())
    if n_bins <= 1:
        bins.loc[valid_values.index] = 1
        return bins
    ranks = valid_values.rank(method="first", ascending=True)
    qcut_bins = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    bins.loc[valid_values.index] = (qcut_bins.astype(int) + 1).astype("Int64")
    return bins


def build_disagreement_summary(
    oof_df: pd.DataFrame,
    disagreement_column: str = DEFAULT_DISAGREEMENT_COLUMN,
    max_bins: int = 10,
) -> pd.DataFrame:
    """Build quantile-bin summaries for disagreement versus residual behavior."""
    _validate_required_columns(
        oof_df,
        [
            "dataset_label",
            "selected_disagreement",
            "residual",
            "abs_error",
        ],
        table_name="oof_df",
    )
    if max_bins < 1:
        raise ValueError("max_bins must be at least 1")

    rows: list[dict[str, Any]] = []
    dataset_order = sorted(oof_df["dataset_label"].astype(str).unique(), key=format_dataset_label)
    for dataset_label in dataset_order:
        dataset_df = oof_df.loc[oof_df["dataset_label"].astype(str).eq(dataset_label)].copy()
        dataset_df["disagreement_bin"] = _assign_disagreement_bins(
            dataset_df["selected_disagreement"],
            max_bins=max_bins,
        )
        dataset_df = dataset_df.loc[dataset_df["disagreement_bin"].notna()].copy()
        for bin_id, bin_df in dataset_df.groupby("disagreement_bin", sort=True):
            disagreement = pd.to_numeric(
                bin_df["selected_disagreement"],
                errors="coerce",
            ).dropna()
            residuals = pd.to_numeric(bin_df["residual"], errors="coerce").dropna()
            abs_errors = pd.to_numeric(bin_df["abs_error"], errors="coerce").dropna()
            rows.append(
                {
                    "dataset_label": dataset_label,
                    "dataset_display": format_dataset_label(dataset_label),
                    "disagreement_column": disagreement_column,
                    "disagreement_bin": int(bin_id),
                    "n": int(len(bin_df)),
                    "disagreement_min": (
                        float(disagreement.min()) if len(disagreement) else float("nan")
                    ),
                    "disagreement_q25": (
                        float(disagreement.quantile(0.25))
                        if len(disagreement)
                        else float("nan")
                    ),
                    "disagreement_median": (
                        float(disagreement.median()) if len(disagreement) else float("nan")
                    ),
                    "disagreement_q75": (
                        float(disagreement.quantile(0.75))
                        if len(disagreement)
                        else float("nan")
                    ),
                    "disagreement_max": (
                        float(disagreement.max()) if len(disagreement) else float("nan")
                    ),
                    "residual_q25": (
                        float(residuals.quantile(0.25))
                        if len(residuals)
                        else float("nan")
                    ),
                    "residual_median": (
                        float(residuals.median()) if len(residuals) else float("nan")
                    ),
                    "residual_q75": (
                        float(residuals.quantile(0.75))
                        if len(residuals)
                        else float("nan")
                    ),
                    "abs_error_q25": (
                        float(abs_errors.quantile(0.25))
                        if len(abs_errors)
                        else float("nan")
                    ),
                    "abs_error_median": (
                        float(abs_errors.median()) if len(abs_errors) else float("nan")
                    ),
                    "abs_error_q75": (
                        float(abs_errors.quantile(0.75))
                        if len(abs_errors)
                        else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise ImportError("matplotlib is required to write selected-model plots.") from error
    return plt, Line2D


def _load_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise ImportError(
            "plotly is required to write selected-model HTML plots. "
            "Install plotly or rerun with --skip-plots."
        ) from error
    return go, make_subplots


def _target_palette(plt, targets: list[str]) -> dict[str, Any]:
    if not targets:
        return {}
    cmap_name = "tab20" if len(targets) > 10 else "tab10"
    cmap = plt.get_cmap(cmap_name)
    denom = max(len(targets), 1)
    return {target: cmap(idx % cmap.N / max(denom - 1, 1)) for idx, target in enumerate(targets)}


def _target_plotly_palette(targets: list[str]) -> dict[str, str]:
    return {
        target: PLOTLY_TARGET_COLORS[idx % len(PLOTLY_TARGET_COLORS)]
        for idx, target in enumerate(targets)
    }


def _dataset_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["dataset_label"].astype(str).unique(), key=format_dataset_label)


def _target_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["target"].astype(str).unique(), key=format_target_label)


def _panel_grid(n_panels: int) -> tuple[int, int]:
    n_cols = min(2, max(1, n_panels))
    n_rows = ceil(n_panels / n_cols)
    return n_rows, n_cols


def _subplot_position(panel_idx: int, n_cols: int) -> tuple[int, int]:
    return (panel_idx // n_cols) + 1, (panel_idx % n_cols) + 1


def _plotly_axis_id(axis: str, panel_number: int) -> str:
    return axis if panel_number == 1 else f"{axis}{panel_number}"


def _plotly_axis_ref(axis: str, panel_number: int, ref_type: str) -> str:
    return f"{_plotly_axis_id(axis, panel_number)} {ref_type}"


def _finite_limits(values: pd.Series | np.ndarray) -> tuple[float, float] | None:
    finite = pd.to_numeric(pd.Series(values), errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    if finite.empty:
        return None
    lower = float(finite.min())
    upper = float(finite.max())
    if lower == upper:
        padding = max(abs(lower) * 0.05, 0.25)
    else:
        padding = max((upper - lower) * 0.06, 0.10)
    return lower - padding, upper + padding


def _format_metric(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _lower_first(value: str) -> str:
    return value[:1].lower() + value[1:] if value else value


def format_disagreement_axis_label(disagreement_column: str) -> str:
    """Return a display label for a disagreement diagnostic column."""
    return DISAGREEMENT_DISPLAY_NAMES.get(
        str(disagreement_column),
        str(disagreement_column).replace("_", " ").capitalize(),
    )


def _annotation_for_dataset(
    prediction_summary_df: pd.DataFrame,
    dataset_label: str,
) -> str:
    summary_rows = prediction_summary_df.loc[
        prediction_summary_df["summary_level"].astype(str).eq("dataset")
        & prediction_summary_df["dataset_label"].astype(str).eq(dataset_label)
    ]
    if summary_rows.empty:
        return ""
    row = summary_rows.iloc[0]
    annotation_parts = [
        f"n={int(row['n'])}",
        f"Spearman={_format_metric(row['spearman'])}",
        f"RMSE={_format_metric(row['rmse'])}",
        f"MAE={_format_metric(row['mae'])}",
        f"mean residual={_format_metric(row['mean_residual'])}",
    ]
    return "\n".join(annotation_parts)


def _html_prediction_annotation_for_dataset(
    prediction_summary_df: pd.DataFrame,
    dataset_label: str,
) -> str:
    summary_rows = prediction_summary_df.loc[
        prediction_summary_df["summary_level"].astype(str).eq("dataset")
        & prediction_summary_df["dataset_label"].astype(str).eq(dataset_label)
    ]
    if summary_rows.empty:
        return ""
    row = summary_rows.iloc[0]
    return (
        f"n={int(row['n'])}<br>"
        f"<i>r_s</i>={_format_metric(row['spearman'])}<br>"
        f"MAE={_format_metric(row['mae'])}"
    )


def _oof_customdata(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    custom_df = df.copy()
    for column in columns:
        if column not in custom_df.columns:
            custom_df[column] = ""
    return custom_df.loc[:, columns].fillna("").to_numpy()


def _oof_prediction_hover_columns() -> list[str]:
    return [
        "id",
        "target_display",
        "label_pkd",
        "pred_mean",
        "residual",
        "outer_fold",
    ]


def _oof_prediction_hover_template() -> str:
    return (
        "ID=%{customdata[0]}<br>"
        "Target=%{customdata[1]}<br>"
        "Observed pKd=%{customdata[2]:.3f}<br>"
        "Predicted pKd=%{customdata[3]:.3f}<br>"
        "Residual=%{customdata[4]:.3f}<br>"
        "Outer fold=%{customdata[5]}<extra></extra>"
    )


def _oof_residual_hover_columns() -> list[str]:
    return [
        "id",
        "target_display",
        "selected_disagreement",
        "residual",
        "label_pkd",
        "pred_mean",
        "outer_fold",
    ]


def _oof_residual_hover_template(disagreement_label: str) -> str:
    return (
        "ID=%{customdata[0]}<br>"
        "Target=%{customdata[1]}<br>"
        f"{disagreement_label}=%{{customdata[2]:.3f}}<br>"
        "Residual=%{customdata[3]:.3f}<br>"
        "Observed pKd=%{customdata[4]:.3f}<br>"
        "Predicted pKd=%{customdata[5]:.3f}<br>"
        "Outer fold=%{customdata[6]}<extra></extra>"
    )


def write_predicted_vs_observed_pkd_html_plot(
    oof_df: pd.DataFrame,
    prediction_summary_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Write faceted Plotly predicted versus observed pKd OOF diagnostics."""
    go, make_subplots = _load_plotly()
    output_path = _ensure_parent_dir(output_path)
    _validate_required_columns(
        oof_df,
        ["dataset_label", "target", "label_pkd", "pred_mean", "residual"],
        table_name="oof_df",
    )

    plot_df = oof_df.copy()
    plot_df["label_pkd"] = pd.to_numeric(plot_df["label_pkd"], errors="coerce")
    plot_df["pred_mean"] = pd.to_numeric(plot_df["pred_mean"], errors="coerce")
    plot_df = plot_df.loc[plot_df[["label_pkd", "pred_mean"]].notna().all(axis=1)]
    if plot_df.empty:
        raise ValueError("No finite label_pkd/pred_mean pairs available for plotting.")

    dataset_order = _dataset_order(plot_df)
    target_order = _target_order(plot_df)
    palette = _target_plotly_palette(target_order)
    n_rows, n_cols = _panel_grid(len(dataset_order))
    subplot_titles = [format_dataset_label(label) for label in dataset_order]
    subplot_titles.extend([""] * (n_rows * n_cols - len(subplot_titles)))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    legend_targets: set[str] = set()
    hover_columns = _oof_prediction_hover_columns()
    for panel_idx, dataset_label in enumerate(dataset_order):
        row, col = _subplot_position(panel_idx, n_cols)
        panel_number = panel_idx + 1
        dataset_df = plot_df.loc[plot_df["dataset_label"].astype(str).eq(dataset_label)]
        limits = _finite_limits(
            pd.concat([dataset_df["label_pkd"], dataset_df["pred_mean"]], ignore_index=True)
        )
        if limits is None:
            continue

        fig.add_shape(
            type="line",
            x0=limits[0],
            y0=limits[0],
            x1=limits[1],
            y1=limits[1],
            line={"color": "#4d4d4d", "width": 1.2, "dash": "dash"},
            row=row,
            col=col,
        )
        for target in target_order:
            target_df = dataset_df.loc[dataset_df["target"].astype(str).eq(target)]
            if target_df.empty:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=target_df["label_pkd"],
                    y=target_df["pred_mean"],
                    mode="markers",
                    name=format_target_label(target),
                    legendgroup=target,
                    showlegend=target not in legend_targets,
                    marker={"size": 7, "color": palette[target], "opacity": 0.62},
                    customdata=_oof_customdata(target_df, hover_columns),
                    hovertemplate=_oof_prediction_hover_template(),
                ),
                row=row,
                col=col,
            )
            legend_targets.add(target)

        annotation = _html_prediction_annotation_for_dataset(
            prediction_summary_df,
            dataset_label,
        )
        if annotation:
            fig.add_annotation(
                text=annotation,
                x=0.03,
                y=0.97,
                xref=_plotly_axis_ref("x", panel_number, "domain"),
                yref=_plotly_axis_ref("y", panel_number, "domain"),
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.76)",
                bordercolor="rgba(0,0,0,0)",
                font={"size": 11},
            )
        fig.update_xaxes(
            title_text="Observed pKd",
            range=list(limits),
            showgrid=True,
            zeroline=False,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            title_text="Predicted pKd",
            range=list(limits),
            showgrid=True,
            zeroline=False,
            scaleanchor=_plotly_axis_id("x", panel_number),
            scaleratio=1,
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Selected-model OOF predicted versus observed pKd",
        template="plotly_white",
        width=max(760, 520 * n_cols),
        height=max(480, 430 * n_rows + 120),
        legend_title_text="Target",
        margin={"l": 70, "r": 30, "t": 95, "b": 70},
    )
    fig.write_html(output_path)
    return output_path


def write_predicted_vs_observed_pkd_plot(
    oof_df: pd.DataFrame,
    prediction_summary_df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Write faceted predicted versus observed pKd selected OOF diagnostics."""
    plt, Line2D = _load_pyplot()
    output_path = _ensure_parent_dir(output_path)
    _validate_required_columns(
        oof_df,
        ["dataset_label", "target", "label_pkd", "pred_mean"],
        table_name="oof_df",
    )

    plot_df = oof_df.copy()
    plot_df["label_pkd"] = pd.to_numeric(plot_df["label_pkd"], errors="coerce")
    plot_df["pred_mean"] = pd.to_numeric(plot_df["pred_mean"], errors="coerce")
    plot_df = plot_df.loc[plot_df[["label_pkd", "pred_mean"]].notna().all(axis=1)]
    if plot_df.empty:
        raise ValueError("No finite label_pkd/pred_mean pairs available for plotting.")

    dataset_order = _dataset_order(plot_df)
    target_order = _target_order(plot_df)
    palette = _target_palette(plt, target_order)
    n_rows, n_cols = _panel_grid(len(dataset_order))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.0 * n_cols, 4.7 * n_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for ax, dataset_label in zip(flat_axes, dataset_order):
        dataset_df = plot_df.loc[plot_df["dataset_label"].astype(str).eq(dataset_label)]
        limits = _finite_limits(
            pd.concat([dataset_df["label_pkd"], dataset_df["pred_mean"]], ignore_index=True)
        )
        if limits is None:
            ax.set_axis_off()
            continue

        for target in target_order:
            target_df = dataset_df.loc[dataset_df["target"].astype(str).eq(target)]
            if target_df.empty:
                continue
            ax.scatter(
                target_df["label_pkd"],
                target_df["pred_mean"],
                s=18,
                alpha=0.58,
                color=palette[target],
                edgecolors="none",
                label=format_target_label(target),
                zorder=3,
            )
        ax.plot(limits, limits, color="#444444", linewidth=1.0, linestyle="--", zorder=2)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(format_dataset_label(dataset_label), loc="left", fontweight="bold")
        ax.set_xlabel("Observed pKd")
        ax.set_ylabel("Predicted pKd")
        ax.grid(color="#d0d0d0", linewidth=0.6, alpha=0.55)
        annotation = _annotation_for_dataset(prediction_summary_df, dataset_label)
        if annotation:
            ax.text(
                0.03,
                0.97,
                annotation,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "alpha": 0.72, "linewidth": 0},
            )
        for spine_name in ["top", "right"]:
            ax.spines[spine_name].set_visible(False)

    for ax in flat_axes[len(dataset_order) :]:
        ax.set_axis_off()

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[target],
            markeredgecolor="none",
            markersize=7,
            label=format_target_label(target),
        )
        for target in target_order
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            title="Target",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(len(legend_handles), 4),
            frameon=False,
        )
    fig.suptitle("Selected-model OOF predictions", y=0.995)
    fig.subplots_adjust(top=0.86, hspace=0.34, wspace=0.22)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _disagreement_axis_label(disagreement_column: str) -> str:
    return format_disagreement_axis_label(disagreement_column)


def _overlay_disagreement_summary(
    ax,
    summary_df: pd.DataFrame,
    y_median_col: str,
    y_q25_col: str,
    y_q75_col: str,
) -> None:
    if summary_df.empty:
        return
    line_df = summary_df.dropna(subset=["disagreement_median", y_median_col]).copy()
    if line_df.empty:
        return
    x = line_df["disagreement_median"].astype(float).to_numpy()
    y = line_df[y_median_col].astype(float).to_numpy()
    ax.plot(x, y, color="#1f1f1f", linewidth=1.8, marker="o", markersize=4, zorder=5)
    band_df = line_df.dropna(subset=[y_q25_col, y_q75_col])
    if not band_df.empty:
        ax.fill_between(
            band_df["disagreement_median"].astype(float).to_numpy(),
            band_df[y_q25_col].astype(float).to_numpy(),
            band_df[y_q75_col].astype(float).to_numpy(),
            color="#1f1f1f",
            alpha=0.16,
            linewidth=0,
            zorder=4,
        )


def _write_disagreement_iqr_band(
    fig,
    go,
    summary_df: pd.DataFrame,
    row: int,
    col: int,
) -> None:
    band_df = summary_df.dropna(
        subset=["disagreement_median", "residual_q25", "residual_q75"]
    ).copy()
    if band_df.empty:
        return
    x_values = band_df["disagreement_median"].astype(float).to_list()
    fig.add_trace(
        go.Scatter(
            x=[*x_values, *reversed(x_values)],
            y=[
                *band_df["residual_q25"].astype(float).to_list(),
                *reversed(band_df["residual_q75"].astype(float).to_list()),
            ],
            mode="lines",
            line={"width": 0, "color": "rgba(31,31,31,0)"},
            fill="toself",
            fillcolor="rgba(31,31,31,0.16)",
            hoverinfo="skip",
            showlegend=False,
            name="Residual IQR",
        ),
        row=row,
        col=col,
    )


def _write_disagreement_median_line(
    fig,
    go,
    summary_df: pd.DataFrame,
    row: int,
    col: int,
) -> None:
    line_df = summary_df.dropna(
        subset=["disagreement_median", "residual_median"]
    ).copy()
    if line_df.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=line_df["disagreement_median"],
            y=line_df["residual_median"],
            mode="lines+markers",
            line={"color": "#1f1f1f", "width": 2},
            marker={"size": 6, "color": "#1f1f1f"},
            hovertemplate=(
                "Bin median disagreement=%{x:.3f}<br>"
                "Median residual=%{y:.3f}<extra></extra>"
            ),
            showlegend=False,
            name="Binned median residual",
        ),
        row=row,
        col=col,
    )


def write_residuals_vs_branch_disagreement_html_plot(
    oof_df: pd.DataFrame,
    disagreement_summary_df: pd.DataFrame,
    output_path: Path | str,
    disagreement_column: str = DEFAULT_DISAGREEMENT_COLUMN,
) -> Path:
    """Write faceted Plotly OOF residuals versus branch disagreement."""
    go, make_subplots = _load_plotly()
    output_path = _ensure_parent_dir(output_path)
    _validate_required_columns(
        oof_df,
        [
            "dataset_label",
            "target",
            "selected_disagreement",
            "residual",
            "label_pkd",
            "pred_mean",
        ],
        table_name="oof_df",
    )

    plot_df = oof_df.copy()
    for column in ["selected_disagreement", "residual", "label_pkd", "pred_mean"]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.loc[plot_df[["selected_disagreement", "residual"]].notna().all(axis=1)]
    dataset_order = _dataset_order(oof_df)
    target_order = _target_order(oof_df)
    palette = _target_plotly_palette(target_order)
    n_cols = max(1, len(dataset_order))
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=[format_dataset_label(label) for label in dataset_order],
        horizontal_spacing=0.06,
    )

    disagreement_label = format_disagreement_axis_label(disagreement_column)
    title_disagreement_label = _lower_first(disagreement_label)
    residual_limits = _finite_limits(plot_df["residual"]) if not plot_df.empty else None
    if residual_limits is not None:
        residual_bound = max(abs(residual_limits[0]), abs(residual_limits[1]))
        residual_limits = (-residual_bound, residual_bound)

    legend_targets: set[str] = set()
    hover_columns = _oof_residual_hover_columns()
    for panel_idx, dataset_label in enumerate(dataset_order):
        col = panel_idx + 1
        dataset_df = plot_df.loc[plot_df["dataset_label"].astype(str).eq(dataset_label)]
        summary_df = disagreement_summary_df.loc[
            disagreement_summary_df["dataset_label"].astype(str).eq(dataset_label)
        ].copy()
        x_limits = (
            _finite_limits(dataset_df["selected_disagreement"])
            if not dataset_df.empty
            else None
        )

        if x_limits is not None:
            fig.add_shape(
                type="line",
                x0=x_limits[0],
                y0=0.0,
                x1=x_limits[1],
                y1=0.0,
                line={"color": "#555555", "width": 1.1, "dash": "dash"},
                row=1,
                col=col,
            )
        _write_disagreement_iqr_band(fig, go, summary_df, row=1, col=col)
        for target in target_order:
            target_df = dataset_df.loc[dataset_df["target"].astype(str).eq(target)]
            if target_df.empty:
                continue
            fig.add_trace(
                go.Scattergl(
                    x=target_df["selected_disagreement"],
                    y=target_df["residual"],
                    mode="markers",
                    name=format_target_label(target),
                    legendgroup=target,
                    showlegend=target not in legend_targets,
                    marker={"size": 7, "color": palette[target], "opacity": 0.48},
                    customdata=_oof_customdata(target_df, hover_columns),
                    hovertemplate=_oof_residual_hover_template(disagreement_label),
                ),
                row=1,
                col=col,
            )
            legend_targets.add(target)
        _write_disagreement_median_line(fig, go, summary_df, row=1, col=col)
        xaxis_kwargs = {
            "title_text": disagreement_label,
            "showgrid": True,
            "zeroline": False,
            "row": 1,
            "col": col,
        }
        if x_limits is not None:
            xaxis_kwargs["range"] = list(x_limits)
        fig.update_xaxes(**xaxis_kwargs)

        yaxis_kwargs = {
            "title_text": "Residual (observed - predicted)" if col == 1 else "",
            "showgrid": True,
            "zeroline": False,
            "row": 1,
            "col": col,
        }
        if residual_limits is not None:
            yaxis_kwargs["range"] = list(residual_limits)
        fig.update_yaxes(**yaxis_kwargs)

    fig.update_layout(
        title=f"Selected-model OOF residuals versus {title_disagreement_label}",
        template="plotly_white",
        width=max(760, 430 * n_cols),
        height=520,
        legend_title_text="Target",
        margin={"l": 70, "r": 30, "t": 95, "b": 70},
    )
    fig.write_html(output_path)
    return output_path


def write_residuals_vs_branch_disagreement_plot(
    oof_df: pd.DataFrame,
    disagreement_summary_df: pd.DataFrame,
    output_path: Path | str,
    disagreement_column: str = DEFAULT_DISAGREEMENT_COLUMN,
) -> Path:
    """Write signed and absolute residuals versus branch disagreement."""
    plt, Line2D = _load_pyplot()
    output_path = _ensure_parent_dir(output_path)
    _validate_required_columns(
        oof_df,
        [
            "dataset_label",
            "target",
            "selected_disagreement",
            "residual",
            "abs_error",
        ],
        table_name="oof_df",
    )

    plot_df = oof_df.copy()
    for column in ["selected_disagreement", "residual", "abs_error"]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.loc[plot_df["selected_disagreement"].notna()].copy()

    dataset_order = _dataset_order(oof_df)
    target_order = _target_order(oof_df)
    palette = _target_palette(plt, target_order)
    fig, axes = plt.subplots(
        2,
        len(dataset_order),
        figsize=(5.2 * len(dataset_order), 7.8),
        squeeze=False,
        sharex=False,
    )

    residual_limits = _finite_limits(plot_df["residual"]) if not plot_df.empty else None
    if residual_limits is not None:
        residual_bound = max(abs(residual_limits[0]), abs(residual_limits[1]))
        residual_limits = (-residual_bound, residual_bound)
    abs_limits = _finite_limits(plot_df["abs_error"]) if not plot_df.empty else None
    if abs_limits is not None:
        abs_limits = (0.0, max(abs_limits[1], 0.1))

    for col_idx, dataset_label in enumerate(dataset_order):
        dataset_df = plot_df.loc[plot_df["dataset_label"].astype(str).eq(dataset_label)]
        summary_df = disagreement_summary_df.loc[
            disagreement_summary_df["dataset_label"].astype(str).eq(dataset_label)
        ].copy()
        top_ax = axes[0, col_idx]
        bottom_ax = axes[1, col_idx]

        if dataset_df.empty:
            for ax in [top_ax, bottom_ax]:
                ax.text(
                    0.5,
                    0.5,
                    "No finite disagreement values",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                )
                ax.set_axis_off()
            continue

        for target in target_order:
            target_df = dataset_df.loc[dataset_df["target"].astype(str).eq(target)]
            if target_df.empty:
                continue
            scatter_kwargs = {
                "x": target_df["selected_disagreement"],
                "s": 16,
                "alpha": 0.42,
                "color": palette[target],
                "edgecolors": "none",
                "zorder": 3,
            }
            top_ax.scatter(y=target_df["residual"], **scatter_kwargs)
            bottom_ax.scatter(y=target_df["abs_error"], **scatter_kwargs)

        top_ax.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--", zorder=2)
        _overlay_disagreement_summary(
            top_ax,
            summary_df,
            y_median_col="residual_median",
            y_q25_col="residual_q25",
            y_q75_col="residual_q75",
        )
        _overlay_disagreement_summary(
            bottom_ax,
            summary_df,
            y_median_col="abs_error_median",
            y_q25_col="abs_error_q25",
            y_q75_col="abs_error_q75",
        )

        top_ax.set_title(format_dataset_label(dataset_label), loc="left", fontweight="bold")
        if col_idx == 0:
            top_ax.set_ylabel("Residual (observed - predicted)")
            bottom_ax.set_ylabel("Absolute error")
        bottom_ax.set_xlabel(_disagreement_axis_label(disagreement_column))
        if residual_limits is not None:
            top_ax.set_ylim(*residual_limits)
        if abs_limits is not None:
            bottom_ax.set_ylim(*abs_limits)
        for ax in [top_ax, bottom_ax]:
            ax.grid(color="#d0d0d0", linewidth=0.6, alpha=0.55)
            for spine_name in ["top", "right"]:
                ax.spines[spine_name].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[target],
            markeredgecolor="none",
            markersize=7,
            label=format_target_label(target),
        )
        for target in target_order
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            title="Target",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=min(len(legend_handles), 4),
            frameon=False,
        )
    fig.suptitle(
        f"Selected-model OOF residuals versus {_disagreement_axis_label(disagreement_column)}",
        y=0.995,
    )
    fig.subplots_adjust(top=0.86, hspace=0.24, wspace=0.24)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_selected_model_diagnostics(
    results_dir: Path | str = "results",
    round_id: str = "round_synthetic",
    dataset_labels: list[str] | None = None,
    disagreement_column: str = DEFAULT_DISAGREEMENT_COLUMN,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Create selected-model behavior diagnostics for one result round."""
    runs = discover_selected_model_runs(
        results_dir=results_dir,
        round_id=round_id,
        dataset_labels=dataset_labels,
    )
    raw_oof_df = load_selected_oof_predictions(runs)
    oof_df = prepare_selected_oof_predictions(
        raw_oof_df,
        disagreement_column=disagreement_column,
    )
    prediction_summary_df = build_prediction_summary(oof_df)
    disagreement_summary_df = build_disagreement_summary(
        oof_df,
        disagreement_column=disagreement_column,
    )

    output_dir = resolve_output_dir(results_dir=results_dir, round_id=round_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_paths = {
        "oof_predictions_csv": output_dir / "selected_model_oof_predictions.csv",
        "prediction_summary_csv": output_dir / "selected_model_prediction_summary.csv",
        "disagreement_summary_csv": output_dir / "selected_model_disagreement_summary.csv",
    }
    plot_paths = {
        "predicted_vs_observed_pkd_html": output_dir / "predicted_vs_observed_pkd.html",
        "residuals_vs_branch_disagreement_html": (
            output_dir / "residuals_vs_branch_disagreement.html"
        ),
    }

    oof_df.to_csv(data_paths["oof_predictions_csv"], index=False)
    prediction_summary_df.to_csv(data_paths["prediction_summary_csv"], index=False)
    disagreement_summary_df.to_csv(
        data_paths["disagreement_summary_csv"],
        index=False,
    )

    if write_plots:
        write_predicted_vs_observed_pkd_html_plot(
            oof_df=oof_df,
            prediction_summary_df=prediction_summary_df,
            output_path=plot_paths["predicted_vs_observed_pkd_html"],
        )
        write_residuals_vs_branch_disagreement_html_plot(
            oof_df=oof_df,
            disagreement_summary_df=disagreement_summary_df,
            output_path=plot_paths["residuals_vs_branch_disagreement_html"],
            disagreement_column=disagreement_column,
        )

    return {
        "output_dir": output_dir,
        "runs": runs,
        "data_paths": data_paths,
        "plot_paths": plot_paths,
    }
