"""Console presentation helpers for FluMolScreen command-line workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


def _resolve_console(target_console: Console | None = None) -> Console:
    return target_console or console


def format_display_value(value: Any) -> str:
    """Return a compact display string for terminal tables and progress lines."""
    if value is None:
        return "-"
    if isinstance(value, dict):
        return ", ".join(f"{key}={value[key]}" for key in sorted(value)) or "{}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def format_fields(fields: Sequence[tuple[str, Any]]) -> str:
    """Format key/value pairs as a single readable progress payload."""
    return " | ".join(
        f"{name}={format_display_value(value)}" for name, value in fields
    )


def print_rule(title: str, target_console: Console | None = None) -> None:
    """Print a visually separated section heading."""
    _resolve_console(target_console).rule(title)


def print_line(message: str, target_console: Console | None = None) -> None:
    """Print a line without Rich markup interpretation."""
    resolved_console = _resolve_console(target_console)
    resolved_console.print(message, markup=False)
    resolved_console.file.flush()


def print_progress(
    tag: str,
    fields: Sequence[tuple[str, Any]],
    target_console: Console | None = None,
) -> None:
    """Print one compact, tagged progress line."""
    print_line(f"{tag} {format_fields(fields)}", target_console=target_console)


def make_key_value_table(title: str, rows: Sequence[tuple[str, Any]]) -> Table:
    """Build a two-column settings table."""
    table = Table(title=title, box=box.ASCII, show_lines=False, expand=False)
    table.add_column("Field", style="bold", no_wrap=True)
    table.add_column("Value", overflow="fold")
    for key, value in rows:
        table.add_row(key, format_display_value(value))
    return table


def make_list_table(
    title: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> Table:
    """Build a simple table from row values."""
    table = Table(title=title, box=box.ASCII, show_lines=False, expand=False)
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*(format_display_value(value) for value in row))
    return table


def _comparison_feature_summary(comparison: dict[str, Any]) -> str:
    feature_labels = []
    for request in comparison["feature_requests"]:
        label = request["feature_set"]
        if request.get("feature_generator") is not None:
            label = f"{label} ({request['feature_generator']})"
        feature_labels.append(label)
    return ", ".join(feature_labels)


def print_job_summary(
    job: dict[str, Any],
    job_idx: int | None = None,
    target_console: Console | None = None,
) -> None:
    """Print the resolved config for one learner job."""
    resolved_console = _resolve_console(target_console)
    title_prefix = f"Job {job_idx}" if job_idx is not None else "Job"
    print_rule(f"{title_prefix}: {job['name']}", target_console=resolved_console)

    target_field = (
        ("target", job["target"])
        if job["dataset_mode"] == "single_target"
        else ("target_class", job["target_class"])
    )
    resolved_console.print(
        make_key_value_table(
            "Run Settings",
            [
                ("output_label", job["output_label"]),
                ("train_round_id", job["train_round_id"]),
                ("dataset_mode", job["dataset_mode"]),
                target_field,
                ("comparison_preset", job.get("comparison_preset")),
                ("tuning_mode", job.get("tuning_mode")),
                ("tuning_metric", job.get("tuning_metric")),
                ("inference_mode", job.get("inference_mode")),
            ],
        )
    )
    resolved_console.print(
        make_list_table(
            "Comparisons",
            ["Name", "Feature sets"],
            [
                (
                    comparison["name"],
                    _comparison_feature_summary(comparison),
                )
                for comparison in job["comparisons"]
            ],
        )
    )
    resolved_console.print(
        make_list_table(
            "Model Runs",
            ["Model", "Parameters", "Standardize"],
            [
                (
                    model_run["model_type"],
                    model_run.get("model_params", {}),
                    model_run.get("standardize_features", job.get("standardize_features")),
                )
                for model_run in job["model_runs"]
            ],
        )
    )


def print_dry_run_jobs(
    jobs: Sequence[dict[str, Any]],
    target_console: Console | None = None,
) -> None:
    """Print all jobs resolved from a run config."""
    resolved_console = _resolve_console(target_console)
    print_rule("Resolved learner jobs", target_console=resolved_console)
    for job_idx, job in enumerate(jobs, start=1):
        print_job_summary(job, job_idx=job_idx, target_console=resolved_console)
