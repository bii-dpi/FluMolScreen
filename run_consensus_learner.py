"""CLI entrypoint for FluMolScreen consensus learner evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from flumolscreen.console import print_dry_run_jobs, print_progress, print_rule
from flumolscreen.run_config import DEFAULT_RUN_CONFIG_PATH, load_run_config

WORKFLOW_ARGUMENT_KEYS = (
    "data_dir",
    "results_dir",
    "train_round_id",
    "target_id",
    "dataset_mode",
    "family_key",
    "comparisons",
    "model_runs",
    "outer_split_type",
    "outer_split_params",
    "tuning_mode",
    "tuning_metric",
    "holdout_validation_fraction",
    "inner_split_type",
    "inner_split_params",
    "tuning_n_trials",
    "tuning_random_seed",
    "inference_mode",
    "calibration_fraction",
    "ensemble_size_m",
    "interval_coverage",
    "inference_random_seed",
    "standardize_features",
    "hit_threshold_pkd",
    "enrichment_top_fractions",
    "precision_at_n_values",
    "output_label",
)

SUMMARY_ARGUMENT_KEYS = (
    "train_round_id",
    "target_id",
    "dataset_mode",
    "family_key",
    "comparisons",
    "model_runs",
    "outer_split_type",
    "tuning_mode",
    "tuning_metric",
    "holdout_validation_fraction",
    "inner_split_type",
    "tuning_n_trials",
    "inference_mode",
    "calibration_fraction",
    "ensemble_size_m",
    "interval_coverage",
    "standardize_features",
    "hit_threshold_pkd",
    "enrichment_top_fractions",
    "precision_at_n_values",
    "output_label",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse learner runner CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run FluMolScreen learner jobs from YAML configuration.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_RUN_CONFIG_PATH),
        help=(
            "Path to a run YAML file. Defaults to configs/runs/current.yml "
            "for IDE-friendly execution."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved jobs without fitting models or writing outputs.",
    )
    return parser.parse_args(argv)


def _workflow_kwargs(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job[key] for key in WORKFLOW_ARGUMENT_KEYS}


def _summary_kwargs(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job[key] for key in SUMMARY_ARGUMENT_KEYS}


def _resolved_config_path(job: dict[str, Any]) -> Path:
    output_dir = Path(job["results_dir"]) / job["train_round_id"] / "configs"
    return output_dir / f"{job['output_label']}_resolved_config.yml"


def save_resolved_job_config(job: dict[str, Any]) -> Path:
    """Save one fully resolved job config beside the run outputs."""
    output_path = _resolved_config_path(job)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(job, handle, sort_keys=False)
    return output_path


def print_dry_run(jobs: list[dict[str, Any]]) -> None:
    """Print the concrete jobs that would be executed."""
    print_dry_run_jobs(jobs)


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one resolved learner job and save its resolved config."""
    from flumolscreen.ml.workflow import print_cv_summary, run_cv_workflow

    resolved_config_path = save_resolved_job_config(job)
    print_progress("[config]", [("saved_resolved_config", resolved_config_path)])

    results = run_cv_workflow(**_workflow_kwargs(job))
    print_cv_summary(results=results, **_summary_kwargs(job))
    return results


def main(argv: list[str] | None = None) -> None:
    """Load a YAML run config and execute its resolved jobs."""
    args = parse_args(argv)
    resolved_run_config = load_run_config(args.config)
    jobs = resolved_run_config["jobs"]

    if args.dry_run:
        print_dry_run(jobs)
        return

    for job in jobs:
        print_rule(f"Running learner job: {job['name']}")
        run_job(job)


if __name__ == "__main__":
    main()
