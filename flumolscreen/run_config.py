"""YAML run configuration loading for FluMolScreen learner workflows."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from flumolscreen.feature_registry import FEATURE_REGISTRY
from flumolscreen.target_registry import TARGET_CLASS_REGISTRY, TARGET_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_CONFIG_PATH = REPO_ROOT / "configs" / "runs" / "current.yml"
DEFAULTS_CONFIG_PATH = REPO_ROOT / "configs" / "defaults.yml"
COMPARISON_PRESET_DIR = REPO_ROOT / "configs" / "comparison_presets"

REQUIRED_SETTING_KEYS = {
    "data_dir",
    "results_dir",
    "train_round_id",
    "outer_split_type",
    "outer_split_params",
    "inner_split_type",
    "inner_split_params",
    "holdout_validation_fraction",
    "tuning_mode",
    "tuning_metric",
    "tuning_n_trials",
    "tuning_random_seed",
    "selection_metric",
    "selection_rule",
    "write_selected_outputs",
    "write_selected_oof_diagnostics",
    "standardize_features",
    "hit_threshold_pkd",
    "enrichment_top_fractions",
    "precision_at_n_values",
    "calibration_fraction",
    "ensemble_size_m",
    "interval_coverage",
    "inference_mode",
    "inference_random_seed",
    "model_runs",
}

RUN_METADATA_KEYS = {"defaults", "settings", "jobs"}
JOB_METADATA_KEYS = {
    "name",
    "dataset_mode",
    "target",
    "targets",
    "target_class",
    "target_classes",
    "comparison_preset",
    "comparisons",
    "settings",
    "output_label",
}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_relative_path(base_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_path.parent / path).resolve()


def _resolve_defaults_path(config_path: Path, run_config: dict[str, Any]) -> Path:
    defaults_value = run_config.get("defaults")
    if defaults_value is None:
        return DEFAULTS_CONFIG_PATH
    if not isinstance(defaults_value, str):
        raise ValueError("'defaults' must be a string path when provided")
    return _resolve_relative_path(config_path, defaults_value)


def _resolve_comparison_preset_path(config_path: Path, preset_name: str) -> Path:
    preset_path = Path(preset_name)
    if preset_path.suffix in {".yml", ".yaml"} or "/" in preset_name:
        return _resolve_relative_path(config_path, preset_name)
    return COMPARISON_PRESET_DIR / f"{preset_name}.yml"


def _validate_settings(settings: dict[str, Any]) -> None:
    missing_keys = sorted(REQUIRED_SETTING_KEYS.difference(settings))
    if missing_keys:
        raise ValueError(f"Missing required learner settings: {missing_keys}")

    model_runs = settings["model_runs"]
    if not isinstance(model_runs, list) or not model_runs:
        raise ValueError("'model_runs' must be a non-empty list")
    for model_run in model_runs:
        if not isinstance(model_run, dict) or "model_type" not in model_run:
            raise ValueError("Each model run must be a mapping with 'model_type'")


def _validate_comparisons(comparisons: Any) -> list[dict[str, Any]]:
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("'comparisons' must be a non-empty list")

    names = []
    validated = deepcopy(comparisons)
    for comparison in validated:
        if not isinstance(comparison, dict):
            raise ValueError("Each comparison must be a mapping")

        comparison_name = comparison.get("name")
        if not isinstance(comparison_name, str) or not comparison_name:
            raise ValueError("Each comparison must define a non-empty 'name'")
        names.append(comparison_name)

        feature_requests = comparison.get("feature_requests")
        if not isinstance(feature_requests, list) or not feature_requests:
            raise ValueError(
                f"Comparison '{comparison_name}' must define feature_requests"
            )

        for request in feature_requests:
            if not isinstance(request, dict):
                raise ValueError(
                    f"Comparison '{comparison_name}' has a non-mapping feature request"
                )
            feature_set = request.get("feature_set")
            if feature_set not in FEATURE_REGISTRY:
                raise ValueError(
                    f"Unknown feature_set '{feature_set}' in comparison "
                    f"'{comparison_name}'"
                )
            base_feature_set = request.get("base_feature_set")
            if (
                base_feature_set is not None
                and base_feature_set not in FEATURE_REGISTRY
            ):
                raise ValueError(
                    f"Unknown base_feature_set '{base_feature_set}' in comparison "
                    f"'{comparison_name}'"
                )

    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Duplicate comparison names: {duplicate_names}")
    return validated


def _load_comparisons(
    config_path: Path,
    job_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    if "comparison_preset" in job_config and "comparisons" in job_config:
        raise ValueError("Use either 'comparison_preset' or 'comparisons', not both")

    if "comparison_preset" in job_config:
        preset_name = job_config["comparison_preset"]
        if not isinstance(preset_name, str) or not preset_name:
            raise ValueError("'comparison_preset' must be a non-empty string")
        preset_path = _resolve_comparison_preset_path(config_path, preset_name)
        preset_config = _load_yaml_mapping(preset_path)
        return _validate_comparisons(preset_config.get("comparisons")), preset_name

    if "comparisons" in job_config:
        return _validate_comparisons(job_config["comparisons"]), None

    raise ValueError("Each job must define 'comparison_preset' or 'comparisons'")


def _as_string_list(value: Any, field_name: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"'{field_name}' must be a non-empty string or list")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"'{field_name}' must contain only non-empty strings")
    return value


def _validate_output_label(output_label: str) -> None:
    if not output_label or "/" in output_label or "\\" in output_label:
        raise ValueError(
            "'output_label' must be non-empty and cannot contain path separators"
        )


def _job_name(raw_name: Any, output_label: str, expanded_value: str, is_batch: bool) -> str:
    if raw_name is None:
        return output_label
    if not isinstance(raw_name, str) or not raw_name:
        raise ValueError("'name' must be a non-empty string when provided")
    if is_batch:
        return f"{raw_name}:{expanded_value}"
    return raw_name


def _build_expanded_job(
    settings: dict[str, Any],
    raw_job: dict[str, Any],
    dataset_mode: str,
    target: str | None,
    target_class: str | None,
    comparisons: list[dict[str, Any]],
    comparison_preset: str | None,
    expanded_value: str,
    is_batch: bool,
) -> dict[str, Any]:
    output_label = raw_job.get("output_label", expanded_value)
    if not isinstance(output_label, str):
        raise ValueError("'output_label' must be a string when provided")
    _validate_output_label(output_label)

    job = deepcopy(settings)
    job.update(
        {
            "name": _job_name(
                raw_name=raw_job.get("name"),
                output_label=output_label,
                expanded_value=expanded_value,
                is_batch=is_batch,
            ),
            "dataset_mode": dataset_mode,
            "target": target,
            "target_class": target_class,
            "comparison_preset": comparison_preset,
            "comparisons": comparisons,
            "output_label": output_label,
        }
    )
    return job


def _expand_single_target_job(
    settings: dict[str, Any],
    raw_job: dict[str, Any],
    comparisons: list[dict[str, Any]],
    comparison_preset: str | None,
) -> list[dict[str, Any]]:
    has_target = "target" in raw_job
    has_targets = "targets" in raw_job
    if has_target == has_targets:
        raise ValueError("Single-target jobs must define exactly one of target/targets")

    targets = _as_string_list(
        raw_job["targets"] if has_targets else raw_job["target"],
        "targets" if has_targets else "target",
    )
    for target in targets:
        if target not in TARGET_REGISTRY:
            raise ValueError(f"Unknown target: {target}")

    if "output_label" in raw_job and len(targets) > 1:
        raise ValueError("'output_label' cannot be used with multi-target jobs")

    return [
        _build_expanded_job(
            settings=settings,
            raw_job=raw_job,
            dataset_mode="single_target",
            target=target,
            target_class=None,
            comparisons=comparisons,
            comparison_preset=comparison_preset,
            expanded_value=target,
            is_batch=len(targets) > 1,
        )
        for target in targets
    ]


def _expand_target_class_job(
    settings: dict[str, Any],
    raw_job: dict[str, Any],
    comparisons: list[dict[str, Any]],
    comparison_preset: str | None,
) -> list[dict[str, Any]]:
    has_target_class = "target_class" in raw_job
    has_target_classes = "target_classes" in raw_job
    if has_target_class == has_target_classes:
        raise ValueError(
            "Target-class jobs must define exactly one of target_class/target_classes"
        )

    target_classes = _as_string_list(
        raw_job["target_classes"] if has_target_classes else raw_job["target_class"],
        "target_classes" if has_target_classes else "target_class",
    )
    for target_class in target_classes:
        if target_class not in TARGET_CLASS_REGISTRY:
            raise ValueError(f"Unknown target_class: {target_class}")

    if "output_label" in raw_job and len(target_classes) > 1:
        raise ValueError("'output_label' cannot be used with multi-target-class jobs")

    return [
        _build_expanded_job(
            settings=settings,
            raw_job=raw_job,
            dataset_mode="target_class",
            target=None,
            target_class=target_class,
            comparisons=comparisons,
            comparison_preset=comparison_preset,
            expanded_value=target_class,
            is_batch=len(target_classes) > 1,
        )
        for target_class in target_classes
    ]


def _expand_job(
    config_path: Path,
    base_settings: dict[str, Any],
    raw_job: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(raw_job, dict):
        raise ValueError("Each job must be a YAML mapping")

    unknown_keys = set(raw_job).difference(JOB_METADATA_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown job config keys: {sorted(unknown_keys)}")

    job_settings = raw_job.get("settings", {})
    if not isinstance(job_settings, dict):
        raise ValueError("Job 'settings' must be a mapping when provided")
    settings = _deep_merge(base_settings, job_settings)
    _validate_settings(settings)

    dataset_mode = raw_job.get("dataset_mode")
    if dataset_mode not in {"single_target", "target_class"}:
        raise ValueError("dataset_mode must be one of: 'single_target', 'target_class'")

    comparisons, comparison_preset = _load_comparisons(config_path, raw_job)

    if dataset_mode == "single_target":
        return _expand_single_target_job(
            settings=settings,
            raw_job=raw_job,
            comparisons=comparisons,
            comparison_preset=comparison_preset,
        )
    return _expand_target_class_job(
        settings=settings,
        raw_job=raw_job,
        comparisons=comparisons,
        comparison_preset=comparison_preset,
    )


def _raw_jobs_from_run_config(run_config: dict[str, Any]) -> list[dict[str, Any]]:
    if "jobs" in run_config:
        jobs = run_config["jobs"]
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("'jobs' must be a non-empty list when provided")
        return jobs

    job_config = {
        key: value
        for key, value in run_config.items()
        if key not in RUN_METADATA_KEYS
    }
    if not job_config:
        raise ValueError("Run config must define 'jobs' or a top-level job")
    return [job_config]


def _validate_duplicate_output_labels(jobs: list[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    duplicates = []
    for job in jobs:
        label_key = (job["train_round_id"], job["output_label"])
        if label_key in seen:
            duplicates.append(job["output_label"])
        seen.add(label_key)

    if duplicates:
        raise ValueError(f"Duplicate output labels in run batch: {sorted(duplicates)}")


def load_run_config(config_path: str | Path = DEFAULT_RUN_CONFIG_PATH) -> dict[str, Any]:
    """Load and expand a learner run config into concrete workflow jobs."""
    resolved_config_path = Path(config_path).expanduser().resolve()
    run_config = _load_yaml_mapping(resolved_config_path)

    unknown_top_level = set(run_config).difference(RUN_METADATA_KEYS | JOB_METADATA_KEYS)
    if "jobs" in run_config and unknown_top_level:
        raise ValueError(f"Unknown run config keys: {sorted(unknown_top_level)}")

    defaults_path = _resolve_defaults_path(resolved_config_path, run_config)
    defaults = _load_yaml_mapping(defaults_path)
    run_settings = run_config.get("settings", {})
    if not isinstance(run_settings, dict):
        raise ValueError("Run 'settings' must be a mapping when provided")
    base_settings = _deep_merge(defaults, run_settings)
    _validate_settings(base_settings)

    jobs = []
    for raw_job in _raw_jobs_from_run_config(run_config):
        jobs.extend(
            _expand_job(
                config_path=resolved_config_path,
                base_settings=base_settings,
                raw_job=raw_job,
            )
        )

    _validate_duplicate_output_labels(jobs)
    return {
        "config_path": str(resolved_config_path),
        "defaults_path": str(defaults_path),
        "jobs": jobs,
    }
