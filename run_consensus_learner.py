"""IDE-friendly entrypoint for FluMolScreen consensus learner evaluation."""

from __future__ import annotations

import flumolscreen.learner_config as imported_config
from flumolscreen.ml.workflow import print_cv_summary, run_cv_workflow

# --- user inputs ---
# Leave an override as None to use the default value from
# flumolscreen/learner_config.py.
# Examples:
# - Single-target furin run:
#     DATASET_MODE_OVERRIDE = "single_target"
#     TARGET_ID_OVERRIDE = "furin"
#     FAMILY_KEY_OVERRIDE = None
# - Pooled PA hierarchical run:
#     DATASET_MODE_OVERRIDE = "target_family"
#     TARGET_ID_OVERRIDE = None
#     FAMILY_KEY_OVERRIDE = "pa"
DATASET_MODE_OVERRIDE = "target_family"  # "single_target" or "target_family"
TARGET_ID_OVERRIDE = None  # e.g. "furin", "pa_ph1n1"
FAMILY_KEY_OVERRIDE = "pa"  # e.g. "pa", "na"


def _load_config() -> dict:
    """Load the canonical learner configuration from flumolscreen/learner_config.py."""
    return {
        name: getattr(imported_config, name)
        for name in imported_config.CONFIG_VARIABLE_NAMES
    }


def _resolve_runtime_settings(config: dict) -> dict:
    """Resolve dataset-selection settings using override > config precedence."""
    resolved_dataset_mode = DATASET_MODE_OVERRIDE or config["DATASET_MODE"]
    resolved_target_id = TARGET_ID_OVERRIDE or config["TARGET_ID"]
    resolved_family_key = FAMILY_KEY_OVERRIDE or config["FAMILY_KEY"]

    if resolved_dataset_mode == "single_target":
        comparisons = config["COMPARISONS_SINGLE_TARGET"]
        if resolved_target_id is None:
            raise ValueError("TARGET_ID is required when DATASET_MODE='single_target'")
        resolved_family_key = None
    elif resolved_dataset_mode == "target_family":
        comparisons = config["COMPARISONS_TARGET_FAMILY"]
        if resolved_family_key is None:
            raise ValueError("FAMILY_KEY is required when DATASET_MODE='target_family'")
        resolved_target_id = None
    else:
        raise ValueError(
            "DATASET_MODE must be one of: 'single_target', 'target_family'"
        )

    return {
        **config,
        "DATASET_MODE": resolved_dataset_mode,
        "TARGET_ID": resolved_target_id,
        "FAMILY_KEY": resolved_family_key,
        "COMPARISONS": comparisons,
    }


def main(config: dict) -> None:
    """Run the learner workflow with the resolved configuration dictionary."""
    resolved_config = _resolve_runtime_settings(config)
    results = run_cv_workflow(
        data_dir=resolved_config["DATA_DIR"],
        results_dir=resolved_config["RESULTS_DIR"],
        train_round_id=resolved_config["TRAIN_ROUND_ID"],
        target_id=resolved_config["TARGET_ID"],
        dataset_mode=resolved_config["DATASET_MODE"],
        family_key=resolved_config["FAMILY_KEY"],
        comparisons=resolved_config["COMPARISONS"],
        model_runs=resolved_config["MODEL_RUNS"],
        outer_split_type=resolved_config["OUTER_SPLIT_TYPE"],
        outer_split_params=resolved_config["OUTER_SPLIT_PARAMS"],
        tuning_mode=resolved_config["TUNING_MODE"],
        tuning_metric=resolved_config["TUNING_METRIC"],
        holdout_validation_fraction=resolved_config["HOLDOUT_VALIDATION_FRACTION"],
        inner_split_type=resolved_config["INNER_SPLIT_TYPE"],
        inner_split_params=resolved_config["INNER_SPLIT_PARAMS"],
        tuning_n_trials=resolved_config["TUNING_N_TRIALS"],
        tuning_random_seed=resolved_config["TUNING_RANDOM_SEED"],
        inference_mode=resolved_config["INFERENCE_MODE"],
        calibration_fraction=resolved_config["CALIBRATION_FRACTION"],
        ensemble_size_m=resolved_config["ENSEMBLE_SIZE_M"],
        interval_coverage=resolved_config["INTERVAL_COVERAGE"],
        inference_random_seed=resolved_config["INFERENCE_RANDOM_SEED"],
    )

    print_cv_summary(
        results=results,
        train_round_id=resolved_config["TRAIN_ROUND_ID"],
        target_id=resolved_config["TARGET_ID"],
        dataset_mode=resolved_config["DATASET_MODE"],
        family_key=resolved_config["FAMILY_KEY"],
        comparisons=resolved_config["COMPARISONS"],
        model_runs=resolved_config["MODEL_RUNS"],
        outer_split_type=resolved_config["OUTER_SPLIT_TYPE"],
        tuning_mode=resolved_config["TUNING_MODE"],
        tuning_metric=resolved_config["TUNING_METRIC"],
        holdout_validation_fraction=resolved_config["HOLDOUT_VALIDATION_FRACTION"],
        inner_split_type=resolved_config["INNER_SPLIT_TYPE"],
        tuning_n_trials=resolved_config["TUNING_N_TRIALS"],
        inference_mode=resolved_config["INFERENCE_MODE"],
        calibration_fraction=resolved_config["CALIBRATION_FRACTION"],
        ensemble_size_m=resolved_config["ENSEMBLE_SIZE_M"],
        interval_coverage=resolved_config["INTERVAL_COVERAGE"],
    )


if __name__ == "__main__":
    main(_load_config())
