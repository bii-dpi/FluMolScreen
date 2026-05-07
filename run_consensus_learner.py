"""IDE-friendly entrypoint for FluMolScreen consensus learner evaluation."""

from __future__ import annotations

import flumolscreen.learner_config as imported_config
from flumolscreen.ml.workflow import print_cv_summary, run_cv_workflow

# Set to True to load the config block from flumolscreen/learner_config.py instead.
USE_IMPORTED_CONFIG = False

# --- data settings ---
DATA_DIR = "data"
RESULTS_DIR = "results"
TRAIN_ROUND_ID = "round_synthetic"
TARGET_ID = "furin"

# --- split settings ---
OUTER_SPLIT_TYPE = "random_kfold"
INNER_SPLIT_TYPE = "random_kfold"
OUTER_K = 5
INNER_L = 3

OUTER_SPLIT_PARAMS = {
    "n_splits": OUTER_K,
    "shuffle": True,
    "random_state": 42,
}
INNER_SPLIT_PARAMS = {
    "n_splits": INNER_L,
    "shuffle": True,
    "random_state": 43,
}

# --- hyperparameter tuning settings ---
HOLDOUT_VALIDATION_FRACTION = 0.2
TUNING_MODE = "nested" # "holdout" # None
TUNING_METRIC = "spearman"
TUNING_N_TRIALS = 20
TUNING_RANDOM_SEED = 42

# --- uncertainty estimation ---
CALIBRATION_FRACTION = 0.2
ENSEMBLE_SIZE_M = 10
INTERVAL_COVERAGE = 0.90
INFERENCE_MODE = "adaptive_conformal"  # "point"
INFERENCE_RANDOM_SEED = 42

# --- modeling settings ---
MODEL_RUNS = [
    {
        "model_type": "ridge",
        "model_params": {"alpha": 1.0},
    },
]

COMPARISONS = [
    {
        "name": "6predictor_pr",
        "feature_requests": [
            {"feature_set": "6predictor_pr", "source": "shared"},
        ],
    },
    {
        "name": "6predictor_pr_plus_derived",
        "feature_requests": [
            {"feature_set": "6predictor_pr", "source": "shared"},
            {"feature_set": "6predictor_pr_derived", "source": "shared"},
        ],
    },
    {
        "name": "6predictor_pr_derived_only",
        "feature_requests": [
            {"feature_set": "6predictor_pr_derived", "source": "shared"},
        ],
    },
    {
        "name": "chemdescriptors_only",
        "feature_requests": [
            {"feature_set": "chemdescriptors", "source": "shared"},
        ],
    },
    {
        "name": "6predictor_pr_plus_chemdescriptors",
        "feature_requests": [
            {"feature_set": "6predictor_pr", "source": "shared"},
            {"feature_set": "chemdescriptors", "source": "shared"},
        ],
    },
    {
        "name": "6predictor_pr_plus_derived_plus_chemdescriptors",
        "feature_requests": [
            {"feature_set": "6predictor_pr", "source": "shared"},
            {"feature_set": "6predictor_pr_derived", "source": "shared"},
            {"feature_set": "chemdescriptors", "source": "shared"},
        ],
    },
]


def _load_config() -> dict:
    """Return either the local config block or the imported config module values."""
    # Load the shared config module in bulk when the toggle is enabled.
    if USE_IMPORTED_CONFIG:
        return {
            name: getattr(imported_config, name)
            for name in imported_config.CONFIG_VARIABLE_NAMES
        }

    # Otherwise use the directly editable variables defined in this script.
    return {
        name: globals()[name]
        for name in imported_config.CONFIG_VARIABLE_NAMES
    }


def main(config: dict) -> None:
    """Run the learner workflow with the resolved configuration dictionary."""
    # Execute the shared CV workflow with the chosen runtime configuration.
    results = run_cv_workflow(
        data_dir=config["DATA_DIR"],
        results_dir=config["RESULTS_DIR"],
        train_round_id=config["TRAIN_ROUND_ID"],
        target_id=config["TARGET_ID"],
        comparisons=config["COMPARISONS"],
        model_runs=config["MODEL_RUNS"],
        outer_split_type=config["OUTER_SPLIT_TYPE"],
        outer_split_params=config["OUTER_SPLIT_PARAMS"],
        tuning_mode=config["TUNING_MODE"],
        tuning_metric=config["TUNING_METRIC"],
        holdout_validation_fraction=config["HOLDOUT_VALIDATION_FRACTION"],
        inner_split_type=config["INNER_SPLIT_TYPE"],
        inner_split_params=config["INNER_SPLIT_PARAMS"],
        tuning_n_trials=config["TUNING_N_TRIALS"],
        tuning_random_seed=config["TUNING_RANDOM_SEED"],
        inference_mode=config["INFERENCE_MODE"],
        calibration_fraction=config["CALIBRATION_FRACTION"],
        ensemble_size_m=config["ENSEMBLE_SIZE_M"],
        interval_coverage=config["INTERVAL_COVERAGE"],
        inference_random_seed=config["INFERENCE_RANDOM_SEED"],
    )

    # Print the saved output locations and the resolved run settings.
    print_cv_summary(
        results=results,
        train_round_id=config["TRAIN_ROUND_ID"],
        target_id=config["TARGET_ID"],
        comparisons=config["COMPARISONS"],
        model_runs=config["MODEL_RUNS"],
        outer_split_type=config["OUTER_SPLIT_TYPE"],
        tuning_mode=config["TUNING_MODE"],
        tuning_metric=config["TUNING_METRIC"],
        holdout_validation_fraction=config["HOLDOUT_VALIDATION_FRACTION"],
        inner_split_type=config["INNER_SPLIT_TYPE"],
        tuning_n_trials=config["TUNING_N_TRIALS"],
        inference_mode=config["INFERENCE_MODE"],
        calibration_fraction=config["CALIBRATION_FRACTION"],
        ensemble_size_m=config["ENSEMBLE_SIZE_M"],
        interval_coverage=config["INTERVAL_COVERAGE"],
    )


if __name__ == "__main__":
    main(_load_config())
