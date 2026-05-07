"""Default configuration for the FluMolScreen consensus learner."""

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
    {
        "model_type": "xgboost",
        "model_params": {"n_estimators": 300, 'max_depth': 4},
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

CONFIG_VARIABLE_NAMES = (
    "DATA_DIR",
    "RESULTS_DIR",
    "TRAIN_ROUND_ID",
    "TARGET_ID",
    "OUTER_SPLIT_TYPE",
    "INNER_SPLIT_TYPE",
    "OUTER_K",
    "INNER_L",
    "OUTER_SPLIT_PARAMS",
    "INNER_SPLIT_PARAMS",
    "HOLDOUT_VALIDATION_FRACTION",
    "CALIBRATION_FRACTION",
    "ENSEMBLE_SIZE_M",
    "INTERVAL_COVERAGE",
    "INFERENCE_MODE",
    "INFERENCE_RANDOM_SEED",
    "TUNING_MODE",
    "TUNING_METRIC",
    "TUNING_N_TRIALS",
    "TUNING_RANDOM_SEED",
    "MODEL_RUNS",
    "COMPARISONS",
)
