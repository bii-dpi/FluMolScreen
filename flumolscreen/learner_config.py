"""Default configuration for the FluMolScreen consensus learner."""

from flumolscreen.feature_registry import FEATURE_REGISTRY

# --- data settings ---
DATA_DIR = "data"
RESULTS_DIR = "results"
TRAIN_ROUND_ID = "round_synthetic"

# --- dataset selection ---
DATASET_MODE = "single_target"  # "single_target" or "target_family"
TARGET_ID = "furin"
FAMILY_KEY = "pa"

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
TUNING_N_TRIALS = 100
TUNING_RANDOM_SEED = 42

# --- feature preprocessing and hit-ranking metrics ---
STANDARDIZE_FEATURES = False
HIT_THRESHOLD_PKD = 5.0
ENRICHMENT_TOP_FRACTIONS = [0.01, 0.05, 0.10]
PRECISION_AT_N_VALUES = [10, 25, 50]

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
        "standardize_features": True,
    },
    {
        "model_type": "xgboost",
        "model_params": {"n_estimators": 300, 'max_depth': 4},
        "standardize_features": False,
    },
]

COMPARISONS_SINGLE_TARGET = [
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

COMPARISONS_TARGET_FAMILY = [
    {
        "name": "6predictor_pr",
        "feature_requests": [
            {"feature_set": "6predictor_pr", "source": "shared"},
        ],
    },
    {
        "name": "6predictor_pr_plus_hxnx_hierarchical_tminus1_pr",
        "feature_requests": [
            {"feature_set": "6predictor_pr", "source": "shared"},
            {
                "feature_set": "hxnx_hierarchical_tminus1_pr",
                "source": "shared",
                "feature_generator": "hierarchical_perstrain",
                "base_feature_set": "6predictor_pr",
                "feature_columns": FEATURE_REGISTRY["6predictor_pr"]["default_columns"],
            },
        ],
    },
    {
        "name": "6predictor_pr_plus_derived_plus_hxnx_hierarchical_tminus1_pr",
        "feature_requests": [
            {"feature_set": "6predictor_pr", "source": "shared"},
            {"feature_set": "6predictor_pr_derived", "source": "shared"},
            {
                "feature_set": "hxnx_hierarchical_tminus1_pr",
                "source": "shared",
                "feature_generator": "hierarchical_perstrain",
                "base_feature_set": "6predictor_pr",
                "feature_columns": FEATURE_REGISTRY["6predictor_pr"]["default_columns"],
            },
        ],
    },
    {
        "name": "6predictor_pr_plus_derived_plus_chemdescriptors_plus_hxnx_hierarchical_tminus1_pr",
        "feature_requests": [
            {"feature_set": "6predictor_pr", "source": "shared"},
            {"feature_set": "6predictor_pr_derived", "source": "shared"},
            {"feature_set": "chemdescriptors", "source": "shared"},
            {
                "feature_set": "hxnx_hierarchical_tminus1_pr",
                "source": "shared",
                "feature_generator": "hierarchical_perstrain",
                "base_feature_set": "6predictor_pr",
                "feature_columns": FEATURE_REGISTRY["6predictor_pr"]["default_columns"],
            },
        ],
    },
]

CONFIG_VARIABLE_NAMES = (
    "DATA_DIR",
    "RESULTS_DIR",
    "TRAIN_ROUND_ID",
    "DATASET_MODE",
    "TARGET_ID",
    "FAMILY_KEY",
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
    "STANDARDIZE_FEATURES",
    "HIT_THRESHOLD_PKD",
    "ENRICHMENT_TOP_FRACTIONS",
    "PRECISION_AT_N_VALUES",
    "MODEL_RUNS",
    "COMPARISONS_SINGLE_TARGET",
    "COMPARISONS_TARGET_FAMILY",
)
