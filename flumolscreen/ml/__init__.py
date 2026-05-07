"""Machine-learning utilities for FluMolScreen."""

from flumolscreen.ml.conformal import (
    CONFORMAL_EPSILON,
    apply_symmetric_conformal_interval,
    compute_symmetric_conformal_half_width,
    compute_absolute_standardized_residuals,
    fit_symmetric_conformal_scaler,
)
from flumolscreen.ml.evaluation import (
    compute_regression_metrics,
    fit_regression_model,
    predict_regression_model,
    run_split_evaluation,
)
from flumolscreen.ml.inference import (
    fit_bootstrap_ensemble,
    fit_final_candidate_and_save_inference,
    predict_bootstrap_ensemble,
)
from flumolscreen.ml.load_dataset import (
    compose_candidate_datasets,
    infer_feature_generation_settings,
)
from flumolscreen.ml.model_registry import (
    get_tuning_space,
    suggest_model_params,
)
from flumolscreen.ml.splits import (
    make_bootstrap_sample_indices,
    make_group_kfold_splits,
    make_holdout_splits_by_column,
    make_random_holdout_split,
    make_random_kfold_splits,
    make_splits,
)
from flumolscreen.ml.tuning import (
    build_tuning_record,
    score_candidate_params,
    tune_candidate,
    tune_candidate_holdout,
    tune_candidate_nested_cv,
    tune_candidate_no_tuning,
)
from flumolscreen.ml.utils import (
    DEFAULT_MODEL_PARAMS,
    NON_FEATURE_COLUMNS,
    evaluation_base_name,
    inference_file_name,
    make_regression_model,
    prepare_result_dirs,
    round_metrics,
    select_model_feature_columns,
)
from flumolscreen.ml.workflow import (
    print_cv_summary,
    run_cv_workflow,
)

__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "NON_FEATURE_COLUMNS",
    "CONFORMAL_EPSILON",
    "apply_symmetric_conformal_interval",
    "compute_symmetric_conformal_half_width",
    "compose_candidate_datasets",
    "compute_regression_metrics",
    "compute_absolute_standardized_residuals",
    "build_tuning_record",
    "evaluation_base_name",
    "fit_bootstrap_ensemble",
    "fit_final_candidate_and_save_inference",
    "fit_symmetric_conformal_scaler",
    "fit_regression_model",
    "get_tuning_space",
    "infer_feature_generation_settings",
    "inference_file_name",
    "make_bootstrap_sample_indices",
    "make_group_kfold_splits",
    "make_holdout_splits_by_column",
    "make_random_holdout_split",
    "make_random_kfold_splits",
    "make_regression_model",
    "make_splits",
    "predict_regression_model",
    "predict_bootstrap_ensemble",
    "print_cv_summary",
    "prepare_result_dirs",
    "round_metrics",
    "run_cv_workflow",
    "run_split_evaluation",
    "score_candidate_params",
    "select_model_feature_columns",
    "suggest_model_params",
    "tune_candidate",
    "tune_candidate_holdout",
    "tune_candidate_nested_cv",
    "tune_candidate_no_tuning",
]
