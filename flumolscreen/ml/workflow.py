"""General cross-validation workflow for FluMolScreen ML experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flumolscreen.ml.evaluation import (
    compute_regression_metrics,
    fit_regression_model,
    predict_regression_model,
)
from flumolscreen.ml.inference import fit_final_candidate_and_save_inference
from flumolscreen.ml.load_dataset import compose_candidate_datasets
from flumolscreen.ml.splits import make_splits
from flumolscreen.ml.tuning import TRIAL_TRACE_COLUMNS, tune_candidate
from flumolscreen.ml.utils import (
    DISPLAY_TUNING_MODES,
    build_merged_inference_path,
    evaluation_base_name,
    merge_inference_predictions,
    prepare_result_dirs,
    round_metrics,
)

METRIC_COLUMNS = ["rmse", "mae", "r2", "spearman"]

__all__ = ["print_cv_summary", "run_cv_workflow"]


def _build_fold_record(
    candidate: dict,
    outer_fold_idx: int,
    tuning_metric: str,
    tuning_score: float | None,
    tuning_mode: str,
    n_train: int,
    n_test: int,
    metrics: dict,
) -> pd.DataFrame:
    """Return one outer-fold result row for a tuned candidate."""
    # Store fold identity, candidate identity, and outer-test metrics together.
    return pd.DataFrame(
        [
            {
                "outer_fold": outer_fold_idx,
                "comparison_name": candidate["comparison_name"],
                "model_type": candidate["model_type"],
                "p": candidate["p"],
                "tuning_mode": tuning_mode,
                "tuning_metric": tuning_metric,
                "tuning_score": tuning_score,
                "n_train": n_train,
                "n_test": n_test,
                **metrics,
            }
        ]
    )


def _summarize_selected_params_by_outer_fold(tuning_df: pd.DataFrame) -> pd.DataFrame:
    """Return one per-candidate row with selected params listed by outer fold."""
    # Keep only outer-fold tuning rows, then serialize the chosen params compactly.
    key_columns = [
        "comparison_name",
        "model_type",
        "p",
        "tuning_mode",
        "tuning_metric",
    ]
    outer_tuning_df = tuning_df[tuning_df["outer_fold"].notna()].copy()
    if outer_tuning_df.empty:
        return pd.DataFrame(columns=[*key_columns, "selected_model_params_by_outer_fold"])

    outer_tuning_df["outer_fold"] = outer_tuning_df["outer_fold"].astype(int)
    outer_tuning_df["selected_model_params_by_outer_fold"] = (
        "fold_"
        + outer_tuning_df["outer_fold"].astype(str)
        + "="
        + outer_tuning_df["model_params"]
    )
    return (
        outer_tuning_df.sort_values("outer_fold")
        .groupby(key_columns, as_index=False)["selected_model_params_by_outer_fold"]
        .agg(" | ".join)
    )


def _initialize_trace_file(trace_path: Path) -> None:
    """Create an empty trial-trace CSV with the expected columns."""
    # Start each workflow run with a clean live trace file for Optuna trials.
    pd.DataFrame(columns=TRIAL_TRACE_COLUMNS).to_csv(trace_path, index=False)


def _build_merged_inference_table(
    inference_paths: dict[tuple[str, str], Path],
    inference_dir: Path,
    target_id: str,
) -> Path:
    """Load per-candidate inference files, merge them, and save one wide table."""
    inference_tables = {
        f"{comparison_name}_{model_type}": pd.read_csv(path)
        for (comparison_name, model_type), path in inference_paths.items()
    }
    merged_df = merge_inference_predictions(inference_tables)
    merged_path = build_merged_inference_path(
        inference_dir=inference_dir,
        target_id=target_id,
    )
    merged_df.to_csv(merged_path, index=False)
    return merged_path


def evaluate_candidate_on_outer_fold(
    candidate: dict,
    outer_fold_idx: int,
    outer_train_idx,
    outer_test_idx,
    trace_path: Path,
    tuning_mode: str | None,
    tuning_metric: str,
    holdout_validation_fraction: float = 0.2,
    inner_split_type: str | None = None,
    inner_split_params: dict | None = None,
    tuning_n_trials: int = 20,
    tuning_random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune one candidate on the outer-train rows, then score the outer-test rows."""
    training_df = candidate["training_df"]

    # Materialize the outer train/test partition for this candidate.
    train_df = training_df.iloc[outer_train_idx].reset_index(drop=True)
    test_df = training_df.iloc[outer_test_idx].reset_index(drop=True)

    # Tune only this candidate's model parameters on the outer-train split.
    model_params, tuning_score, tuning_df = tune_candidate(
        training_df=train_df,
        candidate=candidate,
        tuning_mode=tuning_mode,
        tuning_metric=tuning_metric,
        holdout_validation_fraction=holdout_validation_fraction,
        inner_split_type=inner_split_type,
        inner_split_params=inner_split_params,
        outer_fold_idx=outer_fold_idx,
        n_trials=tuning_n_trials,
        random_seed=tuning_random_seed,
        selection_scope=f"outer_fold_{outer_fold_idx}",
        trace_path=trace_path,
    )

    # Fit the tuned candidate on outer-train and evaluate on outer-test.
    model, feature_columns = fit_regression_model(
        training_df=train_df,
        model_type=candidate["model_type"],
        model_params=model_params,
    )
    predictions = predict_regression_model(
        model=model,
        df=test_df,
        feature_columns=feature_columns,
    )
    metrics = compute_regression_metrics(test_df["label_pkd"], predictions)
    # Save the tuned-candidate outer-fold metrics in one tidy row.
    fold_df = _build_fold_record(
        candidate=candidate,
        outer_fold_idx=outer_fold_idx,
        tuning_metric=tuning_metric,
        tuning_score=None if tuning_score is None else round(tuning_score, 4),
        tuning_mode=tuning_df.iloc[0]["tuning_mode"],
        n_train=len(train_df),
        n_test=len(test_df),
        metrics={name: round(float(value), 4) for name, value in metrics.items()},
    )
    return fold_df, tuning_df


def build_cv_summary(fold_df: pd.DataFrame, tuning_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level metrics to one summary row per candidate."""
    # Average fold metrics per candidate so ablations are easy to compare.
    summary_df = (
        fold_df.groupby(
            ["comparison_name", "model_type", "p", "tuning_mode", "tuning_metric"],
            as_index=False,
        )[[*METRIC_COLUMNS, "n_train", "n_test"]]
        .mean()
    )
    # Attach the per-fold selected hyperparameters to each candidate summary row.
    params_by_fold_df = _summarize_selected_params_by_outer_fold(tuning_df)
    summary_df = summary_df.merge(
        params_by_fold_df,
        on=["comparison_name", "model_type", "p", "tuning_mode", "tuning_metric"],
        how="left",
    )
    summary_df.insert(0, "summary", "mean_across_outer_folds")
    return round_metrics(summary_df)


def save_cv_outputs(
    evaluation_dir,
    target_id: str,
    split_type: str,
    tuning_mode: str | None,
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trace_path: Path,
) -> tuple[Path, Path, Path]:
    """Save the fold-level, summary, and live tuning trace tables."""
    # Reuse one naming stem across all evaluation outputs for this run.
    base_name = evaluation_base_name(target_id, split_type, tuning_mode)

    fold_path = evaluation_dir / f"{base_name}_fold_metrics.csv"
    summary_path = evaluation_dir / f"{base_name}_summary.csv"
    resolved_trace_path = evaluation_dir / f"{base_name}_trace.csv"
    if trace_path != resolved_trace_path:
        trace_path.replace(resolved_trace_path)
        trace_path = resolved_trace_path

    round_metrics(fold_df).to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return fold_path, summary_path, trace_path


def run_cv_workflow(
    data_dir: str,
    results_dir: str,
    train_round_id: str,
    target_id: str,
    comparisons: list[dict],
    model_runs: list[dict],
    outer_split_type: str,
    outer_split_params: dict | None,
    tuning_mode: str | None,
    tuning_metric: str,
    holdout_validation_fraction: float = 0.2,
    inner_split_type: str | None = None,
    inner_split_params: dict | None = None,
    tuning_n_trials: int = 20,
    tuning_random_seed: int = 42,
    inference_mode: str = "point",
    calibration_fraction: float = 0.2,
    ensemble_size_m: int = 10,
    interval_coverage: float = 0.9,
    inference_random_seed: int = 42,
) -> dict:
    """Run the general outer-loop scaffold across all candidate ablations."""
    result_dirs = prepare_result_dirs(results_dir, train_round_id)
    trace_path = result_dirs["evaluation"] / "__trial_trace_in_progress__.csv"
    _initialize_trace_file(trace_path)

    # Flatten the comparison/model grid into one candidate list.
    candidates = compose_candidate_datasets(
        data_dir=data_dir,
        round_id=train_round_id,
        target_id=target_id,
        comparisons=comparisons,
        model_runs=model_runs,
    )

    # Build the shared outer splits from the first candidate's training table.
    outer_splits = make_splits(
        df=candidates[0]["training_df"],
        split_type=outer_split_type,
        split_params=outer_split_params,
    )

    fold_rows = []
    tuning_rows = []

    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        # Evaluate every candidate on this outer fold for full ablation metrics.
        for candidate in candidates:
            fold_df, tuning_df = evaluate_candidate_on_outer_fold(
                candidate=candidate,
                outer_fold_idx=outer_fold_idx,
                outer_train_idx=outer_train_idx,
                outer_test_idx=outer_test_idx,
                trace_path=trace_path,
                tuning_mode=tuning_mode,
                tuning_metric=tuning_metric,
                holdout_validation_fraction=holdout_validation_fraction,
                inner_split_type=inner_split_type,
                inner_split_params=inner_split_params,
                tuning_n_trials=tuning_n_trials,
                tuning_random_seed=tuning_random_seed,
            )
            fold_rows.append(fold_df)
            tuning_rows.append(tuning_df)

    fold_df = pd.concat(fold_rows, ignore_index=True)
    tuning_df = pd.concat(tuning_rows, ignore_index=True)
    summary_df = build_cv_summary(fold_df, tuning_df)

    # Tune on all labeled data and write one inference file per candidate.
    final_tuning_rows = []
    inference_paths = {}
    for candidate in candidates:
        model_params, _, candidate_tuning_df = tune_candidate(
            training_df=candidate["training_df"],
            candidate=candidate,
            tuning_mode=tuning_mode,
            tuning_metric=tuning_metric,
            holdout_validation_fraction=holdout_validation_fraction,
            inner_split_type=inner_split_type,
            inner_split_params=inner_split_params,
            outer_fold_idx=None,
            n_trials=tuning_n_trials,
            random_seed=tuning_random_seed,
            selection_scope="full_labeled_data",
            trace_path=trace_path,
        )
        candidate_tuning_df.insert(0, "selection_scope", "full_labeled_data")
        final_tuning_rows.append(candidate_tuning_df)
        inference_paths[(candidate["comparison_name"], candidate["model_type"])] = (
            fit_final_candidate_and_save_inference(
                inference_dir=result_dirs["inference"],
                target_id=target_id,
                candidate=candidate,
                tuned_model_params=model_params,
                inference_mode=inference_mode,
                calibration_fraction=calibration_fraction,
                n_bootstrap=ensemble_size_m,
                interval_coverage=interval_coverage,
                random_state=inference_random_seed,
            )
        )

    final_tuning_df = pd.concat(final_tuning_rows, ignore_index=True)
    merged_inference_path = _build_merged_inference_table(
        inference_paths=inference_paths,
        inference_dir=result_dirs["inference"],
        target_id=target_id,
    )
    fold_path, summary_path, trace_path = save_cv_outputs(
        evaluation_dir=result_dirs["evaluation"],
        target_id=target_id,
        split_type=outer_split_type,
        tuning_mode=tuning_mode,
        fold_df=fold_df,
        summary_df=summary_df,
        trace_path=trace_path,
    )

    return {
        "candidates": candidates,
        "fold_df": fold_df,
        "summary_df": summary_df,
        "tuning_df": tuning_df,
        "final_tuning_df": final_tuning_df,
        "fold_path": fold_path,
        "summary_path": summary_path,
        "trace_path": trace_path,
        "inference_paths": inference_paths,
        "merged_inference_path": merged_inference_path,
    }


def print_cv_summary(
    results: dict,
    train_round_id: str,
    target_id: str,
    comparisons: list[dict],
    model_runs: list[dict],
    outer_split_type: str,
    tuning_mode: str | None,
    tuning_metric: str,
    holdout_validation_fraction: float = 0.2,
    inner_split_type: str | None = None,
    tuning_n_trials: int = 20,
    inference_mode: str = "point",
    calibration_fraction: float = 0.2,
    ensemble_size_m: int = 10,
    interval_coverage: float = 0.9,
) -> None:
    """Print a compact summary of the CV run for IDE execution."""
    display_tuning_mode = DISPLAY_TUNING_MODES.get(tuning_mode, str(tuning_mode))

    # Report the run configuration first so saved outputs are interpretable.
    print("CV settings:")
    print(f"- train_round_id: {train_round_id}")
    print(f"- target_id: {target_id}")
    print(f"- outer_split_type: {outer_split_type}")
    print(f"- tuning_mode: {display_tuning_mode}")
    print(f"- tuning_metric: {tuning_metric}")
    print(f"- inference_mode: {inference_mode}")
    if tuning_mode is not None:
        print(f"- tuning_n_trials: {tuning_n_trials}")
    if tuning_mode == "holdout":
        print(f"- holdout_validation_fraction: {holdout_validation_fraction}")
    if inner_split_type is not None:
        print(f"- inner_split_type: {inner_split_type}")
    if inference_mode == "adaptive_conformal":
        print(f"- calibration_fraction: {calibration_fraction}")
        print(f"- ensemble_size_m: {ensemble_size_m}")
        print(f"- interval_coverage: {interval_coverage}")
    print()

    # Show the full ablation grid being evaluated.
    print("Comparisons:")
    for comparison in comparisons:
        print(f"- {comparison['name']}")
    print()
    print("Model runs:")
    for model_run in model_runs:
        print(f"- {model_run['model_type']}: {model_run.get('model_params', {})}")
    print()

    # Surface the main saved tables and per-candidate inference files.
    print("Saved outputs:")
    print(f"- fold_metrics: {results['fold_path']}")
    print(f"- summary: {results['summary_path']}")
    print(f"- trace: {results['trace_path']}")
    print(f"- merged_inference: {results['merged_inference_path']}")
    for (comparison_name, model_type), path in results["inference_paths"].items():
        print(f"- inference [{comparison_name}, {model_type}]: {path}")
