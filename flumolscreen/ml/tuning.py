"""Candidate-level tuning strategies for FluMolScreen ML workflows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from flumolscreen.ml.evaluation import run_split_evaluation
from flumolscreen.ml.model_registry import get_tuning_space, suggest_model_params
from flumolscreen.ml.splits import make_splits
from flumolscreen.ml.utils import format_model_params

MAXIMIZE_METRICS = {"spearman", "r2"}
TRIAL_TRACE_COLUMNS = [
    "selection_scope",
    "outer_fold",
    "comparison_name",
    "model_type",
    "p",
    "tuning_mode",
    "tuning_metric",
    "trial_number",
    "n_trials_planned",
    "tuning_score",
    "best_tuning_score_so_far",
    "model_params",
]

__all__ = [
    "TRIAL_TRACE_COLUMNS",
    "build_tuning_record",
    "score_candidate_params",
    "tune_candidate",
    "tune_candidate_holdout",
    "tune_candidate_nested_cv",
    "tune_candidate_no_tuning",
]


def build_tuning_record(
    candidate: dict,
    outer_fold_idx: int | None,
    tuning_mode: str,
    tuning_metric: str,
    model_params: dict | None,
    tuning_score: float | None,
    n_trials: int,
    used_tuning: bool,
) -> pd.DataFrame:
    """Return one tidy tuning record for downstream aggregation."""
    # Package candidate identity, chosen params, and search metadata in one row.
    return pd.DataFrame(
        [
            {
                "outer_fold": outer_fold_idx,
                "comparison_name": candidate["comparison_name"],
                "model_type": candidate["model_type"],
                "p": candidate["p"],
                "tuning_mode": tuning_mode,
                "tuning_metric": tuning_metric,
                "tuning_score": None if tuning_score is None else round(tuning_score, 4),
                "n_trials": int(n_trials),
                "used_tuning": bool(used_tuning),
                "model_params": format_model_params(model_params),
            }
        ]
    )


def score_candidate_params(
    training_df: pd.DataFrame,
    candidate: dict,
    model_params: dict | None,
    tuning_metric: str,
    splits: list[tuple],
    hit_threshold_pkd: float | None = None,
    enrichment_top_fractions: list[float] | None = None,
    precision_at_n_values: list[int] | None = None,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Score one candidate/parameter setting on the provided inner splits."""
    # Run the shared fold evaluator on the requested inner splits.
    fold_df, summary_df = run_split_evaluation(
        training_df=training_df,
        splits=splits,
        model_type=candidate["model_type"],
        model_params=model_params,
        standardize_features=candidate.get("standardize_features", False),
        hit_threshold_pkd=hit_threshold_pkd,
        enrichment_top_fractions=enrichment_top_fractions,
        precision_at_n_values=precision_at_n_values,
    )
    # Extract the optimization metric from the summary row.
    return float(summary_df.iloc[0][tuning_metric]), fold_df, summary_df


def _resolve_direction(tuning_metric: str) -> str:
    """Return whether the tuning metric should be maximized or minimized."""
    if (
        tuning_metric in MAXIMIZE_METRICS
        or tuning_metric.startswith("ef_")
        or tuning_metric.startswith("precision_at_")
    ):
        return "maximize"
    return "minimize"


def _merge_model_params(candidate: dict, tuned_params: dict[str, Any] | None) -> dict[str, Any]:
    """Layer tuned parameters over the candidate's base model params."""
    return {**(candidate.get("base_model_params") or {}), **(tuned_params or {})}


def _build_tuning_record_for_splits(
    training_df: pd.DataFrame,
    candidate: dict,
    tuning_mode: str,
    tuning_metric: str,
    splits: list[tuple],
    n_trials: int,
    random_seed: int,
    selection_scope: str,
    outer_fold_idx: int | None,
    trace_path: str | None,
    hit_threshold_pkd: float | None,
    enrichment_top_fractions: list[float] | None,
    precision_at_n_values: list[int] | None,
) -> tuple[dict[str, Any], float, pd.DataFrame]:
    """Optimize one candidate on precomputed inner splits and return its record."""
    # Score candidate trials on the supplied inner splits, then package the result.
    model_params, tuning_score, n_trials_run, used_tuning = _optimize_candidate(
        training_df=training_df,
        candidate=candidate,
        tuning_mode=tuning_mode,
        tuning_metric=tuning_metric,
        splits=splits,
        n_trials=n_trials,
        random_seed=random_seed,
        selection_scope=selection_scope,
        outer_fold_idx=outer_fold_idx,
        trace_path=trace_path,
        hit_threshold_pkd=hit_threshold_pkd,
        enrichment_top_fractions=enrichment_top_fractions,
        precision_at_n_values=precision_at_n_values,
    )
    record = build_tuning_record(
        candidate=candidate,
        outer_fold_idx=outer_fold_idx,
        tuning_mode=tuning_mode,
        tuning_metric=tuning_metric,
        model_params=model_params,
        tuning_score=tuning_score,
        n_trials=n_trials_run,
        used_tuning=used_tuning,
    )
    return model_params, tuning_score, record


def _optimize_candidate(
    training_df: pd.DataFrame,
    candidate: dict,
    tuning_mode: str,
    tuning_metric: str,
    splits: list[tuple],
    n_trials: int,
    random_seed: int,
    selection_scope: str,
    outer_fold_idx: int | None,
    trace_path: str | None,
    hit_threshold_pkd: float | None,
    enrichment_top_fractions: list[float] | None,
    precision_at_n_values: list[int] | None,
) -> tuple[dict[str, Any], float, int, bool]:
    """Run an Optuna study over the candidate's registered search space."""
    try:
        import optuna
    except Exception as exc:
        raise ImportError("Optuna is required for hyperparameter tuning.") from exc
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tuning_space = get_tuning_space(candidate["model_type"])
    base_model_params = candidate.get("base_model_params") or {}
    if not tuning_space:
        score, _, _ = score_candidate_params(
            training_df=training_df,
            candidate=candidate,
            model_params=base_model_params,
            tuning_metric=tuning_metric,
            splits=splits,
            hit_threshold_pkd=hit_threshold_pkd,
            enrichment_top_fractions=enrichment_top_fractions,
            precision_at_n_values=precision_at_n_values,
        )
        return base_model_params, score, 0, False

    print(
        "[tuning-start] "
        f"scope={selection_scope} | comparison={candidate['comparison_name']} | "
        f"model={candidate['model_type']} | mode={tuning_mode} | "
        f"n_trials={max(int(n_trials), 1)}",
        flush=True,
    )

    def _append_trace_row(study: Any, trial: Any) -> None:
        """Append one completed trial to the on-disk tuning trace."""
        if trace_path is None:
            return
        pd.DataFrame(
            [
                {
                    "selection_scope": selection_scope,
                    "outer_fold": outer_fold_idx,
                    "comparison_name": candidate["comparison_name"],
                    "model_type": candidate["model_type"],
                    "p": candidate["p"],
                    "tuning_mode": tuning_mode,
                    "tuning_metric": tuning_metric,
                    "trial_number": int(trial.number),
                    "n_trials_planned": int(max(int(n_trials), 1)),
                    "tuning_score": round(float(trial.value), 4),
                    "best_tuning_score_so_far": round(float(study.best_value), 4),
                    "model_params": format_model_params(
                        trial.user_attrs.get("model_params", {})
                    ),
                }
            ]
        ).to_csv(trace_path, mode="a", header=False, index=False)

    # Optimize only within this candidate's search space on the provided inner splits.
    def objective(trial: Any) -> float:
        trial_params = suggest_model_params(trial, candidate["model_type"])
        model_params = _merge_model_params(candidate, trial_params)
        score, _, _ = score_candidate_params(
            training_df=training_df,
            candidate=candidate,
            model_params=model_params,
            tuning_metric=tuning_metric,
            splits=splits,
            hit_threshold_pkd=hit_threshold_pkd,
            enrichment_top_fractions=enrichment_top_fractions,
            precision_at_n_values=precision_at_n_values,
        )
        trial.set_user_attr("model_params", model_params)
        return score

    study = optuna.create_study(
        direction=_resolve_direction(tuning_metric),
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(
        objective,
        n_trials=max(int(n_trials), 1),
        show_progress_bar=False,
        callbacks=[_append_trace_row],
    )

    best_params = dict(study.best_trial.user_attrs["model_params"])
    print(
        "[tuning-done] "
        f"scope={selection_scope} | comparison={candidate['comparison_name']} | "
        f"model={candidate['model_type']} | best_{tuning_metric}={float(study.best_value):.4f} | "
        f"trials={len(study.trials)} | params={format_model_params(best_params)}",
        flush=True,
    )
    return best_params, float(study.best_value), len(study.trials), True


def tune_candidate_no_tuning(
    training_df: pd.DataFrame,
    candidate: dict,
    tuning_metric: str,
    outer_fold_idx: int | None = None,
) -> tuple[dict | None, float | None, pd.DataFrame]:
    """Use the candidate's fixed/default params without an inner tuning pass."""
    del training_df
    model_params = candidate.get("base_model_params")
    record = build_tuning_record(
        candidate=candidate,
        outer_fold_idx=outer_fold_idx,
        tuning_mode="no_tuning",
        tuning_metric=tuning_metric,
        model_params=model_params,
        tuning_score=None,
        n_trials=0,
        used_tuning=False,
    )
    return model_params, None, record


def tune_candidate_holdout(
    training_df: pd.DataFrame,
    candidate: dict,
    tuning_metric: str,
    validation_fraction: float,
    n_trials: int,
    random_seed: int,
    selection_scope: str,
    outer_fold_idx: int | None = None,
    trace_path: str | None = None,
    hit_threshold_pkd: float | None = None,
    enrichment_top_fractions: list[float] | None = None,
    precision_at_n_values: list[int] | None = None,
) -> tuple[dict | None, float, pd.DataFrame]:
    """Tune a candidate on one random inner train/validation split."""
    # Build one inner holdout split for scoring candidate hyperparameters.
    splits = make_splits(
        df=training_df,
        split_type="random_holdout",
        split_params={
            "validation_fraction": validation_fraction,
            "random_state": random_seed,
        },
    )
    return _build_tuning_record_for_splits(
        training_df=training_df,
        candidate=candidate,
        tuning_mode="holdout",
        tuning_metric=tuning_metric,
        splits=splits,
        n_trials=n_trials,
        random_seed=random_seed,
        selection_scope=selection_scope,
        outer_fold_idx=outer_fold_idx,
        trace_path=trace_path,
        hit_threshold_pkd=hit_threshold_pkd,
        enrichment_top_fractions=enrichment_top_fractions,
        precision_at_n_values=precision_at_n_values,
    )


def tune_candidate_nested_cv(
    training_df: pd.DataFrame,
    candidate: dict,
    tuning_metric: str,
    inner_split_type: str,
    inner_split_params: dict,
    n_trials: int,
    random_seed: int,
    selection_scope: str,
    outer_fold_idx: int | None = None,
    trace_path: str | None = None,
    hit_threshold_pkd: float | None = None,
    enrichment_top_fractions: list[float] | None = None,
    precision_at_n_values: list[int] | None = None,
) -> tuple[dict | None, float, pd.DataFrame]:
    """Tune a candidate with inner-fold cross-validation on the outer-train rows."""
    # Build the inner CV splits used to score each trial parameter setting.
    splits = make_splits(
        df=training_df,
        split_type=inner_split_type,
        split_params=inner_split_params,
    )
    return _build_tuning_record_for_splits(
        training_df=training_df,
        candidate=candidate,
        tuning_mode="nested",
        tuning_metric=tuning_metric,
        splits=splits,
        n_trials=n_trials,
        random_seed=random_seed,
        selection_scope=selection_scope,
        outer_fold_idx=outer_fold_idx,
        trace_path=trace_path,
        hit_threshold_pkd=hit_threshold_pkd,
        enrichment_top_fractions=enrichment_top_fractions,
        precision_at_n_values=precision_at_n_values,
    )


def tune_candidate(
    training_df: pd.DataFrame,
    candidate: dict,
    tuning_mode: str | None,
    tuning_metric: str,
    holdout_validation_fraction: float = 0.2,
    inner_split_type: str | None = None,
    inner_split_params: dict | None = None,
    outer_fold_idx: int | None = None,
    n_trials: int = 20,
    random_seed: int = 42,
    selection_scope: str | None = None,
    trace_path: str | None = None,
    hit_threshold_pkd: float | None = None,
    enrichment_top_fractions: list[float] | None = None,
    precision_at_n_values: list[int] | None = None,
) -> tuple[dict | None, float | None, pd.DataFrame]:
    """Dispatch to the configured per-candidate tuning strategy."""
    selection_scope = selection_scope or (
        f"outer_fold_{outer_fold_idx}" if outer_fold_idx is not None else "full_labeled_data"
    )
    if tuning_mode is None:
        return tune_candidate_no_tuning(
            training_df=training_df,
            candidate=candidate,
            tuning_metric=tuning_metric,
            outer_fold_idx=outer_fold_idx,
        )
    if tuning_mode == "holdout":
        return tune_candidate_holdout(
            training_df=training_df,
            candidate=candidate,
            tuning_metric=tuning_metric,
            validation_fraction=holdout_validation_fraction,
            n_trials=n_trials,
            random_seed=random_seed,
            selection_scope=selection_scope,
            outer_fold_idx=outer_fold_idx,
            trace_path=trace_path,
            hit_threshold_pkd=hit_threshold_pkd,
            enrichment_top_fractions=enrichment_top_fractions,
            precision_at_n_values=precision_at_n_values,
        )
    if tuning_mode == "nested":
        if inner_split_type is None or inner_split_params is None:
            raise ValueError(
                "inner_split_type and inner_split_params are required for nested tuning"
            )
        return tune_candidate_nested_cv(
            training_df=training_df,
            candidate=candidate,
            tuning_metric=tuning_metric,
            inner_split_type=inner_split_type,
            inner_split_params=inner_split_params,
            n_trials=n_trials,
            random_seed=random_seed,
            selection_scope=selection_scope,
            outer_fold_idx=outer_fold_idx,
            trace_path=trace_path,
            hit_threshold_pkd=hit_threshold_pkd,
            enrichment_top_fractions=enrichment_top_fractions,
            precision_at_n_values=precision_at_n_values,
        )

    raise ValueError(f"Unsupported tuning_mode: {tuning_mode}")
