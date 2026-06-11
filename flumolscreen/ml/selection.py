"""Official selected-model artifacts for FluMolScreen workflows."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from flumolscreen.feature_registry import METHOD_RANK_SUMMARY_COLUMNS
from flumolscreen.ml.conformal import (
    compute_absolute_standardized_residuals,
    compute_symmetric_conformal_half_width,
    fit_symmetric_conformal_scaler,
)
from flumolscreen.ml.evaluation import fit_regression_model, predict_regression_model
from flumolscreen.ml.inference import fit_bootstrap_ensemble, predict_bootstrap_ensemble
from flumolscreen.ml.splits import make_random_holdout_split
from flumolscreen.ml.utils import format_model_params, round_metrics
from flumolscreen.visualization.chemical_space import compute_murcko_scaffold_table

MAXIMIZE_METRICS = {"spearman", "r2"}
SUPPORTED_SELECTION_RULES = {"one_se_simpler", "best_mean"}
MODEL_COMPLEXITY_RANK = {
    "ridge": 0,
    "xgboost": 1,
}
EVIDENCE_TYPE = "outer_cv_oof_selected_after_cv"
DISAGREEMENT_COLUMNS = [
    column
    for column in METHOD_RANK_SUMMARY_COLUMNS
    if column.endswith("_sd") or column.endswith("_range") or "_minus_" in column
]
KEY_COLUMNS = ["id", "target"]
OOF_BASE_COLUMNS = [
    "dataset_label",
    "evidence_type",
    "outer_fold",
    "id",
    "target",
    "target_class",
    "strain",
    "label_pkd",
    "model_type",
    "comparison_name",
    "p",
    "model_params",
    "pred_mean",
    "pred_err",
    "pred_lower",
    "pred_upper",
    "residual",
    "abs_error",
    "covered",
    "pred_score_decile",
    "target_pred_score_decile",
    "murcko_scaffold",
    "scaffold_type",
    "scaffold_library_count",
    "scaffold_assayed_count",
    "scaffold_coverage_fraction",
    "scaffold_coverage_group",
]

__all__ = [
    "DISAGREEMENT_COLUMNS",
    "EVIDENCE_TYPE",
    "build_selected_oof_predictions",
    "metric_higher_is_better",
    "parse_model_params",
    "select_official_candidate",
    "summarize_candidate_outer_cv",
    "write_selected_model_artifacts",
]


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{table_name} is missing required column(s): {missing_columns}")


def metric_higher_is_better(metric: str) -> bool:
    """Return whether larger values are better for an evaluation metric."""
    metric_key = str(metric)
    return (
        metric_key in MAXIMIZE_METRICS
        or metric_key.startswith("ef_")
        or metric_key.startswith("precision_at_")
    )


def parse_model_params(value: Any) -> dict[str, Any]:
    """Parse a saved model-parameter dictionary value."""
    if isinstance(value, dict):
        return dict(value)
    if value is None or pd.isna(value):
        return {}

    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Could not parse model_params value: {value!r}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"model_params must parse to a dict, got: {type(parsed).__name__}")
    return parsed


def _model_complexity_rank(model_type: str) -> int:
    return MODEL_COMPLEXITY_RANK.get(str(model_type), 100)


def summarize_candidate_outer_cv(
    fold_df: pd.DataFrame,
    metric: str,
    dataset_label: str,
) -> pd.DataFrame:
    """Summarize outer-CV evidence for each model-family/feature candidate."""
    required_columns = ["comparison_name", "model_type", "p", metric]
    _validate_required_columns(fold_df, required_columns, "fold_df")

    candidate_df = fold_df.loc[fold_df[metric].notna()].copy()
    if candidate_df.empty:
        raise ValueError(f"No non-null values found for selection metric: {metric}")

    value_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    sem_column = f"{metric}_sem"
    summary_df = (
        candidate_df.groupby(["comparison_name", "model_type", "p"], as_index=False)
        .agg(
            **{
                value_column: (metric, "mean"),
                std_column: (metric, "std"),
                "n_outer_folds": (metric, "count"),
            }
        )
        .reset_index(drop=True)
    )
    summary_df.insert(0, "dataset_label", dataset_label)
    summary_df[std_column] = summary_df[std_column].fillna(0.0)
    summary_df[sem_column] = summary_df[std_column] / np.sqrt(
        summary_df["n_outer_folds"].clip(lower=1)
    )
    summary_df["model_complexity_rank"] = summary_df["model_type"].map(
        _model_complexity_rank
    )
    return summary_df


def _sort_for_best_mean(
    summary_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    value_column = f"{metric}_mean"
    return summary_df.sort_values(
        [
            value_column,
            "p",
            "model_complexity_rank",
            "model_type",
            "comparison_name",
        ],
        ascending=[
            not metric_higher_is_better(metric),
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
    )


def _sort_for_one_se_selection(
    eligible_df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    value_column = f"{metric}_mean"
    return eligible_df.sort_values(
        [
            "p",
            "model_complexity_rank",
            value_column,
            "model_type",
            "comparison_name",
        ],
        ascending=[
            True,
            True,
            not metric_higher_is_better(metric),
            True,
            True,
        ],
        kind="mergesort",
    )


def select_official_candidate(
    fold_df: pd.DataFrame,
    metric: str,
    dataset_label: str,
    selection_rule: str = "one_se_simpler",
) -> tuple[pd.Series, pd.DataFrame]:
    """Select the official candidate from outer-CV evidence."""
    if selection_rule not in SUPPORTED_SELECTION_RULES:
        raise ValueError(
            f"Unsupported selection_rule: {selection_rule}. "
            f"Supported values: {sorted(SUPPORTED_SELECTION_RULES)}"
        )

    summary_df = summarize_candidate_outer_cv(
        fold_df=fold_df,
        metric=metric,
        dataset_label=dataset_label,
    )
    best_row = _sort_for_best_mean(summary_df, metric=metric).iloc[0]
    value_column = f"{metric}_mean"
    sem_column = f"{metric}_sem"

    if selection_rule == "best_mean":
        threshold = float(best_row[value_column])
        eligible_df = summary_df.copy()
        selected_row = best_row.copy()
    else:
        if metric_higher_is_better(metric):
            threshold = float(best_row[value_column] - best_row[sem_column])
            eligible_df = summary_df.loc[summary_df[value_column] >= threshold].copy()
        else:
            threshold = float(best_row[value_column] + best_row[sem_column])
            eligible_df = summary_df.loc[summary_df[value_column] <= threshold].copy()
        selected_row = _sort_for_one_se_selection(
            eligible_df=eligible_df,
            metric=metric,
        ).iloc[0].copy()

    summary_df["selection_metric"] = metric
    summary_df["selection_rule"] = selection_rule
    summary_df["best_metric_mean"] = float(best_row[value_column])
    summary_df["best_metric_sem"] = float(best_row[sem_column])
    summary_df["one_se_threshold"] = threshold
    summary_df["one_se_eligible"] = summary_df.index.isin(eligible_df.index)
    summary_df["official_selected"] = (
        summary_df["comparison_name"].eq(selected_row["comparison_name"])
        & summary_df["model_type"].eq(selected_row["model_type"])
        & summary_df["p"].eq(selected_row["p"])
    )

    selected_row["selection_metric"] = metric
    selected_row["selection_rule"] = selection_rule
    selected_row["best_comparison_name"] = best_row["comparison_name"]
    selected_row["best_model_type"] = best_row["model_type"]
    selected_row["best_p"] = int(best_row["p"])
    selected_row["best_metric_mean"] = float(best_row[value_column])
    selected_row["best_metric_sem"] = float(best_row[sem_column])
    selected_row["one_se_threshold"] = threshold
    selected_row["one_se_eligible_count"] = int(len(eligible_df))
    return selected_row, summary_df


def _candidate_matches(candidate: dict, selected: pd.Series) -> bool:
    return (
        candidate["comparison_name"] == selected["comparison_name"]
        and candidate["model_type"] == selected["model_type"]
        and int(candidate["p"]) == int(selected["p"])
    )


def _find_selected_candidate(candidates: list[dict], selected: pd.Series) -> dict:
    matches = [candidate for candidate in candidates if _candidate_matches(candidate, selected)]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one selected candidate, found "
            f"{len(matches)} for {selected['model_type']} | {selected['comparison_name']}"
        )
    return matches[0]


def _filter_candidate_rows(df: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    return df.loc[
        df["comparison_name"].eq(selected["comparison_name"])
        & df["model_type"].eq(selected["model_type"])
        & df["p"].astype(int).eq(int(selected["p"]))
    ].copy()


def _resolve_final_model_params(
    final_tuning_df: pd.DataFrame,
    selected: pd.Series,
) -> dict[str, Any]:
    _validate_required_columns(
        final_tuning_df,
        ["comparison_name", "model_type", "p", "model_params"],
        "final_tuning_df",
    )
    candidate_rows = _filter_candidate_rows(final_tuning_df, selected)
    if "selection_scope" in candidate_rows.columns:
        candidate_rows = candidate_rows.loc[
            candidate_rows["selection_scope"].eq("full_labeled_data")
        ]
    if candidate_rows.empty:
        raise ValueError("No full-data tuning row found for the selected candidate")
    return parse_model_params(candidate_rows.iloc[0]["model_params"])


def _resolve_outer_fold_model_params(
    tuning_df: pd.DataFrame,
    selected: pd.Series,
    outer_fold: int,
) -> dict[str, Any]:
    _validate_required_columns(
        tuning_df,
        ["outer_fold", "comparison_name", "model_type", "p", "model_params"],
        "tuning_df",
    )
    candidate_rows = _filter_candidate_rows(tuning_df, selected)
    candidate_rows = candidate_rows.loc[candidate_rows["outer_fold"].notna()].copy()
    candidate_rows["outer_fold"] = candidate_rows["outer_fold"].astype(int)
    fold_rows = candidate_rows.loc[candidate_rows["outer_fold"].eq(int(outer_fold))]
    if fold_rows.empty:
        raise ValueError(
            f"No outer-fold tuning row found for selected candidate fold {outer_fold}"
        )
    return parse_model_params(fold_rows.iloc[0]["model_params"])


def _predict_point_outer_fold(
    candidate: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model, feature_columns = fit_regression_model(
        training_df=train_df,
        model_type=candidate["model_type"],
        model_params=model_params,
        standardize_features=candidate.get("standardize_features", False),
    )
    predictions = predict_regression_model(
        model=model,
        df=test_df,
        feature_columns=feature_columns,
        return_uncertainty=True,
    )
    out = pd.DataFrame(index=test_df.index)
    out["pred_mean"] = predictions["prediction_mean"]
    out["pred_err"] = predictions.get("prediction_err", np.nan)
    return out, {"oof_prediction_mode": "point"}


def _predict_adaptive_conformal_outer_fold(
    candidate: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_params: dict[str, Any],
    calibration_fraction: float,
    n_bootstrap: int,
    interval_coverage: float,
    random_state: int,
    outer_fold: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    (proper_train_idx, calibration_idx) = make_random_holdout_split(
        df=train_df,
        validation_fraction=calibration_fraction,
        random_state=random_state,
    )[0]
    proper_train_df = train_df.iloc[proper_train_idx].reset_index(drop=True)
    calibration_df = train_df.iloc[calibration_idx].reset_index(drop=True)
    models, feature_columns = fit_bootstrap_ensemble(
        training_df=proper_train_df,
        model_type=candidate["model_type"],
        model_params=model_params,
        n_bootstrap=n_bootstrap,
        random_state=random_state + int(outer_fold) + 1,
        standardize_features=candidate.get("standardize_features", False),
    )

    calibration_predictions = predict_bootstrap_ensemble(
        models=models,
        df=calibration_df,
        feature_columns=feature_columns,
    )
    z_scores = compute_absolute_standardized_residuals(
        y_true=calibration_df["label_pkd"],
        prediction_mean=calibration_predictions["prediction_mean"],
        prediction_std=calibration_predictions["prediction_std"],
    )
    q = fit_symmetric_conformal_scaler(
        z_scores=z_scores,
        interval_coverage=interval_coverage,
    )
    test_predictions = predict_bootstrap_ensemble(
        models=models,
        df=test_df,
        feature_columns=feature_columns,
    )
    prediction_err = compute_symmetric_conformal_half_width(
        prediction_std=test_predictions["prediction_std"],
        q=q,
    )

    out = pd.DataFrame(index=test_df.index)
    out["pred_mean"] = test_predictions["prediction_mean"]
    out["pred_err"] = prediction_err
    return out, {
        "oof_prediction_mode": "adaptive_conformal",
        "n_proper_train": int(len(proper_train_df)),
        "n_calibration": int(len(calibration_df)),
        "conformal_q": float(q),
    }


def _build_oof_fold_rows(
    dataset_label: str,
    selected: pd.Series,
    outer_fold: int,
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
    model_params: dict[str, Any],
    fold_metadata: dict[str, Any],
) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in ["id", "target", "target_class", "strain", "label_pkd"]
        if column in test_df.columns
    ]
    out = test_df.loc[:, metadata_columns].copy()
    out.insert(0, "outer_fold", int(outer_fold))
    out.insert(0, "evidence_type", EVIDENCE_TYPE)
    out.insert(0, "dataset_label", dataset_label)
    out["model_type"] = selected["model_type"]
    out["comparison_name"] = selected["comparison_name"]
    out["p"] = int(selected["p"])
    out["model_params"] = format_model_params(model_params)
    out["pred_mean"] = predictions["pred_mean"].to_numpy()
    out["pred_err"] = predictions["pred_err"].to_numpy()
    out["pred_lower"] = out["pred_mean"] - out["pred_err"]
    out["pred_upper"] = out["pred_mean"] + out["pred_err"]
    out["residual"] = out["label_pkd"] - out["pred_mean"]
    out["abs_error"] = out["residual"].abs()
    out["covered"] = np.where(
        out["pred_err"].notna(),
        out["label_pkd"].between(out["pred_lower"], out["pred_upper"]),
        np.nan,
    )
    for key, value in fold_metadata.items():
        out[key] = value
    return out


def _build_candidate_diagnostic_table(candidates: list[dict]) -> pd.DataFrame:
    merged_df: pd.DataFrame | None = None
    for candidate in candidates:
        training_df = candidate["training_df"]
        available_columns = [
            column for column in DISAGREEMENT_COLUMNS if column in training_df.columns
        ]
        if not available_columns:
            continue
        source_df = training_df.loc[:, [*KEY_COLUMNS, *available_columns]].drop_duplicates(
            KEY_COLUMNS
        )
        if merged_df is None:
            merged_df = source_df
        else:
            new_columns = [
                column for column in available_columns if column not in merged_df.columns
            ]
            if new_columns:
                merged_df = merged_df.merge(
                    source_df.loc[:, [*KEY_COLUMNS, *new_columns]],
                    on=KEY_COLUMNS,
                    how="outer",
                )
    if merged_df is None:
        return pd.DataFrame(columns=KEY_COLUMNS)
    return merged_df


def _add_cross_candidate_diagnostics(
    oof_df: pd.DataFrame,
    candidates: list[dict],
) -> pd.DataFrame:
    diagnostic_df = _build_candidate_diagnostic_table(candidates)
    if diagnostic_df.empty or len(diagnostic_df.columns) == len(KEY_COLUMNS):
        return oof_df
    return oof_df.merge(diagnostic_df, on=KEY_COLUMNS, how="left")


def _deduplicate_compound_smiles(df: pd.DataFrame) -> pd.DataFrame:
    _validate_required_columns(df, ["id", "smiles"], "compound table")
    compound_df = df.loc[:, ["id", "smiles"]].drop_duplicates()
    smiles_counts = compound_df.groupby("id")["smiles"].nunique()
    conflicting_ids = smiles_counts[smiles_counts > 1].index.tolist()
    if conflicting_ids:
        raise ValueError(
            "Each compound id must map to one SMILES string for scaffold annotation. "
            f"Found conflicts for: {conflicting_ids[:10]}"
        )
    return compound_df.drop_duplicates("id").reset_index(drop=True)


def _format_scaffold_coverage_group(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value <= 0:
        return "0%"
    if value <= 0.10:
        return "0-10%"
    if value <= 0.25:
        return "10-25%"
    if value <= 0.50:
        return "25-50%"
    return ">50%"


def _build_scaffold_annotations(candidate: dict) -> pd.DataFrame:
    if "smiles" not in candidate["inference_df"].columns:
        return pd.DataFrame(columns=[*KEY_COLUMNS, "murcko_scaffold"])

    library_compounds = _deduplicate_compound_smiles(candidate["inference_df"])
    try:
        scaffold_df = compute_murcko_scaffold_table(library_compounds)
    except ImportError:
        out = library_compounds.loc[:, ["id"]].copy()
        out["murcko_scaffold"] = np.nan
        out["scaffold_type"] = np.nan
        out["scaffold_library_count"] = np.nan
        out["scaffold_assayed_count"] = np.nan
        out["scaffold_coverage_fraction"] = np.nan
        out["scaffold_coverage_group"] = "unknown"
        return out

    library_counts = (
        scaffold_df["murcko_scaffold"].value_counts().rename("scaffold_library_count")
    )
    assayed_ids = candidate["training_df"].loc[:, ["id"]].drop_duplicates()
    assayed_scaffold_df = scaffold_df.merge(assayed_ids, on="id", how="inner")
    assayed_counts = (
        assayed_scaffold_df["murcko_scaffold"]
        .value_counts()
        .rename("scaffold_assayed_count")
    )
    out = scaffold_df.merge(
        library_counts,
        left_on="murcko_scaffold",
        right_index=True,
        how="left",
    )
    out = out.merge(
        assayed_counts,
        left_on="murcko_scaffold",
        right_index=True,
        how="left",
    )
    out["scaffold_assayed_count"] = out["scaffold_assayed_count"].fillna(0).astype(int)
    out["scaffold_coverage_fraction"] = (
        out["scaffold_assayed_count"] / out["scaffold_library_count"]
    )
    out["scaffold_coverage_group"] = out["scaffold_coverage_fraction"].map(
        _format_scaffold_coverage_group
    )
    return out


def _add_scaffold_annotations(oof_df: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    scaffold_df = _build_scaffold_annotations(candidate)
    if scaffold_df.empty:
        return oof_df
    scaffold_columns = [
        column for column in scaffold_df.columns if column != "id"
    ]
    return oof_df.merge(scaffold_df.loc[:, ["id", *scaffold_columns]], on="id", how="left")


def _assign_deciles_for_series(values: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid_values = values.dropna()
    if valid_values.empty:
        return out
    n_bins = min(10, int(len(valid_values)))
    if n_bins <= 1:
        out.loc[valid_values.index] = 1
        return out
    ranks = valid_values.rank(method="first", ascending=True)
    deciles = pd.qcut(
        ranks,
        q=n_bins,
        labels=False,
        duplicates="drop",
    )
    out.loc[valid_values.index] = (deciles + 1).astype("Int64")
    return out


def _add_score_deciles(oof_df: pd.DataFrame) -> pd.DataFrame:
    out = oof_df.copy()
    out["pred_score_decile"] = _assign_deciles_for_series(out["pred_mean"])
    out["target_pred_score_decile"] = (
        out.groupby("target", group_keys=False)["pred_mean"]
        .apply(_assign_deciles_for_series)
        .astype("Int64")
    )
    return out


def _order_oof_columns(oof_df: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [
        column for column in OOF_BASE_COLUMNS if column in oof_df.columns
    ]
    ordered_columns.extend(
        column for column in DISAGREEMENT_COLUMNS if column in oof_df.columns
    )
    ordered_columns.extend(
        column for column in oof_df.columns if column not in set(ordered_columns)
    )
    return oof_df.loc[:, ordered_columns]


def build_selected_oof_predictions(
    dataset_label: str,
    selected: pd.Series,
    selected_candidate: dict,
    candidates: list[dict],
    outer_splits: list[tuple],
    tuning_df: pd.DataFrame,
    inference_mode: str,
    calibration_fraction: float,
    n_bootstrap: int,
    interval_coverage: float,
    random_state: int,
) -> pd.DataFrame:
    """Build held-out predictions for the official selected candidate."""
    training_df = selected_candidate["training_df"]
    oof_rows = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        train_df = training_df.iloc[outer_train_idx].reset_index(drop=True)
        test_df = training_df.iloc[outer_test_idx].reset_index(drop=True)
        model_params = _resolve_outer_fold_model_params(
            tuning_df=tuning_df,
            selected=selected,
            outer_fold=outer_fold,
        )

        if inference_mode == "adaptive_conformal":
            predictions, fold_metadata = _predict_adaptive_conformal_outer_fold(
                candidate=selected_candidate,
                train_df=train_df,
                test_df=test_df,
                model_params=model_params,
                calibration_fraction=calibration_fraction,
                n_bootstrap=n_bootstrap,
                interval_coverage=interval_coverage,
                random_state=random_state,
                outer_fold=outer_fold,
            )
        elif inference_mode == "point":
            predictions, fold_metadata = _predict_point_outer_fold(
                candidate=selected_candidate,
                train_df=train_df,
                test_df=test_df,
                model_params=model_params,
            )
        else:
            raise ValueError(f"Unsupported inference_mode for OOF diagnostics: {inference_mode}")

        oof_rows.append(
            _build_oof_fold_rows(
                dataset_label=dataset_label,
                selected=selected,
                outer_fold=outer_fold,
                test_df=test_df,
                predictions=predictions,
                model_params=model_params,
                fold_metadata=fold_metadata,
            )
        )

    oof_df = pd.concat(oof_rows, ignore_index=True)
    oof_df = _add_cross_candidate_diagnostics(oof_df, candidates=candidates)
    oof_df = _add_scaffold_annotations(oof_df, candidate=selected_candidate)
    oof_df = _add_score_deciles(oof_df)
    oof_df = _order_oof_columns(oof_df)
    return round_metrics(oof_df)


def _copy_selected_final_inference(
    inference_paths: dict[tuple[str, str], Path],
    selected: pd.Series,
    output_dir: Path,
    dataset_label: str,
) -> Path:
    key = (selected["comparison_name"], selected["model_type"])
    if key not in inference_paths:
        raise ValueError(f"No final inference path found for selected candidate: {key}")
    output_path = output_dir / f"{dataset_label}_selected_final_inference.csv"
    shutil.copyfile(inference_paths[key], output_path)
    return output_path


def _build_selected_summary_output(
    candidate_summary_df: pd.DataFrame,
    selected: pd.Series,
    final_model_params: dict[str, Any],
    final_inference_path: Path,
    oof_predictions_path: Path | None,
) -> pd.DataFrame:
    out = candidate_summary_df.copy()
    out["final_model_params"] = ""
    out["selected_final_inference_path"] = ""
    out["selected_oof_predictions_path"] = ""
    out["evidence_type"] = ""

    selected_mask = out["official_selected"].astype(bool)
    out.loc[selected_mask, "final_model_params"] = format_model_params(final_model_params)
    out.loc[selected_mask, "selected_final_inference_path"] = str(final_inference_path)
    out.loc[selected_mask, "selected_oof_predictions_path"] = (
        "" if oof_predictions_path is None else str(oof_predictions_path)
    )
    out.loc[selected_mask, "evidence_type"] = EVIDENCE_TYPE
    return out


def _write_manifest(
    selected: pd.Series,
    final_model_params: dict[str, Any],
    output_path: Path,
    final_inference_path: Path,
    summary_path: Path,
    oof_predictions_path: Path | None,
) -> Path:
    manifest = {
        "dataset_label": selected["dataset_label"],
        "model_type": selected["model_type"],
        "comparison_name": selected["comparison_name"],
        "p": int(selected["p"]),
        "selection": {
            "metric": selected["selection_metric"],
            "rule": selected["selection_rule"],
            "outer_cv_metric_mean": float(
                selected[f"{selected['selection_metric']}_mean"]
            ),
            "outer_cv_metric_std": float(
                selected[f"{selected['selection_metric']}_std"]
            ),
            "outer_cv_metric_sem": float(
                selected[f"{selected['selection_metric']}_sem"]
            ),
            "n_outer_folds": int(selected["n_outer_folds"]),
            "best_metric_mean": float(selected["best_metric_mean"]),
            "best_metric_sem": float(selected["best_metric_sem"]),
            "one_se_threshold": float(selected["one_se_threshold"]),
            "one_se_eligible_count": int(selected["one_se_eligible_count"]),
            "best_candidate": {
                "comparison_name": selected["best_comparison_name"],
                "model_type": selected["best_model_type"],
                "p": int(selected["best_p"]),
            },
        },
        "final_model_params": final_model_params,
        "artifacts": {
            "selected_model_summary": str(summary_path),
            "selected_final_inference": str(final_inference_path),
            "selected_oof_predictions": (
                None if oof_predictions_path is None else str(oof_predictions_path)
            ),
        },
        "evidence_type": EVIDENCE_TYPE,
        "diagnostic_caveat": (
            "Selected OOF diagnostics are held out by outer CV for the selected "
            "candidate after model selection. A later assay round is stronger "
            "external validation."
        ),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return output_path


def write_selected_model_artifacts(
    output_dir: Path,
    dataset_label: str,
    candidates: list[dict],
    fold_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    final_tuning_df: pd.DataFrame,
    inference_paths: dict[tuple[str, str], Path],
    outer_splits: list[tuple],
    selection_metric: str,
    selection_rule: str,
    inference_mode: str,
    calibration_fraction: float,
    n_bootstrap: int,
    interval_coverage: float,
    random_state: int,
    write_selected_oof_diagnostics: bool = True,
) -> dict[str, Any]:
    """Write official selected-model manifest, summary, final inference, and OOF data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, candidate_summary_df = select_official_candidate(
        fold_df=fold_df,
        metric=selection_metric,
        dataset_label=dataset_label,
        selection_rule=selection_rule,
    )
    selected_candidate = _find_selected_candidate(candidates=candidates, selected=selected)
    final_model_params = _resolve_final_model_params(
        final_tuning_df=final_tuning_df,
        selected=selected,
    )

    final_inference_path = _copy_selected_final_inference(
        inference_paths=inference_paths,
        selected=selected,
        output_dir=output_dir,
        dataset_label=dataset_label,
    )

    oof_predictions_path = None
    if write_selected_oof_diagnostics:
        oof_predictions_path = output_dir / f"{dataset_label}_selected_oof_predictions.csv"
        oof_df = build_selected_oof_predictions(
            dataset_label=dataset_label,
            selected=selected,
            selected_candidate=selected_candidate,
            candidates=candidates,
            outer_splits=outer_splits,
            tuning_df=tuning_df,
            inference_mode=inference_mode,
            calibration_fraction=calibration_fraction,
            n_bootstrap=n_bootstrap,
            interval_coverage=interval_coverage,
            random_state=random_state,
        )
        oof_df.to_csv(oof_predictions_path, index=False)

    summary_path = output_dir / f"{dataset_label}_selected_model_summary.csv"
    summary_output_df = _build_selected_summary_output(
        candidate_summary_df=candidate_summary_df,
        selected=selected,
        final_model_params=final_model_params,
        final_inference_path=final_inference_path,
        oof_predictions_path=oof_predictions_path,
    )
    round_metrics(summary_output_df).to_csv(summary_path, index=False)

    manifest_path = _write_manifest(
        selected=selected,
        final_model_params=final_model_params,
        output_path=output_dir / f"{dataset_label}_selected_model.yml",
        final_inference_path=final_inference_path,
        summary_path=summary_path,
        oof_predictions_path=oof_predictions_path,
    )

    return {
        "selected": selected,
        "candidate_summary_df": candidate_summary_df,
        "final_model_params": final_model_params,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "final_inference_path": final_inference_path,
        "oof_predictions_path": oof_predictions_path,
    }
