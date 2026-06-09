"""Dataset composition helpers for FluMolScreen ML workflows."""

from __future__ import annotations

import pandas as pd

from build_dataset import compose_target_class_datasets, compose_target_datasets
from flumolscreen.ml.utils import select_model_feature_columns

__all__ = ["compose_candidate_datasets"]

ROW_KEY_COLUMNS = ["id", "target"]


def _resolve_candidate_standardization(
    model_run: dict,
    default_standardize_features: bool,
) -> bool:
    """Resolve whether one candidate should standardize features before fitting."""
    return model_run.get("standardize_features", default_standardize_features)


def _key_tuples(df: pd.DataFrame) -> list[tuple]:
    return list(df.loc[:, ROW_KEY_COLUMNS].itertuples(index=False, name=None))


def _validate_unique_keys(df: pd.DataFrame, frame_name: str) -> None:
    if df.duplicated(ROW_KEY_COLUMNS).any():
        duplicate_keys = (
            df.loc[df.duplicated(ROW_KEY_COLUMNS), ROW_KEY_COLUMNS]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(f"{frame_name} contains duplicate row keys: {duplicate_keys}")


def _common_ordered_keys(frames: list[pd.DataFrame], frame_name: str) -> list[tuple]:
    if not frames:
        raise ValueError("frames must contain at least one dataframe")
    for idx, frame in enumerate(frames):
        _validate_unique_keys(frame, f"{frame_name}[{idx}]")

    common_keys = set(_key_tuples(frames[0]))
    for frame in frames[1:]:
        common_keys &= set(_key_tuples(frame))

    ordered_keys = [key for key in _key_tuples(frames[0]) if key in common_keys]
    if not ordered_keys:
        raise ValueError(
            f"No shared complete-case {frame_name} rows remain across comparisons."
        )
    return ordered_keys


def _align_frame_to_keys(df: pd.DataFrame, ordered_keys: list[tuple]) -> pd.DataFrame:
    keyed_df = df.set_index(ROW_KEY_COLUMNS, drop=False)
    aligned_df = keyed_df.reindex(ordered_keys).reset_index(drop=True)
    if aligned_df.isna().all(axis=1).any():
        raise ValueError("Could not align dataframe to shared row keys")
    return aligned_df.loc[:, df.columns].copy()


def _align_candidates_to_common_rows(candidates: list[dict]) -> list[dict]:
    """Filter every comparison/model candidate to the same row universe."""
    if not candidates:
        return candidates

    training_keys = _common_ordered_keys(
        [candidate["training_df"] for candidate in candidates],
        frame_name="training",
    )
    inference_keys = _common_ordered_keys(
        [candidate["inference_df"] for candidate in candidates],
        frame_name="inference",
    )

    aligned_candidates = []
    for candidate in candidates:
        aligned_candidate = {**candidate}
        aligned_candidate["training_df"] = _align_frame_to_keys(
            candidate["training_df"],
            training_keys,
        )
        aligned_candidate["inference_df"] = _align_frame_to_keys(
            candidate["inference_df"],
            inference_keys,
        )
        aligned_candidates.append(aligned_candidate)
    return aligned_candidates


def compose_candidate_datasets(
    data_dir: str,
    round_id: str,
    target: str | None,
    comparisons: list[dict],
    model_runs: list[dict],
    dataset_mode: str = "single_target",
    target_class: str | None = None,
    targets: list[str] | None = None,
    standardize_features: bool = False,
) -> list[dict]:
    """Build one flat candidate per comparison/model combination."""
    if dataset_mode not in {"single_target", "target_class"}:
        raise ValueError("dataset_mode must be one of: 'single_target', 'target_class'")
    if dataset_mode == "single_target" and target is None:
        raise ValueError("target is required when dataset_mode='single_target'")
    if dataset_mode == "target_class" and target_class is None:
        raise ValueError("target_class is required when dataset_mode='target_class'")

    candidates = []

    for comparison in comparisons:
        feature_requests = comparison["feature_requests"]
        comparison_dataset_mode = comparison.get("dataset_mode", dataset_mode)
        if comparison_dataset_mode not in {"single_target", "target_class"}:
            raise ValueError(
                "comparison dataset_mode must be one of: "
                "'single_target', 'target_class'"
            )

        if comparison_dataset_mode == "single_target":
            comparison_target = comparison.get("target", target)
            if comparison_target is None:
                raise ValueError("target is required for single-target comparisons")
            training_df, inference_df, _, _ = compose_target_datasets(
                data_dir=data_dir,
                round_id=round_id,
                target=comparison_target,
                feature_requests=feature_requests,
                training_dataset_name=None,
                inference_dataset_name=None,
            )
        else:
            comparison_target_class = comparison.get("target_class", target_class)
            comparison_targets = comparison.get("targets", targets)
            if comparison_target_class is None:
                raise ValueError(
                    "target_class is required for target-class comparisons"
                )
            training_df, inference_df, _, _ = compose_target_class_datasets(
                data_dir=data_dir,
                round_id=round_id,
                target_class=comparison_target_class,
                targets=comparison_targets,
                feature_requests=feature_requests,
                training_dataset_name=None,
                inference_dataset_name=None,
            )
        p = len(select_model_feature_columns(training_df))

        for model_run in model_runs:
            candidates.append(
                {
                    "comparison_name": comparison["name"],
                    "dataset_mode": comparison_dataset_mode,
                    "feature_requests": feature_requests,
                    "model_type": model_run["model_type"],
                    "base_model_params": model_run.get("model_params"),
                    "standardize_features": _resolve_candidate_standardization(
                        model_run=model_run,
                        default_standardize_features=standardize_features,
                    ),
                    "training_df": training_df,
                    "inference_df": inference_df,
                    "p": p,
                }
            )

    return _align_candidates_to_common_rows(candidates)
