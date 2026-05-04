"""Assembly utilities for FluMolScreen training and inference tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flumolscreen.feature_registry import FEATURE_REGISTRY
from flumolscreen.loaders import load_assay_data, load_feature_table


def _merge_feature_frames(feature_frames: list[pd.DataFrame], feature_sets: list[str]) -> pd.DataFrame:
    merged = feature_frames[0]

    for feature_set, frame in zip(feature_sets[1:], feature_frames[1:]):
        join_keys = FEATURE_REGISTRY[feature_set]["join_keys"]
        merged = merged.merge(frame, on=join_keys, how="inner")

    return merged


def assemble_features(
    data_dir: Path | str,
    target_id: str,
    feature_requests: list[dict],
    round_id: str | None = None,
) -> pd.DataFrame:
    if not feature_requests:
        raise ValueError("feature_requests must contain at least one feature request")

    feature_frames = []
    feature_sets = []

    for request in feature_requests:
        feature_set = request["feature_set"]
        columns = request.get("columns")
        source = request.get("source", "auto")

        feature_frame = load_feature_table(
            data_dir=data_dir,
            target_id=target_id,
            feature_set=feature_set,
            round_id=round_id,
            source=source,
            columns=columns,
        )
        feature_frames.append(feature_frame)
        feature_sets.append(feature_set)

    return _merge_feature_frames(feature_frames, feature_sets)


def assemble_training_data(
    data_dir: Path | str,
    round_id: str,
    target_id: str,
    feature_requests: list[dict],
) -> pd.DataFrame:
    assay_df = load_assay_data(data_dir=data_dir, round_id=round_id, target_id=target_id)
    feature_df = assemble_features(
        data_dir=data_dir,
        target_id=target_id,
        feature_requests=feature_requests,
        round_id=round_id,
    )

    feature_only_columns = [
        col
        for col in feature_df.columns
        if col not in assay_df.columns or col in {"compound_id", "target_id"}
    ]
    feature_df = feature_df.loc[:, feature_only_columns]

    return assay_df.merge(feature_df, on=["compound_id", "target_id"], how="inner")


def assemble_inference_data(
    data_dir: Path | str,
    round_id: str,
    target_id: str,
    feature_requests: list[dict],
) -> pd.DataFrame:
    return assemble_features(
        data_dir=data_dir,
        target_id=target_id,
        feature_requests=feature_requests,
        round_id=round_id,
    )


def separate_features_and_label(
    training_df: pd.DataFrame,
    label_column: str = "label_pkd",
) -> tuple[pd.DataFrame, pd.Series]:
    if label_column not in training_df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    y = training_df[label_column].copy()
    X = training_df.drop(columns=[label_column]).copy()
    return X, y


def save_dataset(
    df: pd.DataFrame,
    data_dir: Path | str,
    dataset_name: str,
    round_id: str | None = None,
    scope: str = "round",
) -> Path:
    if scope not in {"round", "shared"}:
        raise ValueError("scope must be one of: 'round', 'shared'")

    if scope == "round":
        if round_id is None:
            raise ValueError("round_id is required when scope='round'")
        output_dir = Path(data_dir) / round_id / "datasets"
    else:
        output_dir = Path(data_dir) / "shared" / "datasets"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / dataset_name
    df.to_csv(output_path, index=False)
    return output_path


def _feature_requests_are_shared_only(feature_requests: list[dict]) -> bool:
    return all(request.get("source", "auto") == "shared" for request in feature_requests)


def build_target_datasets(
    data_dir: Path | str,
    round_id: str,
    target_id: str,
    feature_requests: list[dict],
    save_training_dataset: bool = False,
    save_inference_dataset: bool = False,
    training_dataset_name: str | None = None,
    inference_dataset_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None, Path | None]:
    training_df = assemble_training_data(
        data_dir=data_dir,
        round_id=round_id,
        target_id=target_id,
        feature_requests=feature_requests,
    )
    inference_df = assemble_inference_data(
        data_dir=data_dir,
        round_id=round_id,
        target_id=target_id,
        feature_requests=feature_requests,
    )

    training_path = None
    inference_path = None

    if save_training_dataset:
        if training_dataset_name is None:
            raise ValueError(
                "training_dataset_name is required when save_training_dataset=True"
            )
        training_path = save_dataset(
            df=training_df,
            data_dir=data_dir,
            dataset_name=training_dataset_name,
            round_id=round_id,
            scope="round",
        )

    if save_inference_dataset:
        if inference_dataset_name is None:
            raise ValueError(
                "inference_dataset_name is required when save_inference_dataset=True"
            )
        inference_scope = "shared" if _feature_requests_are_shared_only(feature_requests) else "round"
        inference_path = save_dataset(
            df=inference_df,
            data_dir=data_dir,
            dataset_name=inference_dataset_name,
            round_id=round_id,
            scope=inference_scope,
        )

    return training_df, inference_df, training_path, inference_path
