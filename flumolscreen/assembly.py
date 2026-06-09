"""Assembly utilities for FluMolScreen training and inference tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flumolscreen.feature_registry import FEATURE_REGISTRY
from flumolscreen.features.source_features import build_target_library
from flumolscreen.loaders import load_assay_data, load_feature_table

DATASET_METADATA_COLUMNS = [
    "id",
    "target",
    "target_class",
    "strain",
    "smiles",
]

NON_ESSENTIAL_ASSEMBLY_COLUMNS = {
    "label_source",
    "round_id",
}


def _merge_feature_frames(
    feature_frames: list[pd.DataFrame],
    feature_sets: list[str],
) -> pd.DataFrame:
    merged = feature_frames[0]

    for feature_set, frame in zip(feature_sets[1:], feature_frames[1:]):
        join_keys = FEATURE_REGISTRY[feature_set]["join_keys"]
        merged = merged.merge(frame, on=join_keys, how="inner")

    return merged


def assemble_features(
    data_dir: Path | str,
    target: str,
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
            target=target,
            feature_set=feature_set,
            round_id=round_id,
            source=source,
            columns=columns,
            base_feature_set=request.get("base_feature_set"),
            feature_columns=request.get("feature_columns"),
        )
        feature_frames.append(feature_frame)
        feature_sets.append(feature_set)

    return _merge_feature_frames(feature_frames, feature_sets)


def _standardize_dataset_frame(
    df: pd.DataFrame,
    include_label: bool,
) -> pd.DataFrame:
    metadata_columns = [col for col in DATASET_METADATA_COLUMNS if col in df.columns]
    label_columns = (
        ["label_pkd"] if include_label and "label_pkd" in df.columns else []
    )
    excluded = set(metadata_columns + label_columns) | NON_ESSENTIAL_ASSEMBLY_COLUMNS
    feature_columns = [col for col in df.columns if col not in excluded]
    ordered_columns = [*metadata_columns, *label_columns, *feature_columns]
    return df.loc[:, ordered_columns].copy()


def assemble_training_data(
    data_dir: Path | str,
    round_id: str,
    target: str,
    feature_requests: list[dict],
) -> pd.DataFrame:
    assay_df = load_assay_data(
        data_dir=data_dir,
        round_id=round_id,
        target=target,
    )
    feature_df = assemble_features(
        data_dir=data_dir,
        target=target,
        feature_requests=feature_requests,
        round_id=round_id,
    )

    assay_base_columns = [
        col
        for col in [*DATASET_METADATA_COLUMNS, "label_pkd"]
        if col in assay_df.columns
    ]
    assay_base_df = assay_df.loc[:, assay_base_columns].copy()
    training_df = assay_base_df.merge(
        feature_df,
        on=["id", "target"],
        how="inner",
    )
    return _standardize_dataset_frame(training_df, include_label=True)


def assemble_inference_data(
    data_dir: Path | str,
    round_id: str,
    target: str,
    feature_requests: list[dict],
) -> pd.DataFrame:
    target_library_df = build_target_library(data_dir=data_dir, target=target)
    feature_df = assemble_features(
        data_dir=data_dir,
        target=target,
        feature_requests=feature_requests,
        round_id=round_id,
    )
    inference_df = target_library_df.merge(
        feature_df,
        on=["id", "target"],
        how="inner",
    )
    return _standardize_dataset_frame(inference_df, include_label=False)


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
    return all(
        request.get("source", "auto") in {"auto", "shared"}
        for request in feature_requests
    )


def build_target_datasets(
    data_dir: Path | str,
    round_id: str,
    target: str,
    feature_requests: list[dict],
    training_dataset_name: str | None = None,
    inference_dataset_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None, Path | None]:
    training_df = assemble_training_data(
        data_dir=data_dir,
        round_id=round_id,
        target=target,
        feature_requests=feature_requests,
    )
    inference_df = assemble_inference_data(
        data_dir=data_dir,
        round_id=round_id,
        target=target,
        feature_requests=feature_requests,
    )

    training_path = None
    inference_path = None

    if training_dataset_name is not None:
        training_path = save_dataset(
            df=training_df,
            data_dir=data_dir,
            dataset_name=training_dataset_name,
            round_id=round_id,
            scope="round",
        )

    if inference_dataset_name is not None:
        inference_scope = (
            "shared"
            if _feature_requests_are_shared_only(feature_requests)
            else "round"
        )
        inference_path = save_dataset(
            df=inference_df,
            data_dir=data_dir,
            dataset_name=inference_dataset_name,
            round_id=round_id,
            scope=inference_scope,
        )

    return training_df, inference_df, training_path, inference_path
