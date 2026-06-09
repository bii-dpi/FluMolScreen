"""IDE-friendly entrypoint for FluMolScreen dataset assembly."""

from __future__ import annotations

from pprint import pprint

import pandas as pd

from flumolscreen.assembly import build_target_datasets, save_dataset
from flumolscreen.target_registry import resolve_target_class_targets

DATASET_NON_FEATURE_COLUMNS = {
    "id",
    "target",
    "target_class",
    "strain",
    "smiles",
    "label_pkd",
}


def compose_target_datasets(
    data_dir: str,
    round_id: str,
    target: str,
    feature_requests: list[dict],
    training_dataset_name: str | None = None,
    inference_dataset_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, object | None, object | None]:
    """Assemble training and inference datasets for one concrete target."""
    return build_target_datasets(
        data_dir=data_dir,
        round_id=round_id,
        target=target,
        feature_requests=feature_requests,
        training_dataset_name=training_dataset_name,
        inference_dataset_name=inference_dataset_name,
    )


def compose_target_class_datasets(
    data_dir: str,
    round_id: str,
    target_class: str,
    feature_requests: list[dict],
    targets: list[str] | None = None,
    training_dataset_name: str | None = None,
    inference_dataset_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, object | None, object | None]:
    """Assemble pooled datasets across all targets in one target class."""
    resolved_targets = targets or resolve_target_class_targets(target_class)

    training_tables = []
    inference_tables = []
    for target in resolved_targets:
        training_df, inference_df, _, _ = compose_target_datasets(
            data_dir=data_dir,
            round_id=round_id,
            target=target,
            feature_requests=feature_requests,
            training_dataset_name=None,
            inference_dataset_name=None,
        )
        training_tables.append(training_df)
        inference_tables.append(inference_df)

    pooled_training_df = pd.concat(training_tables, axis=0, ignore_index=True)
    pooled_inference_df = pd.concat(inference_tables, axis=0, ignore_index=True)

    training_path = None
    inference_path = None

    if training_dataset_name is not None:
        training_path = save_dataset(
            df=pooled_training_df,
            data_dir=data_dir,
            dataset_name=training_dataset_name,
            round_id=round_id,
            scope="round",
        )

    if inference_dataset_name is not None:
        inference_path = save_dataset(
            df=pooled_inference_df,
            data_dir=data_dir,
            dataset_name=inference_dataset_name,
            round_id=round_id,
            scope="shared",
        )

    return pooled_training_df, pooled_inference_df, training_path, inference_path


def main(
    data_dir: str,
    round_id: str,
    target: str,
    feature_requests: list[dict],
    training_dataset_name: str | None = None,
    inference_dataset_name: str | None = None,
) -> None:
    training_df, inference_df, training_path, inference_path = compose_target_datasets(
        data_dir=data_dir,
        round_id=round_id,
        target=target,
        feature_requests=feature_requests,
        training_dataset_name=training_dataset_name,
        inference_dataset_name=inference_dataset_name,
    )

    feature_columns = [
        column
        for column in training_df.columns
        if column not in DATASET_NON_FEATURE_COLUMNS
    ]

    print("Feature requests:")
    pprint(feature_requests)
    print("\nTraining dataset size:", training_df.shape)
    print("Inference dataset size:", inference_df.shape)
    print(f"\nFeature columns included ({len(feature_columns)}):")
    for column in feature_columns:
        print(f"- {column}")
    if training_path is not None:
        print(f"Saved training dataset to: {training_path}")
    if inference_path is not None:
        print(f"Saved inference dataset to: {inference_path}")
    print("\nTraining data preview:")
    print(training_df.head(3).to_string(index=False))
    print("\nInference data preview:")
    print(inference_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main(
        data_dir="data",
        round_id="round_synthetic",
        target="furin",
        feature_requests=[
            {"feature_set": "method_scores"},
            {"feature_set": "method_ranks"},
            {"feature_set": "method_rank_summary"},
            {"feature_set": "glide_uncertainty"},
            {"feature_set": "chemical_descriptors"},
        ],
    )
