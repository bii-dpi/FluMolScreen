"""IDE-friendly entrypoint for FluMolScreen dataset assembly.

This script assembles datasets from assay-data and feature tables.
It does not load precompiled datasets from `data/<round_id>/datasets/`.
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

from flumolscreen.assembly import build_target_datasets
from flumolscreen.features.chemical_descriptors import write_chemical_descriptor_features
from flumolscreen.features.derived_predictors import (
    write_derived_6predictor_features,
)

DATASET_NON_FEATURE_COLUMNS = {
    "compound_id",
    "target_id",
    "isomeric_smiles",
    "label_pkd",
}


def generate_derived_feature_family(
    data_dir: str,
    target_id: str,
    feature_set: str,
) -> Path:
    """Generate a derived six-predictor feature table under ``data/shared/features``."""
    if feature_set not in {"6predictor_pr", "6predictor_sc"}:
        raise ValueError(
            "Derived feature generation currently supports only "
            "'6predictor_pr' and '6predictor_sc'."
        )

    shared_features_dir = Path(data_dir) / "shared" / "features"
    input_path = shared_features_dir / f"{target_id}_{feature_set}.csv"
    output_path = shared_features_dir / f"{target_id}_{feature_set}_derived.csv"
    value_suffix = feature_set.rsplit("_", maxsplit=1)[-1]

    write_derived_6predictor_features(
        input_path=input_path,
        output_path=output_path,
        value_suffix=value_suffix,
    )
    return output_path


def compose_target_datasets(
    data_dir: str,
    round_id: str,
    target_id: str,
    feature_requests: list[dict],
    derived_feature_sets_to_generate: list[str] | None = None,
    generate_chemdescriptors: bool = False,
    training_dataset_name: str | None = None,
    inference_dataset_name: str | None = None,
) -> tuple:
    """Build optional prerequisite features, then assemble training and inference datasets."""
    shared_features_dir = Path(data_dir) / "shared" / "features"

    derived_feature_sets_to_generate = derived_feature_sets_to_generate or []

    for feature_set in derived_feature_sets_to_generate:
        output_path = generate_derived_feature_family(
            data_dir=data_dir,
            target_id=target_id,
            feature_set=feature_set,
        )
        print(f"Generated derived {feature_set} features: {output_path}")

    if generate_chemdescriptors:
        input_path = Path(data_dir) / round_id / "assay_data" / f"{target_id}.csv"
        output_path = shared_features_dir / f"{target_id}_chemdescriptors.csv"
        write_chemical_descriptor_features(input_path=input_path, output_path=output_path)
        print(f"Generated chemical descriptors: {output_path}")

    return build_target_datasets(
        data_dir=data_dir,
        round_id=round_id,
        target_id=target_id,
        feature_requests=feature_requests,
        training_dataset_name=training_dataset_name,
        inference_dataset_name=inference_dataset_name,
    )


def main(
    data_dir: str,
    round_id: str,
    target_id: str,
    feature_requests: list[dict],
    derived_feature_sets_to_generate: list[str] | None = None,
    generate_chemdescriptors: bool = False,
    training_dataset_name: str | None = None,
    inference_dataset_name: str | None = None,
) -> None:
    training_df, inference_df, training_path, inference_path = compose_target_datasets(
        data_dir=data_dir,
        round_id=round_id,
        target_id=target_id,
        feature_requests=feature_requests,
        derived_feature_sets_to_generate=derived_feature_sets_to_generate,
        generate_chemdescriptors=generate_chemdescriptors,
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
    DATA_DIR = "data"
    ROUND_ID = "round_synthetic"
    TARGET_ID = "furin"
    DERIVED_FEATURE_SETS_TO_GENERATE: list[str] = ["6predictor_pr"]
    GENERATE_CHEMDESCRIPTORS = True
    TRAINING_DATASET_NAME = "furin_6predictor_pr_derived_chemdescriptors_train.csv"
    INFERENCE_DATASET_NAME = "furin_6predictor_pr_derived_chemdescriptors_inference.csv"
    FEATURE_REQUESTS = [
        {
            "feature_set": "6predictor_pr",
            "source": "shared",
        },
        {
            "feature_set": "6predictor_pr_derived",
            "source": "shared",
        },
        {
            "feature_set": "chemdescriptors",
            "source": "shared",
        },
    ]

    main(
        data_dir=DATA_DIR,
        round_id=ROUND_ID,
        target_id=TARGET_ID,
        feature_requests=FEATURE_REQUESTS,
        derived_feature_sets_to_generate=DERIVED_FEATURE_SETS_TO_GENERATE,
        generate_chemdescriptors=GENERATE_CHEMDESCRIPTORS,
        training_dataset_name=TRAINING_DATASET_NAME,
        inference_dataset_name=INFERENCE_DATASET_NAME,
    )
