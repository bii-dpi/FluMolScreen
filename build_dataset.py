"""IDE-friendly entrypoint for FluMolScreen dataset assembly.

This script assembles datasets from assay-data and feature tables.
It does not load precompiled datasets from `data/<round_id>/datasets/`.
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

from flumolscreen.assembly import build_target_datasets
from flumolscreen.features.chemical_descriptors import write_chemical_descriptor_features
from flumolscreen.features.derived_predictors import write_derived_6predictor_features


def main(
    data_dir: str,
    round_id: str,
    target_id: str,
    feature_requests: list[dict],
    generate_derived_6predictor: bool = False,
    generate_chemdescriptors: bool = False,
    save_training_dataset: bool = False,
    save_inference_dataset: bool = False,
    training_dataset_name: str | None = None,
    inference_dataset_name: str | None = None,
    output_dataset_dir: str | None = None,
) -> None:
    static_dir = Path(data_dir) / "static_features"

    if generate_derived_6predictor:
        input_path = static_dir / f"{target_id}_6predictor.csv"
        output_path = static_dir / f"{target_id}_6predictor_derived.csv"
        write_derived_6predictor_features(input_path=input_path, output_path=output_path)
        print(f"Generated derived 6predictor features: {output_path}")

    if generate_chemdescriptors:
        input_path = Path(data_dir) / round_id / "assay_data" / f"{target_id}.csv"
        output_path = static_dir / f"{target_id}_chemdescriptors.csv"
        write_chemical_descriptor_features(input_path=input_path, output_path=output_path)
        print(f"Generated chemical descriptors: {output_path}")

    training_df, inference_df, training_path, inference_path = build_target_datasets(
        data_dir=data_dir,
        round_id=round_id,
        target_id=target_id,
        feature_requests=feature_requests,
        save_training_dataset=save_training_dataset,
        save_inference_dataset=save_inference_dataset,
        training_dataset_name=training_dataset_name,
        inference_dataset_name=inference_dataset_name,
    )

    if output_dataset_dir is not None:
        output_dir = Path(output_dataset_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if training_dataset_name is not None:
            training_path = output_dir / training_dataset_name
            training_df.to_csv(training_path, index=False)

        if inference_dataset_name is not None:
            inference_path = output_dir / inference_dataset_name
            inference_df.to_csv(inference_path, index=False)

    print("Feature requests:")
    pprint(feature_requests)
    print("\nTraining data shape:", training_df.shape)
    print("Inference data shape:", inference_df.shape)
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
    GENERATE_DERIVED_6PREDICTOR = False
    GENERATE_CHEMDESCRIPTORS = True
    SAVE_TRAINING_DATASET = False
    SAVE_INFERENCE_DATASET = False
    TRAINING_DATASET_NAME = "furin_6predictor_chemdescriptors_train.csv"
    INFERENCE_DATASET_NAME = "furin_6predictor_chemdescriptors_inference.csv"
    OUTPUT_DATASET_DIR = "data/datasets"
    FEATURE_REQUESTS = [
        {
            "feature_set": "6predictor",
            "source": "static",
        },
        {
            "feature_set": "chemdescriptors",
            "source": "static",
        },
    ]

    main(
        data_dir=DATA_DIR,
        round_id=ROUND_ID,
        target_id=TARGET_ID,
        feature_requests=FEATURE_REQUESTS,
        generate_derived_6predictor=GENERATE_DERIVED_6PREDICTOR,
        generate_chemdescriptors=GENERATE_CHEMDESCRIPTORS,
        save_training_dataset=SAVE_TRAINING_DATASET,
        save_inference_dataset=SAVE_INFERENCE_DATASET,
        training_dataset_name=TRAINING_DATASET_NAME,
        inference_dataset_name=INFERENCE_DATASET_NAME,
        output_dataset_dir=OUTPUT_DATASET_DIR,
    )
