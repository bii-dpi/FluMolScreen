"""IDE-friendly entrypoint for FluMolScreen dataset assembly.

This script assembles datasets from assay-data and feature tables.
It does not load precompiled datasets from `data/<round_id>/datasets/`.
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

import pandas as pd

from flumolscreen.assembly import build_target_datasets, save_dataset
from flumolscreen.feature_registry import (
    FEATURE_REGISTRY,
    resolve_hierarchical_reference_target_id,
    resolve_hierarchical_target_id_to_label,
    resolve_hierarchical_target_ids,
)
from flumolscreen.features.chemical_descriptors import write_chemical_descriptor_features
from flumolscreen.features.derived_predictors import (
    write_derived_6predictor_features,
)
from flumolscreen.features.hierarchical_perstrain import (
    build_hierarchical_perstrain_features,
)
from flumolscreen.loaders import load_feature_table

DATASET_NON_FEATURE_COLUMNS = {
    "compound_id",
    "target_id",
    "isomeric_smiles",
    "label_pkd",
}


def _feature_requests_are_shared_only(feature_requests: list[dict]) -> bool:
    """Return True when all feature requests read from shared sources."""
    return all(request.get("source", "auto") == "shared" for request in feature_requests)


def _generate_target_prerequisites(
    data_dir: str,
    round_id: str,
    target_id: str,
    derived_feature_sets_to_generate: list[str],
    generate_chemdescriptors: bool,
) -> None:
    """Generate any per-target prerequisite shared feature tables."""
    shared_features_dir = Path(data_dir) / "shared" / "features"

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


def _resolve_target_family_inputs(
    family_key: str | None,
    target_ids: list[str] | None,
    reference_target_id: str | None,
    target_id_to_label: dict[str, str] | None,
) -> tuple[list[str], str, dict[str, str]]:
    """Resolve pooled-family inputs, optionally deriving them from a family key."""
    if family_key is not None:
        # The family registry is the canonical source for pooled target ids,
        # the reference target, and the short labels used in interaction names.
        target_ids = target_ids or resolve_hierarchical_target_ids(family_key)
        reference_target_id = (
            reference_target_id or resolve_hierarchical_reference_target_id(family_key)
        )
        target_id_to_label = (
            target_id_to_label
            or resolve_hierarchical_target_id_to_label(family_key, target_ids)
        )

    if not target_ids:
        raise ValueError(
            "target_ids must contain at least one target_id, or supply family_key."
        )
    if reference_target_id is None:
        raise ValueError(
            "reference_target_id must be provided, or supply family_key."
        )
    if target_id_to_label is None:
        raise ValueError(
            "target_id_to_label must be provided, or supply family_key."
        )

    return target_ids, reference_target_id, target_id_to_label


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
    derived_feature_sets_to_generate = derived_feature_sets_to_generate or []
    _generate_target_prerequisites(
        data_dir=data_dir,
        round_id=round_id,
        target_id=target_id,
        derived_feature_sets_to_generate=derived_feature_sets_to_generate,
        generate_chemdescriptors=generate_chemdescriptors,
    )

    return build_target_datasets(
        data_dir=data_dir,
        round_id=round_id,
        target_id=target_id,
        feature_requests=feature_requests,
        training_dataset_name=training_dataset_name,
        inference_dataset_name=inference_dataset_name,
    )


def compose_target_family_datasets(
    data_dir: str,
    round_id: str,
    target_ids: list[str] | None,
    reference_target_id: str | None,
    feature_requests: list[dict],
    derived_feature_sets_to_generate: list[str] | None = None,
    generate_chemdescriptors: bool = False,
    training_dataset_name: str | None = None,
    inference_dataset_name: str | None = None,
    target_id_to_label: dict[str, str] | None = None,
    family_key: str | None = None,
) -> tuple:
    """Build pooled training and inference datasets across multiple target ids.

    This function keeps the existing single-target assembly path intact by:
    1. generating any requested per-target prerequisite feature tables,
    2. generating pooled hierarchical-per-strain features when requested,
    3. splitting the hierarchical feature family back into per-target shared CSVs,
    4. reusing ``build_target_datasets(...)`` for each target independently, and
    5. concatenating the resulting datasets into one pooled output.
    """
    target_ids, reference_target_id, target_id_to_label = _resolve_target_family_inputs(
        family_key=family_key,
        target_ids=target_ids,
        reference_target_id=reference_target_id,
        target_id_to_label=target_id_to_label,
    )

    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target_ids must not contain duplicates")
    if reference_target_id not in target_ids:
        raise ValueError("reference_target_id must be present in target_ids")

    shared_features_dir = Path(data_dir) / "shared" / "features"
    derived_feature_sets_to_generate = derived_feature_sets_to_generate or []

    for target_id in target_ids:
        # Derived features and chemistry descriptors are still generated per
        # concrete target because their source tables remain target-specific.
        _generate_target_prerequisites(
            data_dir=data_dir,
            round_id=round_id,
            target_id=target_id,
            derived_feature_sets_to_generate=derived_feature_sets_to_generate,
            generate_chemdescriptors=generate_chemdescriptors,
        )

    hierarchical_requests = [
        request
        for request in feature_requests
        if request.get("feature_generator") == "hierarchical_perstrain"
    ]

    for request in hierarchical_requests:
        feature_set = request["feature_set"]
        base_feature_set = request["base_feature_set"]
        feature_columns = request.get(
            "feature_columns",
            FEATURE_REGISTRY[base_feature_set]["default_columns"],
        )
        base_source = request.get("base_source", request.get("source", "shared"))

        # Hierarchical per-strain features are generated from one pooled base
        # table, then split back into per-target shared feature CSVs so the
        # existing single-target assembly code can be reused unchanged.
        pooled_base_df = pd.concat(
            [
                load_feature_table(
                    data_dir=data_dir,
                    target_id=target_id,
                    feature_set=base_feature_set,
                    round_id=round_id,
                    source=base_source,
                    columns=feature_columns,
                )
                for target_id in target_ids
            ],
            axis=0,
            ignore_index=True,
        )

        pooled_hierarchical_df = build_hierarchical_perstrain_features(
            df=pooled_base_df,
            reference_target_id=reference_target_id,
            feature_columns=feature_columns,
            target_id_to_label=target_id_to_label,
        )

        for target_id in target_ids:
            output_path = shared_features_dir / f"{target_id}_{feature_set}.csv"
            target_feature_df = pooled_hierarchical_df[
                pooled_hierarchical_df["target_id"] == target_id
            ].copy()
            target_feature_df.to_csv(output_path, index=False)
            print(f"Generated hierarchical {feature_set} features: {output_path}")

    training_tables = []
    inference_tables = []
    for target_id in target_ids:
        # Reuse the standard single-target dataset builder for each member of
        # the family, then concatenate the assembled datasets row-wise.
        training_df, inference_df, _, _ = compose_target_datasets(
            data_dir=data_dir,
            round_id=round_id,
            target_id=target_id,
            feature_requests=feature_requests,
            derived_feature_sets_to_generate=None,
            generate_chemdescriptors=False,
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
        inference_scope = (
            "shared" if _feature_requests_are_shared_only(feature_requests) else "round"
        )
        inference_path = save_dataset(
            df=pooled_inference_df,
            data_dir=data_dir,
            dataset_name=inference_dataset_name,
            round_id=round_id,
            scope=inference_scope,
        )

    return pooled_training_df, pooled_inference_df, training_path, inference_path


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
