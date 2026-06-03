"""Dataset composition helpers for FluMolScreen ML workflows."""

from __future__ import annotations

from build_dataset import compose_target_datasets, compose_target_family_datasets
from flumolscreen.ml.utils import select_model_feature_columns

__all__ = ["compose_candidate_datasets", "infer_feature_generation_settings"]


def _resolve_candidate_standardization(
    model_run: dict,
    default_standardize_features: bool,
) -> bool:
    """Resolve whether one candidate should standardize features before fitting."""
    return model_run.get("standardize_features", default_standardize_features)


def infer_feature_generation_settings(
    feature_requests: list[dict],
) -> tuple[list[str], bool]:
    """Infer which shared feature families must be generated before assembly."""
    derived_feature_sets = []
    generate_chemdescriptors = False

    for request in feature_requests:
        feature_set = request["feature_set"]
        if feature_set.endswith("_derived"):
            base_feature_set = feature_set.removesuffix("_derived")
            if base_feature_set not in derived_feature_sets:
                derived_feature_sets.append(base_feature_set)
        if feature_set == "chemdescriptors":
            generate_chemdescriptors = True

    return derived_feature_sets, generate_chemdescriptors


def compose_candidate_datasets(
    data_dir: str,
    round_id: str,
    target_id: str | None,
    comparisons: list[dict],
    model_runs: list[dict],
    dataset_mode: str = "single_target",
    family_key: str | None = None,
    target_ids: list[str] | None = None,
    reference_target_id: str | None = None,
    target_id_to_label: dict[str, str] | None = None,
    standardize_features: bool = False,
) -> list[dict]:
    """Build one flat candidate per comparison/model combination."""
    if dataset_mode not in {"single_target", "target_family"}:
        raise ValueError("dataset_mode must be one of: 'single_target', 'target_family'")
    if dataset_mode == "single_target" and target_id is None:
        raise ValueError("target_id is required when dataset_mode='single_target'")

    candidates = []

    for comparison in comparisons:
        # Build the shared train/inference tables once per feature comparison.
        feature_requests = comparison["feature_requests"]
        derived_feature_sets, generate_chemdescriptors = infer_feature_generation_settings(
            feature_requests
        )

        comparison_dataset_mode = comparison.get("dataset_mode", dataset_mode)
        if comparison_dataset_mode not in {"single_target", "target_family"}:
            raise ValueError(
                "comparison dataset_mode must be one of: "
                "'single_target', 'target_family'"
            )

        if comparison_dataset_mode == "single_target":
            comparison_target_id = comparison.get("target_id", target_id)
            if comparison_target_id is None:
                raise ValueError(
                    "target_id is required for single-target comparisons"
                )
            training_df, inference_df, _, _ = compose_target_datasets(
                data_dir=data_dir,
                round_id=round_id,
                target_id=comparison_target_id,
                feature_requests=feature_requests,
                derived_feature_sets_to_generate=derived_feature_sets,
                generate_chemdescriptors=generate_chemdescriptors,
                training_dataset_name=None,
                inference_dataset_name=None,
            )
        else:
            comparison_family_key = comparison.get("family_key", family_key)
            comparison_target_ids = comparison.get("target_ids", target_ids)
            comparison_reference_target_id = comparison.get(
                "reference_target_id",
                reference_target_id,
            )
            comparison_target_id_to_label = comparison.get(
                "target_id_to_label",
                target_id_to_label,
            )
            training_df, inference_df, _, _ = compose_target_family_datasets(
                data_dir=data_dir,
                round_id=round_id,
                target_ids=comparison_target_ids,
                reference_target_id=comparison_reference_target_id,
                feature_requests=feature_requests,
                derived_feature_sets_to_generate=derived_feature_sets,
                generate_chemdescriptors=generate_chemdescriptors,
                training_dataset_name=None,
                inference_dataset_name=None,
                target_id_to_label=comparison_target_id_to_label,
                family_key=comparison_family_key,
            )
        p = len(select_model_feature_columns(training_df))

        # Flatten the comparison/model grid into one candidate list.
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

    return candidates
