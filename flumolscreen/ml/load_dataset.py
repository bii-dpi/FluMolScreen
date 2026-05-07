"""Dataset composition helpers for FluMolScreen ML workflows."""

from __future__ import annotations

from build_dataset import compose_target_datasets
from flumolscreen.ml.utils import select_model_feature_columns

__all__ = ["compose_candidate_datasets", "infer_feature_generation_settings"]


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
    target_id: str,
    comparisons: list[dict],
    model_runs: list[dict],
) -> list[dict]:
    """Build one flat candidate per comparison/model combination."""
    candidates = []

    for comparison in comparisons:
        # Build the shared train/inference tables once per feature comparison.
        feature_requests = comparison["feature_requests"]
        derived_feature_sets, generate_chemdescriptors = infer_feature_generation_settings(
            feature_requests
        )
        training_df, inference_df, _, _ = compose_target_datasets(
            data_dir=data_dir,
            round_id=round_id,
            target_id=target_id,
            feature_requests=feature_requests,
            derived_feature_sets_to_generate=derived_feature_sets,
            generate_chemdescriptors=generate_chemdescriptors,
            training_dataset_name=None,
            inference_dataset_name=None,
        )
        p = len(select_model_feature_columns(training_df))

        # Flatten the comparison/model grid into one candidate list.
        for model_run in model_runs:
            candidates.append(
                {
                    "comparison_name": comparison["name"],
                    "feature_requests": feature_requests,
                    "model_type": model_run["model_type"],
                    "base_model_params": model_run.get("model_params"),
                    "training_df": training_df,
                    "inference_df": inference_df,
                    "p": p,
                }
            )

    return candidates
