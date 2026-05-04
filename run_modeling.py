"""IDE-friendly entrypoint for FluMolScreen model evaluation and inference."""

from __future__ import annotations

from flumolscreen.assembly import assemble_inference_data, assemble_training_data
from flumolscreen.evaluation import run_split_evaluation
from flumolscreen.modeling import fit_regression_model, predict_regression_model
from flumolscreen.splits import make_random_kfold_splits


def main(
    data_dir: str,
    round_id: str,
    target_id: str,
    feature_requests: list[dict],
    n_splits: int = 5,
    alpha: float = 1.0,
) -> None:
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

    splits = make_random_kfold_splits(training_df, n_splits=n_splits)
    evaluation_df = run_split_evaluation(
        training_df=training_df,
        splits=splits,
        alpha=alpha,
    )

    model, feature_columns = fit_regression_model(
        training_df=training_df,
        alpha=alpha,
    )
    inference_predictions = predict_regression_model(
        model=model,
        df=inference_df,
        feature_columns=feature_columns,
    )

    print("Training data shape:", training_df.shape)
    print("Inference data shape:", inference_df.shape)
    print("\nCross-validation results:")
    print(evaluation_df.to_string(index=False))
    print("\nInference predictions preview:")
    print(inference_predictions.head(5).to_string(index=False))


if __name__ == "__main__":
    DATA_DIR = "data"
    ROUND_ID = "round_synthetic"
    TARGET_ID = "furin"
    N_SPLITS = 5
    ALPHA = 1.0
    FEATURE_REQUESTS = [
        {
            "feature_set": "6predictor",
            "source": "shared",
        },
    ]

    main(
        data_dir=DATA_DIR,
        round_id=ROUND_ID,
        target_id=TARGET_ID,
        feature_requests=FEATURE_REQUESTS,
        n_splits=N_SPLITS,
        alpha=ALPHA,
    )
