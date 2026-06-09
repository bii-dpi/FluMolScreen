"""Source-native feature builders from prediction and compound CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from flumolscreen.feature_registry import (
    BRANCH_METHODS,
    FEATURE_REGISTRY,
    GLIDE_UNCERTAINTY_COLUMN,
    METHOD_PREDICTION_FILES,
    METHOD_RANK_COLUMNS,
    METHOD_RANK_SUMMARY_COLUMNS,
    METHOD_SCORE_COLUMNS,
    METHODS,
)
from flumolscreen.features.chemical_descriptors import build_chemical_descriptor_features
from flumolscreen.target_registry import (
    TARGET_REGISTRY,
    resolve_compound_file_for_target_class,
    resolve_target_class_for_target,
    resolve_target_class_reference_target,
    resolve_target_class_targets,
    resolve_target_to_label,
)

KEY_COLUMNS = ["id", "target"]
COMPOUND_COLUMNS = ["id", "target", "target_class", "strain", "smiles"]


def _normalize_data_dir(data_dir: Path | str) -> Path:
    return Path(data_dir)


def _validate_target(target: str) -> None:
    if target not in TARGET_REGISTRY:
        raise ValueError(f"Unknown target: {target}")


def _validate_feature_set(feature_set: str) -> None:
    if feature_set not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature_set: {feature_set}")


def _select_feature_columns(
    df: pd.DataFrame,
    feature_set: str,
    columns: list[str] | None,
) -> pd.DataFrame:
    requested = FEATURE_REGISTRY[feature_set]["default_columns"] if columns is None else columns
    selected_columns = list(dict.fromkeys([*KEY_COLUMNS, *requested]))
    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{feature_set} table is missing requested column(s): {missing_columns}"
        )
    return df.loc[:, selected_columns].copy()


def _merge_complete_case(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("frames must contain at least one dataframe")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=KEY_COLUMNS, how="inner")
    return merged


def load_compound_library(
    data_dir: Path | str,
    target_class: str,
) -> pd.DataFrame:
    """Load the source compound map for one target class."""
    compound_file = resolve_compound_file_for_target_class(target_class)
    path = _normalize_data_dir(data_dir) / "shared" / "datasets" / compound_file
    if not path.exists():
        raise FileNotFoundError(f"Compound library file not found: {path}")

    df = pd.read_csv(path)
    required_columns = ["id", "smiles"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required column(s): {missing_columns}")
    if df["id"].duplicated().any():
        duplicate_ids = df.loc[df["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"{path} contains duplicate id values: {duplicate_ids}")

    return df.loc[:, required_columns].copy()


def build_target_library(
    data_dir: Path | str,
    target: str,
) -> pd.DataFrame:
    """Return the compound metadata table for one concrete target."""
    _validate_target(target)
    target_spec = TARGET_REGISTRY[target]
    target_class = target_spec["target_class"]
    compound_df = load_compound_library(data_dir=data_dir, target_class=target_class)
    out = compound_df.copy()
    out.insert(1, "target", target)
    out.insert(2, "target_class", target_class)
    out.insert(3, "strain", target_spec["strain"])
    return out.loc[:, COMPOUND_COLUMNS]


def load_method_prediction_table(
    data_dir: Path | str,
    method: str,
) -> pd.DataFrame:
    """Load and validate one source method prediction file."""
    if method not in METHOD_PREDICTION_FILES:
        raise ValueError(f"Unknown method: {method}")
    path = (
        _normalize_data_dir(data_dir)
        / "shared"
        / "features"
        / METHOD_PREDICTION_FILES[method]
    )
    if not path.exists():
        raise FileNotFoundError(f"Method prediction file not found: {path}")

    df = pd.read_csv(path)
    required_columns = ["id", "target", "prediction"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required column(s): {missing_columns}")
    if df.duplicated(KEY_COLUMNS).any():
        duplicate_keys = (
            df.loc[df.duplicated(KEY_COLUMNS), KEY_COLUMNS]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(f"{path} contains duplicate id/target rows: {duplicate_keys}")
    return df.copy()


def _build_one_method_score_table(
    data_dir: Path | str,
    method: str,
    target: str,
) -> pd.DataFrame:
    df = load_method_prediction_table(data_dir=data_dir, method=method)
    score_column = f"{method}_score"
    target_df = df[df["target"] == target].loc[:, [*KEY_COLUMNS, "prediction"]].copy()
    target_df = target_df.rename(columns={"prediction": score_column})
    return target_df


def build_method_scores(
    data_dir: Path | str,
    target: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build six raw method score columns for one target using complete cases."""
    _validate_target(target)
    frames = [
        _build_one_method_score_table(
            data_dir=data_dir,
            method=method,
            target=target,
        )
        for method in METHODS
    ]
    return _select_feature_columns(
        _merge_complete_case(frames),
        feature_set="method_scores",
        columns=columns,
    )


def _build_one_method_rank_table(
    data_dir: Path | str,
    method: str,
    target: str,
) -> pd.DataFrame:
    df = load_method_prediction_table(data_dir=data_dir, method=method)
    rank_column = f"{method}_rank"
    target_df = df[df["target"] == target].loc[:, [*KEY_COLUMNS, "prediction"]].copy()
    target_df[rank_column] = target_df["prediction"].rank(
        method="average",
        pct=True,
    )
    return target_df.loc[:, [*KEY_COLUMNS, rank_column]]


def build_method_ranks(
    data_dir: Path | str,
    target: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build per-target percentile ranks from real method outputs."""
    _validate_target(target)
    frames = [
        _build_one_method_rank_table(
            data_dir=data_dir,
            method=method,
            target=target,
        )
        for method in METHODS
    ]
    return _select_feature_columns(
        _merge_complete_case(frames),
        feature_set="method_ranks",
        columns=columns,
    )


def build_glide_uncertainty(
    data_dir: Path | str,
    target: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build the Glide-SP geometric uncertainty covariate."""
    _validate_target(target)
    df = load_method_prediction_table(data_dir=data_dir, method="glide-sp")
    if "glidescore_sd" not in df.columns:
        raise ValueError("glide-sp prediction table is missing glidescore_sd")
    out = (
        df[df["target"] == target]
        .loc[:, [*KEY_COLUMNS, "glidescore_sd"]]
        .rename(columns={"glidescore_sd": GLIDE_UNCERTAINTY_COLUMN})
    )
    return _select_feature_columns(
        out,
        feature_set="glide_uncertainty",
        columns=columns,
    )


def build_method_rank_summary(
    data_dir: Path | str,
    target: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build branch, consensus, and disagreement features from method ranks."""
    rank_df = build_method_ranks(data_dir=data_dir, target=target)
    out = rank_df.loc[:, KEY_COLUMNS].copy()

    branch_columns = []
    for branch_name, branch_methods in BRANCH_METHODS.items():
        branch_column = f"{branch_name}_rank_mean"
        out[branch_column] = rank_df.loc[
            :,
            [f"{method}_rank" for method in branch_methods],
        ].mean(axis=1)
        branch_columns.append(branch_column)

    out["consensus_123_rank"] = (
        2.0 * rank_df["glide-sp_rank"]
        + 2.0 * rank_df["pignet2_rank"]
        + 3.0 * rank_df["ligunity_rank"]
        + 3.0 * rank_df["boltz-2_rank"]
        + rank_df["balm_rank"]
        + rank_df["mammal_rank"]
    ) / 12.0

    rank_columns = METHOD_RANK_COLUMNS
    out["method_rank_sd"] = rank_df.loc[:, rank_columns].std(axis=1, ddof=1)
    out["method_rank_range"] = (
        rank_df.loc[:, rank_columns].max(axis=1)
        - rank_df.loc[:, rank_columns].min(axis=1)
    )
    out["branch_rank_sd"] = out.loc[:, branch_columns].std(axis=1, ddof=1)
    out["structure_minus_pose_rank"] = (
        out["structure_rank_mean"] - out["pose_rank_mean"]
    )
    out["sequence_minus_structure_rank"] = (
        out["sequence_rank_mean"] - out["structure_rank_mean"]
    )
    return _select_feature_columns(
        out.loc[:, [*KEY_COLUMNS, *METHOD_RANK_SUMMARY_COLUMNS]],
        feature_set="method_rank_summary",
        columns=columns,
    )


def build_chemical_descriptor_table(
    data_dir: Path | str,
    target: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Generate RDKit chemical descriptors from the target compound library."""
    target_library_df = build_target_library(data_dir=data_dir, target=target)
    descriptor_df = build_chemical_descriptor_features(target_library_df)
    descriptor_df.insert(1, "target", target)
    return _select_feature_columns(
        descriptor_df,
        feature_set="chemical_descriptors",
        columns=columns,
    )


def build_target_context(
    data_dir: Path | str,
    target: str,
    base_feature_set: str,
    feature_columns: list[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build target indicators and target-by-feature interactions."""
    _validate_target(target)
    _validate_feature_set(base_feature_set)
    if base_feature_set == "target_context":
        raise ValueError("target_context cannot use itself as base_feature_set")

    target_class = resolve_target_class_for_target(target)
    targets = resolve_target_class_targets(target_class)
    reference_target = resolve_target_class_reference_target(target_class)
    target_to_label = resolve_target_to_label(target_class, targets)
    base_columns = feature_columns or FEATURE_REGISTRY[base_feature_set]["default_columns"]
    base_df = build_feature_table(
        data_dir=data_dir,
        target=target,
        feature_set=base_feature_set,
        columns=base_columns,
    )

    out = base_df.loc[:, KEY_COLUMNS].copy()
    non_reference_targets = [
        class_target for class_target in targets if class_target != reference_target
    ]

    for class_target in non_reference_targets:
        target_label = target_to_label[class_target]
        indicator_column = f"is_{target_label}"
        out[indicator_column] = int(target == class_target)

    for class_target in non_reference_targets:
        target_label = target_to_label[class_target]
        indicator_column = f"is_{target_label}"
        for feature_column in base_columns:
            out[f"{feature_column}_x_{target_label}"] = (
                base_df[feature_column] * out[indicator_column]
            )

    if columns is not None:
        selected_columns = list(dict.fromkeys([*KEY_COLUMNS, *columns]))
        missing_columns = [column for column in selected_columns if column not in out.columns]
        if missing_columns:
            raise ValueError(
                f"target_context table is missing requested column(s): {missing_columns}"
            )
        return out.loc[:, selected_columns].copy()
    return out


def build_feature_table(
    data_dir: Path | str,
    target: str,
    feature_set: str,
    columns: list[str] | None = None,
    base_feature_set: str | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build one source-native feature table for one target."""
    _validate_feature_set(feature_set)
    if feature_set == "method_scores":
        return build_method_scores(data_dir=data_dir, target=target, columns=columns)
    if feature_set == "method_ranks":
        return build_method_ranks(data_dir=data_dir, target=target, columns=columns)
    if feature_set == "method_rank_summary":
        return build_method_rank_summary(data_dir=data_dir, target=target, columns=columns)
    if feature_set == "glide_uncertainty":
        return build_glide_uncertainty(data_dir=data_dir, target=target, columns=columns)
    if feature_set == "chemical_descriptors":
        return build_chemical_descriptor_table(
            data_dir=data_dir,
            target=target,
            columns=columns,
        )
    if feature_set == "target_context":
        if base_feature_set is None:
            raise ValueError("target_context requires base_feature_set")
        return build_target_context(
            data_dir=data_dir,
            target=target,
            base_feature_set=base_feature_set,
            feature_columns=feature_columns,
            columns=columns,
        )
    raise ValueError(f"Unknown feature_set: {feature_set}")
