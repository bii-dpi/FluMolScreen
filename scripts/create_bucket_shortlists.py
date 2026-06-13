"""Create bucketing inputs for selected-model shortlists.

This entrypoint currently writes only the chemistry novelty/extrapolation
artifact needed by later bucket decision rules.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from flumolscreen.target_registry import TARGET_REGISTRY
from flumolscreen.visualization import chemical_space as chemical_space_viz

DEFAULT_TANIMOTO_THRESHOLD = 0.40
DEFAULT_FINGERPRINT_RADIUS = chemical_space_viz.DEFAULT_FINGERPRINT_RADIUS
DEFAULT_FINGERPRINT_BITS = chemical_space_viz.DEFAULT_FINGERPRINT_BITS
SELECTED_INFERENCE_SUFFIX = "_selected_final_inference.csv"
ARTIFACT_SUFFIX = "_extrapolation_artifact.csv"
SUMMARY_FILE_NAME = "extrapolation_summary.csv"
KEY_COLUMNS = ["target_class", "id", "target"]

compute_murcko_scaffold_table = chemical_space_viz.compute_murcko_scaffold_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create extrapolation artifacts for future bucket shortlists.",
    )
    parser.add_argument("--data-dir", default="data", help="Input data directory.")
    parser.add_argument("--results-dir", default="results", help="Output results directory.")
    parser.add_argument(
        "--round-id",
        default="round_synthetic",
        help="Result round id containing selected-model outputs.",
    )
    parser.add_argument(
        "--assay-round-ids",
        nargs="+",
        default=None,
        help="Assay round ids to treat as labeled. Defaults to --round-id.",
    )
    parser.add_argument(
        "--dataset-labels",
        nargs="+",
        default=None,
        help="Dataset labels to process. Defaults to all selected final inference files.",
    )
    parser.add_argument(
        "--tanimoto-threshold",
        type=float,
        default=DEFAULT_TANIMOTO_THRESHOLD,
        help="Nearest-labeled Tanimoto threshold below which rows are extrapolative.",
    )
    parser.add_argument(
        "--fingerprint-radius",
        type=int,
        default=DEFAULT_FINGERPRINT_RADIUS,
        help="Morgan fingerprint radius for nearest-labeled similarity.",
    )
    parser.add_argument(
        "--fingerprint-bits",
        type=int,
        default=DEFAULT_FINGERPRINT_BITS,
        help="Morgan fingerprint bit length for nearest-labeled similarity.",
    )
    return parser


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    table_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{table_name} is missing required column(s): {missing_columns}")


def discover_dataset_labels(results_dir: Path | str, round_id: str) -> list[str]:
    """Return dataset labels with selected final inference outputs."""
    selected_dir = Path(results_dir) / round_id / "selected_models"
    if not selected_dir.is_dir():
        raise FileNotFoundError(f"Selected-model directory not found: {selected_dir}")

    labels = [
        path.name.removesuffix(SELECTED_INFERENCE_SUFFIX)
        for path in sorted(selected_dir.glob(f"*{SELECTED_INFERENCE_SUFFIX}"))
    ]
    if not labels:
        raise FileNotFoundError(
            f"No *{SELECTED_INFERENCE_SUFFIX} files found in {selected_dir}"
        )
    return labels


def selected_inference_path(
    results_dir: Path | str,
    round_id: str,
    dataset_label: str,
) -> Path:
    return (
        Path(results_dir)
        / round_id
        / "selected_models"
        / f"{dataset_label}{SELECTED_INFERENCE_SUFFIX}"
    )


def _load_selected_inference(
    results_dir: Path | str,
    round_id: str,
    dataset_label: str,
) -> pd.DataFrame:
    path = selected_inference_path(
        results_dir=results_dir,
        round_id=round_id,
        dataset_label=dataset_label,
    )
    if not path.exists():
        raise FileNotFoundError(f"Selected final inference file not found: {path}")
    df = pd.read_csv(path)
    _validate_required_columns(df, ["id", "target"], str(path))
    if df.duplicated(["id", "target"]).any():
        duplicate_keys = (
            df.loc[df.duplicated(["id", "target"]), ["id", "target"]]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(f"{path} contains duplicate id/target rows: {duplicate_keys}")
    return df


def _add_target_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical target_class and strain metadata from the registry."""
    _validate_required_columns(df, ["target"], "inference table")
    unknown_targets = sorted(set(df["target"]).difference(TARGET_REGISTRY))
    if unknown_targets:
        raise ValueError(f"Unknown target value(s): {unknown_targets}")

    target_metadata = pd.DataFrame(
        [
            {
                "target": target,
                "target_class": spec["target_class"],
                "strain": spec["strain"],
            }
            for target, spec in TARGET_REGISTRY.items()
        ]
    )
    out = df.drop(columns=[column for column in ["target_class", "strain"] if column in df.columns])
    return out.merge(target_metadata, on="target", how="left")


def _load_libraries_for_target_classes(
    data_dir: Path | str,
    target_classes: Iterable[str],
) -> pd.DataFrame:
    tables = [
        chemical_space_viz.load_target_class_library(
            data_dir=data_dir,
            target_class=target_class,
        )
        for target_class in sorted(set(target_classes))
    ]
    if not tables:
        return pd.DataFrame(columns=["target_class", "id", "smiles", "source"])
    return pd.concat(tables, ignore_index=True)


def _load_labeled_assays_for_target_classes(
    data_dir: Path | str,
    round_ids: Iterable[str],
    target_classes: Iterable[str],
) -> pd.DataFrame:
    tables = [
        chemical_space_viz.load_target_class_assays(
            data_dir=data_dir,
            round_ids=round_ids,
            target_class=target_class,
        )
        for target_class in sorted(set(target_classes))
    ]
    tables = [table for table in tables if not table.empty]
    if not tables:
        return pd.DataFrame(
            columns=["id", "target", "target_class", "strain", "smiles", "label_pkd"]
        )
    return pd.concat(tables, ignore_index=True)


def _deduplicate_labeled_compounds(labeled_df: pd.DataFrame) -> pd.DataFrame:
    if labeled_df.empty:
        return labeled_df.copy()
    _validate_required_columns(
        labeled_df,
        ["target_class", "id", "target", "smiles"],
        "labeled assay table",
    )
    return (
        labeled_df.loc[:, ["target_class", "id", "target", "smiles"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_scaffold_novelty_table(
    library_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return scaffold counts and labeled-coverage flags for library compounds."""
    _validate_required_columns(
        library_df,
        ["target_class", "id", "smiles"],
        "compound library",
    )

    scaffold_tables = []
    for target_class, class_library_df in library_df.groupby("target_class", sort=True):
        scaffold_df = compute_murcko_scaffold_table(
            class_library_df.loc[:, ["id", "smiles"]].drop_duplicates()
        )
        scaffold_df.insert(0, "target_class", target_class)
        scaffold_tables.append(scaffold_df)

    scaffold_df = pd.concat(scaffold_tables, ignore_index=True)
    library_counts = (
        scaffold_df.groupby(["target_class", "murcko_scaffold"], as_index=False)
        .size()
        .rename(columns={"size": "scaffold_library_count"})
    )

    labeled_ids = (
        labeled_df.loc[:, ["target_class", "id"]].drop_duplicates()
        if not labeled_df.empty
        else pd.DataFrame(columns=["target_class", "id"])
    )
    labeled_scaffold_df = scaffold_df.merge(
        labeled_ids,
        on=["target_class", "id"],
        how="inner",
    )
    if labeled_scaffold_df.empty:
        labeled_counts = pd.DataFrame(
            columns=["target_class", "murcko_scaffold", "scaffold_labeled_count"]
        )
    else:
        labeled_counts = (
            labeled_scaffold_df.groupby(["target_class", "murcko_scaffold"], as_index=False)
            .size()
            .rename(columns={"size": "scaffold_labeled_count"})
        )

    out = scaffold_df.merge(
        library_counts,
        on=["target_class", "murcko_scaffold"],
        how="left",
    )
    out = out.merge(
        labeled_counts,
        on=["target_class", "murcko_scaffold"],
        how="left",
    )
    out["scaffold_labeled_count"] = out["scaffold_labeled_count"].fillna(0).astype(int)
    out["scaffold_labeled_fraction"] = (
        out["scaffold_labeled_count"] / out["scaffold_library_count"]
    )
    out["unseen_scaffold_flag"] = out["scaffold_labeled_count"].eq(0).astype(int)
    return out


def _load_rdkit_similarity_modules():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:
        raise ImportError(
            "RDKit is required to compute nearest-labeled Tanimoto similarity. "
            "Install and activate the consensus.yml environment first."
        ) from error
    return Chem, DataStructs, rdFingerprintGenerator


def _fingerprint_records(
    df: pd.DataFrame,
    radius: int,
    n_bits: int,
) -> list[dict]:
    Chem, _, rdFingerprintGenerator = _load_rdkit_similarity_modules()
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
    )

    records = []
    for row in df.to_dict(orient="records"):
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise ValueError(f"Could not parse SMILES for id={row['id']}")
        records.append({**row, "fingerprint": generator.GetFingerprint(mol)})
    return records


def _nearest_record(query_fingerprint, labeled_records: list[dict]) -> dict | None:
    if not labeled_records:
        return None

    _, DataStructs, _ = _load_rdkit_similarity_modules()
    similarities = DataStructs.BulkTanimotoSimilarity(
        query_fingerprint,
        [record["fingerprint"] for record in labeled_records],
    )
    best_idx = int(np.argmax(similarities))
    best_record = labeled_records[best_idx]
    return {
        "id": best_record["id"],
        "target": best_record["target"],
        "tanimoto": float(similarities[best_idx]),
    }


def compute_nearest_labeled_similarity(
    query_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    fingerprint_radius: int = DEFAULT_FINGERPRINT_RADIUS,
    fingerprint_bits: int = DEFAULT_FINGERPRINT_BITS,
) -> pd.DataFrame:
    """Return nearest labeled compounds by target class and same target."""
    _validate_required_columns(
        query_df,
        ["target_class", "id", "target", "smiles"],
        "query table",
    )
    out = query_df.loc[:, KEY_COLUMNS].copy()
    out["nearest_labeled_id"] = pd.NA
    out["nearest_labeled_target"] = pd.NA
    out["nearest_labeled_tanimoto"] = np.nan
    out["nearest_labeled_same_target_id"] = pd.NA
    out["nearest_labeled_same_target_tanimoto"] = np.nan

    labeled_compounds = _deduplicate_labeled_compounds(labeled_df)
    if labeled_compounds.empty:
        return out

    for target_class, class_query_df in query_df.groupby("target_class", sort=True):
        class_labeled_df = labeled_compounds.loc[
            labeled_compounds["target_class"].eq(target_class)
        ].sort_values(["id", "target"])
        if class_labeled_df.empty:
            continue

        labeled_records = _fingerprint_records(
            class_labeled_df,
            radius=fingerprint_radius,
            n_bits=fingerprint_bits,
        )
        query_records = _fingerprint_records(
            class_query_df.reset_index().rename(columns={"index": "_query_index"}),
            radius=fingerprint_radius,
            n_bits=fingerprint_bits,
        )

        labeled_by_target = {
            target: [record for record in labeled_records if record["target"] == target]
            for target in class_labeled_df["target"].unique()
        }

        for query_record in query_records:
            query_index = query_record["_query_index"]
            nearest_any = _nearest_record(
                query_record["fingerprint"],
                labeled_records,
            )
            if nearest_any is not None:
                out.loc[query_index, "nearest_labeled_id"] = nearest_any["id"]
                out.loc[query_index, "nearest_labeled_target"] = nearest_any["target"]
                out.loc[query_index, "nearest_labeled_tanimoto"] = nearest_any[
                    "tanimoto"
                ]

            same_target_records = labeled_by_target.get(query_record["target"], [])
            nearest_same_target = _nearest_record(
                query_record["fingerprint"],
                same_target_records,
            )
            if nearest_same_target is not None:
                out.loc[query_index, "nearest_labeled_same_target_id"] = (
                    nearest_same_target["id"]
                )
                out.loc[query_index, "nearest_labeled_same_target_tanimoto"] = (
                    nearest_same_target["tanimoto"]
                )

    return out


def _build_extrapolation_reason(row: pd.Series) -> str:
    if bool(row["no_labeled_compounds_flag"]):
        return "no_labeled_compounds"

    reasons = []
    if bool(row["unseen_scaffold_flag"]):
        reasons.append("unseen_scaffold")
    if bool(row["low_similarity_flag"]):
        reasons.append("low_similarity")
    return "|".join(reasons)


def build_extrapolation_artifact(
    selected_inference_df: pd.DataFrame,
    library_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    fingerprint_radius: int = DEFAULT_FINGERPRINT_RADIUS,
    fingerprint_bits: int = DEFAULT_FINGERPRINT_BITS,
) -> pd.DataFrame:
    """Join selected inference rows to scaffold and nearest-labeled novelty fields."""
    selected_df = _add_target_metadata(selected_inference_df)
    library_columns = ["target_class", "id", "smiles", "source"]
    _validate_required_columns(library_df, library_columns, "compound library")
    if library_df.duplicated(["target_class", "id"]).any():
        duplicate_keys = (
            library_df.loc[
                library_df.duplicated(["target_class", "id"]),
                ["target_class", "id"],
            ]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(f"Compound library contains duplicate rows: {duplicate_keys}")

    artifact_df = selected_df.merge(
        library_df.loc[:, library_columns],
        on=["target_class", "id"],
        how="left",
    )
    missing_library = artifact_df["smiles"].isna()
    if missing_library.any():
        missing_keys = (
            artifact_df.loc[missing_library, ["target_class", "id"]]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(f"Selected inference rows are missing library metadata: {missing_keys}")

    scaffold_df = build_scaffold_novelty_table(
        library_df=library_df,
        labeled_df=labeled_df,
    )
    artifact_df = artifact_df.merge(
        scaffold_df,
        on=["target_class", "id"],
        how="left",
    )

    nearest_df = compute_nearest_labeled_similarity(
        query_df=artifact_df.loc[:, ["target_class", "id", "target", "smiles"]],
        labeled_df=labeled_df,
        fingerprint_radius=fingerprint_radius,
        fingerprint_bits=fingerprint_bits,
    )
    artifact_df = artifact_df.merge(nearest_df, on=KEY_COLUMNS, how="left")

    labeled_counts = (
        labeled_df.groupby("target_class")["id"].nunique()
        if not labeled_df.empty
        else pd.Series(dtype=int)
    )
    artifact_df["no_labeled_compounds_flag"] = (
        artifact_df["target_class"].map(labeled_counts).fillna(0).astype(int).eq(0)
    )
    artifact_df["low_similarity_flag"] = (
        artifact_df["nearest_labeled_tanimoto"].notna()
        & artifact_df["nearest_labeled_tanimoto"].lt(float(tanimoto_threshold))
    )
    for column in [
        "unseen_scaffold_flag",
        "no_labeled_compounds_flag",
        "low_similarity_flag",
    ]:
        artifact_df[column] = artifact_df[column].astype(int)

    artifact_df["extrapolation_flag"] = (
        artifact_df[
            [
                "unseen_scaffold_flag",
                "no_labeled_compounds_flag",
                "low_similarity_flag",
            ]
        ]
        .any(axis=1)
        .astype(int)
    )
    artifact_df["extrapolation_reason"] = artifact_df.apply(
        _build_extrapolation_reason,
        axis=1,
    )
    artifact_df.loc[artifact_df["extrapolation_flag"].eq(0), "extrapolation_reason"] = ""

    ordered_columns = [
        "id",
        "target",
        "target_class",
        "strain",
        "smiles",
        "source",
        *[
            column
            for column in selected_inference_df.columns
            if column not in {"id", "target", "target_class", "strain", "smiles", "source"}
        ],
        "murcko_scaffold",
        "scaffold_type",
        "scaffold_library_count",
        "scaffold_labeled_count",
        "scaffold_labeled_fraction",
        "nearest_labeled_id",
        "nearest_labeled_target",
        "nearest_labeled_tanimoto",
        "nearest_labeled_same_target_id",
        "nearest_labeled_same_target_tanimoto",
        "unseen_scaffold_flag",
        "low_similarity_flag",
        "no_labeled_compounds_flag",
        "extrapolation_flag",
        "extrapolation_reason",
    ]
    ordered_columns.extend(
        column for column in artifact_df.columns if column not in set(ordered_columns)
    )
    return artifact_df.loc[:, ordered_columns]


def summarize_extrapolation_artifact(
    dataset_label: str,
    artifact_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize extrapolation counts for one dataset label by target class."""
    records = []
    for target_class, class_df in artifact_df.groupby("target_class", sort=True):
        n_rows = int(len(class_df))
        records.append(
            {
                "dataset_label": dataset_label,
                "target_class": target_class,
                "n_rows": n_rows,
                "n_extrapolation": int(class_df["extrapolation_flag"].sum()),
                "extrapolation_fraction": (
                    float(class_df["extrapolation_flag"].mean()) if n_rows else np.nan
                ),
                "n_unseen_scaffold": int(class_df["unseen_scaffold_flag"].sum()),
                "n_low_similarity": int(class_df["low_similarity_flag"].sum()),
                "n_no_labeled_compounds": int(
                    class_df["no_labeled_compounds_flag"].sum()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# FUTURE BUCKET RULES
#
# This script currently computes only the novelty/extrapolation axis. Score and
# uncertainty are already available in selected final inference files under
# results/<round_id>/selected_models/. Branch support and disagreement can be
# rebuilt from method_ranks and method_rank_summary, registered in
# flumolscreen/feature_registry.py and computed in
# flumolscreen/features/source_features.py. Scaffold and chemical-space helpers
# live in flumolscreen/visualization/chemical_space.py, while chemical
# clustering helpers live in flumolscreen/features/chemical_clustering.py.
# Broad-PA bucket inputs can be built by aggregating selected PA inference rows
# by id across pa_ph1n1, pa_h3n2, and pa_h5n1. Calibration-control buckets can
# use score deciles already computed for selected OOF diagnostics in
# flumolscreen/ml/selection.py. Final bucket membership rules are intentionally
# left for the collaborator to implement below this comment.
# ---------------------------------------------------------------------------


def create_bucket_shortlists(
    data_dir: Path | str = "data",
    results_dir: Path | str = "results",
    round_id: str = "round_synthetic",
    assay_round_ids: list[str] | None = None,
    dataset_labels: list[str] | None = None,
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    fingerprint_radius: int = DEFAULT_FINGERPRINT_RADIUS,
    fingerprint_bits: int = DEFAULT_FINGERPRINT_BITS,
) -> dict:
    """Write extrapolation artifacts for selected-model inference outputs."""
    resolved_assay_round_ids = assay_round_ids or [round_id]
    resolved_dataset_labels = dataset_labels or discover_dataset_labels(
        results_dir=results_dir,
        round_id=round_id,
    )
    output_dir = Path(results_dir) / round_id / "buckets"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {}
    summary_tables = []

    for dataset_label in resolved_dataset_labels:
        selected_df = _load_selected_inference(
            results_dir=results_dir,
            round_id=round_id,
            dataset_label=dataset_label,
        )
        selected_with_metadata_df = _add_target_metadata(selected_df)
        target_classes = selected_with_metadata_df["target_class"].unique().tolist()
        library_df = _load_libraries_for_target_classes(
            data_dir=data_dir,
            target_classes=target_classes,
        )
        labeled_df = _load_labeled_assays_for_target_classes(
            data_dir=data_dir,
            round_ids=resolved_assay_round_ids,
            target_classes=target_classes,
        )
        artifact_df = build_extrapolation_artifact(
            selected_inference_df=selected_df,
            library_df=library_df,
            labeled_df=labeled_df,
            tanimoto_threshold=tanimoto_threshold,
            fingerprint_radius=fingerprint_radius,
            fingerprint_bits=fingerprint_bits,
        )

        artifact_path = output_dir / f"{dataset_label}{ARTIFACT_SUFFIX}"
        artifact_df.to_csv(artifact_path, index=False)
        artifact_paths[dataset_label] = artifact_path
        summary_tables.append(
            summarize_extrapolation_artifact(
                dataset_label=dataset_label,
                artifact_df=artifact_df,
            )
        )

    summary_df = (
        pd.concat(summary_tables, ignore_index=True)
        if summary_tables
        else pd.DataFrame(
            columns=[
                "dataset_label",
                "target_class",
                "n_rows",
                "n_extrapolation",
                "extrapolation_fraction",
                "n_unseen_scaffold",
                "n_low_similarity",
                "n_no_labeled_compounds",
            ]
        )
    )
    summary_path = output_dir / SUMMARY_FILE_NAME
    summary_df.to_csv(summary_path, index=False)

    return {
        "output_dir": output_dir,
        "artifact_paths": artifact_paths,
        "summary_path": summary_path,
        "summary_df": summary_df,
    }


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    outputs = create_bucket_shortlists(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        round_id=args.round_id,
        assay_round_ids=args.assay_round_ids,
        dataset_labels=args.dataset_labels,
        tanimoto_threshold=args.tanimoto_threshold,
        fingerprint_radius=args.fingerprint_radius,
        fingerprint_bits=args.fingerprint_bits,
    )

    print(f"Wrote bucketing extrapolation artifacts to: {outputs['output_dir']}")
    print(f"Wrote summary table to: {outputs['summary_path']}")
    for dataset_label, path in outputs["artifact_paths"].items():
        print(f"{dataset_label}: {path}")
    return outputs


if __name__ == "__main__":
    main()
