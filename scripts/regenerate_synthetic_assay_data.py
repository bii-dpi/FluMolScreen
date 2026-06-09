"""Regenerate source-native synthetic assay CSVs."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from flumolscreen.features.source_features import (
    build_method_rank_summary,
    build_target_library,
)
from flumolscreen.target_registry import TARGET_REGISTRY

DATA_DIR = REPO_ROOT / "data"
ASSAY_DIR = DATA_DIR / "round_synthetic" / "assay_data"
OUTPUT_COLUMNS = [
    "id",
    "target",
    "target_class",
    "strain",
    "smiles",
    "label_pkd",
    "label_source",
    "round_id",
]

EXISTING_LABEL_TARGETS = ["furin", "pa_ph1n1", "pa_h3n2", "pa_h5n1"]
GENERATED_LABEL_SPECS = {
    "fasn": {"template_target": "furin", "n_rows": 300, "seed": 101},
    "na_ph1n1": {"template_target": "pa_ph1n1", "n_rows": 400, "seed": 201},
    "na_h3n2": {"template_target": "pa_h3n2", "n_rows": 422, "seed": 202},
    "na_h5n1": {"template_target": "pa_h5n1", "n_rows": 414, "seed": 203},
}


def _standardize_existing_assay(target: str) -> pd.DataFrame:
    path = ASSAY_DIR / f"{target}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Existing synthetic assay file not found: {path}")

    df = pd.read_csv(path)
    rename_map = {
        "compound_id": "id",
        "target_id": "target",
        "isomeric_smiles": "smiles",
    }
    df = df.rename(columns=rename_map)
    required_columns = ["id", "label_pkd"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required column(s): {missing_columns}")

    target_library_df = build_target_library(data_dir=DATA_DIR, target=target)
    out = df.loc[:, ["id", "label_pkd"]].merge(
        target_library_df,
        on="id",
        how="left",
    )
    if out["smiles"].isna().any():
        missing_ids = out.loc[out["smiles"].isna(), "id"].head(10).tolist()
        raise ValueError(f"{path} has ids missing from compound map: {missing_ids}")

    out["target"] = target
    out["target_class"] = TARGET_REGISTRY[target]["target_class"]
    out["strain"] = TARGET_REGISTRY[target]["strain"]
    out["label_pkd"] = out["label_pkd"].round(3)
    out["label_source"] = "synthetic"
    out["round_id"] = "round_synthetic"
    return out.loc[:, OUTPUT_COLUMNS]


def _stratified_sample_by_consensus(
    df: pd.DataFrame,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["consensus_decile"] = pd.qcut(
        out["consensus_123_rank"],
        q=10,
        labels=False,
        duplicates="drop",
    )
    deciles = sorted(out["consensus_decile"].dropna().unique().tolist())
    base_count = n_rows // len(deciles)
    remainder = n_rows % len(deciles)

    sampled_indices: list[int] = []
    for decile_idx, decile in enumerate(deciles):
        group = out[out["consensus_decile"] == decile]
        take = base_count + int(decile_idx < remainder)
        sampled_indices.extend(
            group.sample(
                n=min(take, len(group)),
                random_state=int(seed + decile_idx),
            ).index.tolist()
        )

    if len(sampled_indices) < n_rows:
        remaining = out.drop(index=sampled_indices)
        fill_count = n_rows - len(sampled_indices)
        sampled_indices.extend(
            remaining.sample(
                n=fill_count,
                random_state=int(rng.integers(0, 1_000_000)),
            ).index.tolist()
        )

    sampled_df = out.loc[sampled_indices].drop(columns=["consensus_decile"])
    return sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _quantile_map_labels(
    sampled_df: pd.DataFrame,
    template_labels: pd.Series,
    seed: int,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    score = sampled_df["consensus_123_rank"].to_numpy() + rng.normal(
        loc=0.0,
        scale=0.06,
        size=len(sampled_df),
    )
    sorted_score_idx = np.argsort(score)
    sorted_labels = np.sort(template_labels.to_numpy())
    mapped = np.empty(len(sampled_df), dtype=float)
    mapped[sorted_score_idx] = sorted_labels
    return pd.Series(mapped, index=sampled_df.index).round(3)


def _generate_assay_from_predictions(
    target: str,
    template_df: pd.DataFrame,
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    rank_summary_df = build_method_rank_summary(data_dir=DATA_DIR, target=target)
    sampled_df = _stratified_sample_by_consensus(
        rank_summary_df,
        n_rows=n_rows,
        seed=seed,
    )
    target_library_df = build_target_library(data_dir=DATA_DIR, target=target)
    out = sampled_df.loc[:, ["id", "target"]].merge(
        target_library_df,
        on=["id", "target"],
        how="left",
    )
    out["label_pkd"] = _quantile_map_labels(
        sampled_df=sampled_df,
        template_labels=template_df["label_pkd"],
        seed=seed,
    )
    out["label_source"] = "synthetic"
    out["round_id"] = "round_synthetic"
    return out.loc[:, OUTPUT_COLUMNS]


def main() -> None:
    ASSAY_DIR.mkdir(parents=True, exist_ok=True)

    standardized_existing = {
        target: _standardize_existing_assay(target)
        for target in EXISTING_LABEL_TARGETS
    }

    output_tables = dict(standardized_existing)
    for target, spec in GENERATED_LABEL_SPECS.items():
        output_tables[target] = _generate_assay_from_predictions(
            target=target,
            template_df=standardized_existing[spec["template_target"]],
            n_rows=spec["n_rows"],
            seed=spec["seed"],
        )

    for target, df in output_tables.items():
        output_path = ASSAY_DIR / f"{target}.csv"
        df.to_csv(output_path, index=False)
        print(f"Wrote {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
