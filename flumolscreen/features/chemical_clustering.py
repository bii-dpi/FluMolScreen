"""Chemical clustering utilities based on Morgan fingerprints and Tanimoto distance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "build_chemical_cluster_table",
    "compute_tanimoto_distance_matrix",
    "smiles_to_morgan_fingerprints",
    "write_chemical_cluster_table",
]

COMPOUND_KEY_COLUMNS = ["compound_id"]


def _load_rdkit_modules():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:
        raise ImportError(
            "RDKit is required to generate chemical clusters. "
            "Install the repo requirements in the target environment first."
        ) from error

    return Chem, DataStructs, rdFingerprintGenerator


def _validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Raise when a dataframe is missing required columns."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Input table is missing required column(s): {missing_columns}")


def _deduplicate_compounds(
    df: pd.DataFrame,
    smiles_column: str,
) -> pd.DataFrame:
    """Return one row per compound id and ensure SMILES are not conflicting."""
    _validate_required_columns(df, [*COMPOUND_KEY_COLUMNS, smiles_column])

    compound_smiles_df = df.loc[:, [*COMPOUND_KEY_COLUMNS, smiles_column]].drop_duplicates()
    smiles_counts = compound_smiles_df.groupby("compound_id")[smiles_column].nunique()
    conflicting_compound_ids = smiles_counts[smiles_counts > 1].index.tolist()
    if conflicting_compound_ids:
        raise ValueError(
            "Each compound_id must map to exactly one SMILES string for clustering. "
            f"Found conflicts for: {conflicting_compound_ids[:10]}"
        )
    return compound_smiles_df.reset_index(drop=True)


def smiles_to_morgan_fingerprints(
    df: pd.DataFrame,
    smiles_column: str = "isomeric_smiles",
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """Build one Morgan fingerprint per unique compound."""
    Chem, _, rdFingerprintGenerator = _load_rdkit_modules()
    compound_df = _deduplicate_compounds(df=df, smiles_column=smiles_column)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
    )

    records = []
    for record in compound_df.to_dict(orient="records"):
        compound_id = record["compound_id"]
        smiles = record[smiles_column]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES for compound_id={compound_id}")
        records.append(
            {
                "compound_id": compound_id,
                smiles_column: smiles,
                "fingerprint": generator.GetFingerprint(mol),
            }
        )
    return pd.DataFrame.from_records(records)


def compute_tanimoto_distance_matrix(
    fingerprint_df: pd.DataFrame,
    fingerprint_column: str = "fingerprint",
) -> pd.DataFrame:
    """Compute the pairwise Tanimoto distance matrix between compounds."""
    _, DataStructs, _ = _load_rdkit_modules()
    _validate_required_columns(
        fingerprint_df,
        [*COMPOUND_KEY_COLUMNS, fingerprint_column],
    )

    compound_ids = fingerprint_df["compound_id"].tolist()
    fingerprints = fingerprint_df[fingerprint_column].tolist()
    n_compounds = len(fingerprints)
    distance_matrix = np.zeros((n_compounds, n_compounds), dtype=float)

    for idx, fingerprint in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint,
            fingerprints[idx:],
        )
        for offset, similarity in enumerate(similarities):
            jdx = idx + offset
            distance = 1.0 - float(similarity)
            distance_matrix[idx, jdx] = distance
            distance_matrix[jdx, idx] = distance

    return pd.DataFrame(distance_matrix, index=compound_ids, columns=compound_ids)


def _initialize_kmedoids(
    distance_matrix: np.ndarray,
    n_clusters: int,
) -> list[int]:
    """Choose initial medoids greedily by maximizing distance from existing medoids."""
    n_samples = distance_matrix.shape[0]
    if not 1 <= n_clusters <= n_samples:
        raise ValueError("n_clusters must be between 1 and the number of compounds")

    medoids = [int(np.argmin(distance_matrix.sum(axis=1)))]
    while len(medoids) < n_clusters:
        min_distances = distance_matrix[:, medoids].min(axis=1)
        min_distances[medoids] = -np.inf
        medoids.append(int(np.argmax(min_distances)))
    return medoids


def _assign_points_to_medoids(
    distance_matrix: np.ndarray,
    medoids: list[int],
) -> np.ndarray:
    """Assign each sample to its nearest medoid."""
    return distance_matrix[:, medoids].argmin(axis=1)


def _compute_cluster_cost(
    distance_matrix: np.ndarray,
    medoids: list[int],
    assignments: np.ndarray,
) -> float:
    """Return the total within-cluster distance to medoids."""
    medoid_indices = np.array(medoids, dtype=int)[assignments]
    row_indices = np.arange(len(assignments))
    return float(distance_matrix[row_indices, medoid_indices].sum())


def _update_medoids(
    distance_matrix: np.ndarray,
    medoids: list[int],
    assignments: np.ndarray,
) -> list[int]:
    """Update each medoid to the best in-cluster representative."""
    updated_medoids: list[int] = []
    for cluster_idx, current_medoid in enumerate(medoids):
        cluster_member_indices = np.flatnonzero(assignments == cluster_idx)
        if cluster_member_indices.size == 0:
            updated_medoids.append(current_medoid)
            continue
        within_cluster = distance_matrix[
            np.ix_(cluster_member_indices, cluster_member_indices)
        ]
        best_member_offset = int(np.argmin(within_cluster.sum(axis=1)))
        updated_medoids.append(int(cluster_member_indices[best_member_offset]))
    return updated_medoids


def _run_kmedoids(
    distance_matrix: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
) -> tuple[list[int], np.ndarray]:
    """Run a simple deterministic k-medoids procedure on a precomputed distance matrix."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    medoids = _initialize_kmedoids(distance_matrix=distance_matrix, n_clusters=n_clusters)
    assignments = _assign_points_to_medoids(distance_matrix=distance_matrix, medoids=medoids)
    current_cost = _compute_cluster_cost(
        distance_matrix=distance_matrix,
        medoids=medoids,
        assignments=assignments,
    )

    for _ in range(max_iter):
        candidate_medoids = _update_medoids(
            distance_matrix=distance_matrix,
            medoids=medoids,
            assignments=assignments,
        )
        candidate_assignments = _assign_points_to_medoids(
            distance_matrix=distance_matrix,
            medoids=candidate_medoids,
        )
        candidate_cost = _compute_cluster_cost(
            distance_matrix=distance_matrix,
            medoids=candidate_medoids,
            assignments=candidate_assignments,
        )
        if candidate_medoids == medoids or candidate_cost >= current_cost:
            break
        medoids = candidate_medoids
        assignments = candidate_assignments
        current_cost = candidate_cost

    return medoids, assignments


def build_chemical_cluster_table(
    df: pd.DataFrame,
    n_clusters: int,
    smiles_column: str = "isomeric_smiles",
    radius: int = 2,
    n_bits: int = 2048,
    max_iter: int = 100,
) -> pd.DataFrame:
    """Cluster unique compounds with k-medoids using Tanimoto distance on Morgan fingerprints."""
    fingerprint_df = smiles_to_morgan_fingerprints(
        df=df,
        smiles_column=smiles_column,
        radius=radius,
        n_bits=n_bits,
    )
    distance_df = compute_tanimoto_distance_matrix(fingerprint_df=fingerprint_df)
    _, assignments = _run_kmedoids(
        distance_matrix=distance_df.to_numpy(),
        n_clusters=n_clusters,
        max_iter=max_iter,
    )

    cluster_df = fingerprint_df.loc[:, COMPOUND_KEY_COLUMNS].copy()
    cluster_df["chem_cluster_id"] = assignments.astype(int)
    cluster_sizes = cluster_df["chem_cluster_id"].value_counts().rename("chem_cluster_size")
    cluster_df = cluster_df.merge(
        cluster_sizes,
        left_on="chem_cluster_id",
        right_index=True,
        how="left",
    )
    return cluster_df.sort_values(["chem_cluster_id", "compound_id"]).reset_index(drop=True)


def write_chemical_cluster_table(
    input_path: Path | str,
    output_path: Path | str,
    n_clusters: int,
    smiles_column: str = "isomeric_smiles",
    radius: int = 2,
    n_bits: int = 2048,
    max_iter: int = 100,
) -> Path:
    """Read a CSV, assign chemical cluster ids, and write the cluster table."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    input_df = pd.read_csv(input_path)
    cluster_df = build_chemical_cluster_table(
        df=input_df,
        n_clusters=n_clusters,
        smiles_column=smiles_column,
        radius=radius,
        n_bits=n_bits,
        max_iter=max_iter,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(output_path, index=False)
    return output_path
