"""Chemical-space and scaffold coverage diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flumolscreen.schema import ASSAY_DATA_REQUIRED_COLUMNS
from flumolscreen.target_registry import (
    TARGET_CLASS_REGISTRY,
    TARGET_REGISTRY,
    resolve_compound_file_for_target_class,
    resolve_target_class_targets,
)

DEFAULT_FINGERPRINT_RADIUS = 2
DEFAULT_FINGERPRINT_BITS = 2048
DEFAULT_UMAP_N_NEIGHBORS = 30
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_RANDOM_STATE = 42
DEFAULT_TOP_N_SCAFFOLDS = 20
STRAIN_DISPLAY_NAMES = {
    "ph1n1": "pH1N1",
    "h3n2": "H3N2",
    "h5n1": "H5N1",
}

COMPOUND_LIBRARY_COLUMNS = ["id", "smiles", "source"]
ASSAY_AGGREGATE_COLUMNS = [
    "id",
    "assayed_any",
    "n_assays",
    "n_assayed_targets",
    "assayed_targets",
    "assay_rounds",
    "label_pkd_mean",
    "label_pkd_max",
    "label_pkd_min",
]
CHEMICAL_SPACE_COLUMNS = [
    "target_class",
    "id",
    "smiles",
    "source",
    "assayed_any",
    "n_assays",
    "n_assayed_targets",
    "assayed_targets",
    "assay_rounds",
    "label_pkd_mean",
    "label_pkd_max",
    "label_pkd_min",
    "murcko_scaffold",
    "scaffold_type",
    "scaffold_library_count",
    "scaffold_assayed_count",
    "umap_x",
    "umap_y",
    "assay_plot_group",
]
SCAFFOLD_COVERAGE_COLUMNS = [
    "target_class",
    "murcko_scaffold",
    "scaffold_type",
    "scaffold_library_count",
    "scaffold_assayed_count",
    "scaffold_assayed_fraction",
    "n_assays",
    "assayed_targets",
]
SUMMARY_COLUMNS = [
    "target_class",
    "library_size",
    "assayed_count",
    "assay_fraction",
    "scaffold_count",
    "assayed_scaffold_count",
    "scaffold_coverage_fraction",
]


def discover_round_ids(data_dir: Path | str) -> list[str]:
    """Return round directories with assay data, sorted by directory name."""
    data_path = Path(data_dir)
    return sorted(
        path.name
        for path in data_path.glob("round_*")
        if path.is_dir() and (path / "assay_data").is_dir()
    )


def resolve_output_dir(results_dir: Path | str, round_ids: Iterable[str]) -> Path:
    """Resolve the standard chemical-space output directory."""
    resolved_round_ids = list(round_ids)
    if len(resolved_round_ids) == 1:
        output_root = Path(results_dir) / resolved_round_ids[0]
    else:
        output_root = Path(results_dir) / "all_rounds"
    return output_root / "visualizations" / "chemical_space"


def _validate_target_classes(target_classes: Iterable[str]) -> list[str]:
    resolved = list(target_classes)
    unknown = sorted(set(resolved).difference(TARGET_CLASS_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown target_class value(s): {unknown}")
    return resolved


def _resolve_round_ids(data_dir: Path | str, round_ids: Iterable[str] | None) -> list[str]:
    resolved_round_ids = (
        discover_round_ids(data_dir=data_dir) if round_ids is None else list(round_ids)
    )
    if not resolved_round_ids:
        raise ValueError("No assay round ids were provided or discovered.")

    missing_assay_dirs = [
        round_id
        for round_id in resolved_round_ids
        if not (Path(data_dir) / round_id / "assay_data").is_dir()
    ]
    if missing_assay_dirs:
        raise FileNotFoundError(
            "Assay data folder not found for round id(s): "
            f"{missing_assay_dirs}"
        )
    return resolved_round_ids


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    table_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{table_name} is missing required column(s): {missing_columns}")


def _join_sorted_unique(values: Iterable[Any]) -> str:
    cleaned_values = {
        str(value)
        for value in values
        if pd.notna(value) and str(value) != ""
    }
    return "|".join(sorted(cleaned_values))


def format_target_class_label(target_class: str) -> str:
    """Return the display label for a target class."""
    return TARGET_CLASS_REGISTRY.get(target_class, {}).get("display_name", target_class)


def format_strain_label(strain: str) -> str:
    """Return the display label for a strain token."""
    strain_key = str(strain)
    return STRAIN_DISPLAY_NAMES.get(strain_key.lower(), strain_key.upper())


def format_target_label(target: str) -> str:
    """Return the display label for a concrete target."""
    if target in TARGET_CLASS_REGISTRY:
        return format_target_class_label(target)

    if target in TARGET_REGISTRY:
        target_spec = TARGET_REGISTRY[target]
        target_class_label = format_target_class_label(target_spec["target_class"])
        strain = target_spec.get("strain")
        if strain is None:
            return target_class_label
        return f"{target_class_label} {format_strain_label(strain)}"

    if "_" in str(target):
        target_class, strain = str(target).split("_", 1)
        if target_class in TARGET_CLASS_REGISTRY:
            return f"{format_target_class_label(target_class)} {format_strain_label(strain)}"
    return str(target)


def format_target_set_label(targets: str) -> str:
    """Return display labels for a pipe-delimited target set."""
    if pd.isna(targets) or str(targets) == "":
        return ""
    return " | ".join(format_target_label(target) for target in str(targets).split("|"))


def format_assay_group_label(group: str) -> str:
    """Return a sentence-case display label for an assay plot group."""
    if group == "unassayed":
        return "Unassayed"
    if group == "multi-target":
        return "Multi-target"
    return format_target_set_label(group) or "Assayed"


def load_target_class_library(data_dir: Path | str, target_class: str) -> pd.DataFrame:
    """Load the shared compound library for one target class."""
    _validate_target_classes([target_class])
    compound_file = resolve_compound_file_for_target_class(target_class)
    path = Path(data_dir) / "shared" / "datasets" / compound_file
    if not path.exists():
        raise FileNotFoundError(f"Compound library file not found: {path}")

    library_df = pd.read_csv(path)
    _validate_required_columns(
        library_df,
        required_columns=COMPOUND_LIBRARY_COLUMNS,
        table_name=str(path),
    )
    if library_df["id"].duplicated().any():
        duplicate_ids = library_df.loc[library_df["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"{path} contains duplicate compound ids: {duplicate_ids}")

    out = library_df.loc[:, COMPOUND_LIBRARY_COLUMNS].copy()
    out.insert(0, "target_class", target_class)
    return out


def load_target_class_assays(
    data_dir: Path | str,
    round_ids: Iterable[str],
    target_class: str,
) -> pd.DataFrame:
    """Load assay rows for one target class across the requested rounds."""
    _validate_target_classes([target_class])
    target_class_targets = set(resolve_target_class_targets(target_class))
    assay_tables = []

    for round_id in round_ids:
        assay_dir = Path(data_dir) / round_id / "assay_data"
        if not assay_dir.is_dir():
            raise FileNotFoundError(f"Assay data folder not found: {assay_dir}")

        for assay_path in sorted(assay_dir.glob("*.csv")):
            assay_df = pd.read_csv(assay_path)
            _validate_required_columns(
                assay_df,
                required_columns=ASSAY_DATA_REQUIRED_COLUMNS,
                table_name=str(assay_path),
            )
            target_class_mask = assay_df["target_class"].eq(target_class)
            target_mask = assay_df["target"].isin(target_class_targets)
            target_class_df = assay_df.loc[target_class_mask | target_mask].copy()
            if target_class_df.empty:
                continue
            assay_tables.append(
                target_class_df.loc[:, ASSAY_DATA_REQUIRED_COLUMNS].assign(
                    round_id=lambda df: df["round_id"].fillna(round_id)
                )
            )

    if not assay_tables:
        return pd.DataFrame(columns=ASSAY_DATA_REQUIRED_COLUMNS)
    return pd.concat(assay_tables, ignore_index=True)


def aggregate_assay_coverage(assay_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate label-bearing assay rows to one record per compound id."""
    if assay_df.empty:
        return pd.DataFrame(columns=ASSAY_AGGREGATE_COLUMNS)

    _validate_required_columns(
        assay_df,
        required_columns=["id", "target", "round_id", "label_pkd"],
        table_name="assay_df",
    )
    assay_df = assay_df.copy()
    assay_df["label_pkd"] = pd.to_numeric(assay_df["label_pkd"], errors="coerce")

    records = []
    for compound_id, compound_assay_df in assay_df.groupby("id", sort=True):
        records.append(
            {
                "id": compound_id,
                "assayed_any": True,
                "n_assays": int(len(compound_assay_df)),
                "n_assayed_targets": int(compound_assay_df["target"].nunique()),
                "assayed_targets": _join_sorted_unique(compound_assay_df["target"]),
                "assay_rounds": _join_sorted_unique(compound_assay_df["round_id"]),
                "label_pkd_mean": compound_assay_df["label_pkd"].mean(),
                "label_pkd_max": compound_assay_df["label_pkd"].max(),
                "label_pkd_min": compound_assay_df["label_pkd"].min(),
            }
        )

    return pd.DataFrame.from_records(records, columns=ASSAY_AGGREGATE_COLUMNS)


def build_assay_coverage_table(
    library_df: pd.DataFrame,
    assay_df: pd.DataFrame,
    target_class: str,
) -> pd.DataFrame:
    """Return a library-wide compound table annotated with assay coverage."""
    _validate_required_columns(
        library_df,
        required_columns=["target_class", *COMPOUND_LIBRARY_COLUMNS],
        table_name="library_df",
    )
    if not assay_df.empty:
        missing_assay_ids = sorted(set(assay_df["id"]).difference(library_df["id"]))
        if missing_assay_ids:
            raise ValueError(
                f"Assay rows for {target_class} include ids absent from the shared "
                f"compound library: {missing_assay_ids[:10]}"
            )

    aggregate_df = aggregate_assay_coverage(assay_df=assay_df)
    out = library_df.merge(aggregate_df, on="id", how="left")
    out["assayed_any"] = out["assayed_any"].fillna(False).astype(bool)
    out["n_assays"] = out["n_assays"].fillna(0).astype(int)
    out["n_assayed_targets"] = out["n_assayed_targets"].fillna(0).astype(int)
    out["assayed_targets"] = out["assayed_targets"].fillna("")
    out["assay_rounds"] = out["assay_rounds"].fillna("")
    return out


def _load_rdkit_modules():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError as error:
        raise ImportError(
            "RDKit is required for chemical-space diagnostics. "
            "Install and activate the consensus.yml environment first."
        ) from error
    return Chem, DataStructs, rdFingerprintGenerator, MurckoScaffold


def compute_murcko_scaffold_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return Murcko scaffold annotations, preserving acyclic compounds separately."""
    _validate_required_columns(df, ["id", "smiles"], "df")
    Chem, _, _, MurckoScaffold = _load_rdkit_modules()

    records = []
    for record in df.loc[:, ["id", "smiles"]].to_dict(orient="records"):
        compound_id = record["id"]
        smiles = record["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES for id={compound_id}")

        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold_mol is None or scaffold_mol.GetNumAtoms() == 0:
            scaffold = Chem.MolToSmiles(mol, isomericSmiles=True)
            scaffold_type = "acyclic"
        else:
            scaffold = Chem.MolToSmiles(scaffold_mol, isomericSmiles=True)
            scaffold_type = "murcko"
        records.append(
            {
                "id": compound_id,
                "murcko_scaffold": scaffold,
                "scaffold_type": scaffold_type,
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=["id", "murcko_scaffold", "scaffold_type"],
    )


def compute_fingerprint_matrix(
    df: pd.DataFrame,
    radius: int = DEFAULT_FINGERPRINT_RADIUS,
    n_bits: int = DEFAULT_FINGERPRINT_BITS,
) -> np.ndarray:
    """Return a dense binary Morgan fingerprint matrix for the provided compounds."""
    _validate_required_columns(df, ["id", "smiles"], "df")
    Chem, DataStructs, rdFingerprintGenerator, _ = _load_rdkit_modules()
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius),
        fpSize=int(n_bits),
    )

    matrix = np.zeros((len(df), int(n_bits)), dtype=np.uint8)
    for row_idx, record in enumerate(df.loc[:, ["id", "smiles"]].to_dict(orient="records")):
        mol = Chem.MolFromSmiles(record["smiles"])
        if mol is None:
            raise ValueError(f"Could not parse SMILES for id={record['id']}")
        fingerprint = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fingerprint, matrix[row_idx])
    return matrix


def compute_umap_coordinates(
    fingerprint_matrix: np.ndarray,
    n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    random_state: int = DEFAULT_UMAP_RANDOM_STATE,
) -> pd.DataFrame:
    """Project binary fingerprints into two UMAP dimensions."""
    n_compounds = int(fingerprint_matrix.shape[0])
    if n_compounds == 0:
        return pd.DataFrame(columns=["umap_x", "umap_y"])
    if n_compounds == 1:
        return pd.DataFrame({"umap_x": [0.0], "umap_y": [0.0]})
    if n_compounds == 2:
        return pd.DataFrame({"umap_x": [0.0, 1.0], "umap_y": [0.0, 0.0]})

    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "umap-learn is required for chemical-space diagnostics. "
            "Install and activate the consensus.yml environment first."
        ) from error

    adjusted_neighbors = min(max(2, int(n_neighbors)), n_compounds - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=adjusted_neighbors,
        min_dist=float(min_dist),
        metric="jaccard",
        random_state=int(random_state),
    )
    embedding = reducer.fit_transform(fingerprint_matrix.astype(bool))
    return pd.DataFrame(
        {
            "umap_x": embedding[:, 0].astype(float),
            "umap_y": embedding[:, 1].astype(float),
        }
    )


def assign_assay_plot_group(df: pd.DataFrame) -> pd.Series:
    """Return plot categories for assayed and unassayed compounds."""
    groups = pd.Series("Unassayed", index=df.index, dtype="object")
    assayed_mask = df["assayed_any"].astype(bool)
    multi_target_mask = assayed_mask & (df["n_assayed_targets"] > 1)
    single_target_mask = assayed_mask & ~multi_target_mask
    groups.loc[multi_target_mask] = "Multi-target"
    groups.loc[single_target_mask] = (
        df.loc[single_target_mask, "assayed_targets"]
        .replace("", "assayed")
        .map(format_assay_group_label)
    )
    return groups


def add_chemistry_annotations(
    df: pd.DataFrame,
    fingerprint_radius: int = DEFAULT_FINGERPRINT_RADIUS,
    fingerprint_bits: int = DEFAULT_FINGERPRINT_BITS,
    umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
    umap_random_state: int = DEFAULT_UMAP_RANDOM_STATE,
) -> pd.DataFrame:
    """Add scaffold counts and UMAP coordinates to a compound coverage table."""
    out = df.copy()
    scaffold_df = compute_murcko_scaffold_table(out)
    fingerprint_matrix = compute_fingerprint_matrix(
        out,
        radius=fingerprint_radius,
        n_bits=fingerprint_bits,
    )
    embedding_df = compute_umap_coordinates(
        fingerprint_matrix=fingerprint_matrix,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=umap_random_state,
    )

    out = out.merge(scaffold_df, on="id", how="left")
    out.loc[:, ["umap_x", "umap_y"]] = embedding_df.loc[:, ["umap_x", "umap_y"]].to_numpy()

    library_counts = out["murcko_scaffold"].value_counts().rename("scaffold_library_count")
    assayed_counts = (
        out.groupby("murcko_scaffold")["assayed_any"]
        .sum()
        .astype(int)
        .rename("scaffold_assayed_count")
    )
    out = out.merge(library_counts, left_on="murcko_scaffold", right_index=True, how="left")
    out = out.merge(assayed_counts, left_on="murcko_scaffold", right_index=True, how="left")
    out["scaffold_library_count"] = out["scaffold_library_count"].astype(int)
    out["scaffold_assayed_count"] = out["scaffold_assayed_count"].fillna(0).astype(int)
    out["assay_plot_group"] = assign_assay_plot_group(out)
    return out.loc[:, CHEMICAL_SPACE_COLUMNS]


def build_target_class_chemical_space_table(
    data_dir: Path | str,
    round_ids: Iterable[str],
    target_class: str,
    fingerprint_radius: int = DEFAULT_FINGERPRINT_RADIUS,
    fingerprint_bits: int = DEFAULT_FINGERPRINT_BITS,
    umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
    umap_random_state: int = DEFAULT_UMAP_RANDOM_STATE,
) -> pd.DataFrame:
    """Build one complete chemical-space diagnostics table for a target class."""
    library_df = load_target_class_library(data_dir=data_dir, target_class=target_class)
    assay_df = load_target_class_assays(
        data_dir=data_dir,
        round_ids=round_ids,
        target_class=target_class,
    )
    coverage_df = build_assay_coverage_table(
        library_df=library_df,
        assay_df=assay_df,
        target_class=target_class,
    )
    return add_chemistry_annotations(
        coverage_df,
        fingerprint_radius=fingerprint_radius,
        fingerprint_bits=fingerprint_bits,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_random_state=umap_random_state,
    )


def build_scaffold_coverage_table(compound_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize scaffold-level assay coverage for one compound table."""
    _validate_required_columns(
        compound_df,
        required_columns=[
            "target_class",
            "murcko_scaffold",
            "scaffold_type",
            "assayed_any",
            "n_assays",
            "assayed_targets",
        ],
        table_name="compound_df",
    )

    records = []
    for (target_class, scaffold), scaffold_df in compound_df.groupby(
        ["target_class", "murcko_scaffold"],
        sort=False,
    ):
        library_count = int(len(scaffold_df))
        assayed_count = int(scaffold_df["assayed_any"].sum())
        records.append(
            {
                "target_class": target_class,
                "murcko_scaffold": scaffold,
                "scaffold_type": _join_sorted_unique(scaffold_df["scaffold_type"]),
                "scaffold_library_count": library_count,
                "scaffold_assayed_count": assayed_count,
                "scaffold_assayed_fraction": (
                    assayed_count / library_count if library_count else np.nan
                ),
                "n_assays": int(scaffold_df["n_assays"].sum()),
                "assayed_targets": _join_sorted_unique(
                    target
                    for value in scaffold_df["assayed_targets"]
                    for target in str(value).split("|")
                    if target
                ),
            }
        )

    if not records:
        return pd.DataFrame(columns=SCAFFOLD_COVERAGE_COLUMNS)
    return (
        pd.DataFrame.from_records(records, columns=SCAFFOLD_COVERAGE_COLUMNS)
        .sort_values(
            ["scaffold_assayed_count", "scaffold_library_count", "murcko_scaffold"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def summarize_target_class(
    target_class: str,
    compound_df: pd.DataFrame,
    scaffold_df: pd.DataFrame,
) -> dict[str, Any]:
    """Return one run-summary row for a target class."""
    library_size = int(len(compound_df))
    assayed_count = int(compound_df["assayed_any"].sum())
    scaffold_count = int(scaffold_df["murcko_scaffold"].nunique())
    assayed_scaffold_count = int((scaffold_df["scaffold_assayed_count"] > 0).sum())
    return {
        "target_class": target_class,
        "library_size": library_size,
        "assayed_count": assayed_count,
        "assay_fraction": assayed_count / library_size if library_size else np.nan,
        "scaffold_count": scaffold_count,
        "assayed_scaffold_count": assayed_scaffold_count,
        "scaffold_coverage_fraction": (
            assayed_scaffold_count / scaffold_count if scaffold_count else np.nan
        ),
    }


def _shorten_scaffold_labels(values: pd.Series, max_chars: int = 42) -> pd.Series:
    return values.astype(str).map(
        lambda value: value if len(value) <= max_chars else f"{value[: max_chars - 3]}..."
    )


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "matplotlib is required to write static chemical-space plots."
        ) from error
    return plt


def _add_display_assay_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["assayed_targets_display"] = out["assayed_targets"].map(format_target_set_label)
    return out


def write_chemical_space_static_plot(
    compound_df: pd.DataFrame,
    output_path: Path | str,
    target_class: str,
) -> Path:
    """Write a static chemical-space scatter plot."""
    plt = _load_pyplot()
    target_class_label = format_target_class_label(target_class)

    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    unassayed_df = compound_df[~compound_df["assayed_any"].astype(bool)]
    assayed_df = compound_df[compound_df["assayed_any"].astype(bool)]

    ax.scatter(
        unassayed_df["umap_x"],
        unassayed_df["umap_y"],
        s=9,
        c="#d0d0d0",
        alpha=0.35,
        linewidths=0,
        label="Unassayed",
    )
    palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#000000",
    ]
    for color_idx, (group_name, group_df) in enumerate(
        assayed_df.groupby("assay_plot_group", sort=True)
    ):
        ax.scatter(
            group_df["umap_x"],
            group_df["umap_y"],
            s=22,
            c=palette[color_idx % len(palette)],
            alpha=0.9,
            linewidths=0.2,
            edgecolors="white",
            label=group_name,
        )

    ax.set_title(f"{target_class_label} chemical-space assay coverage")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_chemical_space_interactive_plot(
    compound_df: pd.DataFrame,
    output_path: Path | str,
    target_class: str,
) -> Path:
    """Write an interactive chemical-space scatter plot."""
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "plotly is required to write interactive chemical-space plots."
        ) from error

    output_path = Path(output_path)
    target_class_label = format_target_class_label(target_class)
    compound_df = _add_display_assay_targets(compound_df)
    fig = go.Figure()
    hover_columns = [
        "id",
        "source",
        "murcko_scaffold",
        "assay_rounds",
        "assayed_targets_display",
        "label_pkd_mean",
        "label_pkd_max",
        "label_pkd_min",
    ]

    unassayed_df = compound_df[~compound_df["assayed_any"].astype(bool)]
    fig.add_trace(
        go.Scattergl(
            x=unassayed_df["umap_x"],
            y=unassayed_df["umap_y"],
            mode="markers",
            name="Unassayed",
            marker={"size": 4, "color": "#c8c8c8", "opacity": 0.35},
            customdata=unassayed_df.loc[:, hover_columns].fillna("").to_numpy(),
            hovertemplate=(
                "ID=%{customdata[0]}<br>"
                "Source=%{customdata[1]}<br>"
                "Scaffold=%{customdata[2]}<extra></extra>"
            ),
        )
    )
    assayed_df = compound_df[compound_df["assayed_any"].astype(bool)]
    palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#000000",
    ]
    for color_idx, (group_name, group_df) in enumerate(
        assayed_df.groupby("assay_plot_group", sort=True)
    ):
        fig.add_trace(
            go.Scattergl(
                x=group_df["umap_x"],
                y=group_df["umap_y"],
                mode="markers",
                name=group_name,
                marker={
                    "size": 7,
                    "color": palette[color_idx % len(palette)],
                    "opacity": 0.9,
                    "line": {"width": 0.5, "color": "white"},
                },
                customdata=group_df.loc[:, hover_columns].fillna("").to_numpy(),
                hovertemplate=(
                    "ID=%{customdata[0]}<br>"
                    "Source=%{customdata[1]}<br>"
                    "Scaffold=%{customdata[2]}<br>"
                    "Rounds=%{customdata[3]}<br>"
                    "Targets=%{customdata[4]}<br>"
                    "Label mean=%{customdata[5]}<br>"
                    "Label max=%{customdata[6]}<br>"
                    "Label min=%{customdata[7]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{target_class_label} chemical-space assay coverage",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_white",
        legend_title_text="Assay coverage",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    return output_path


def write_scaffold_coverage_static_plot(
    scaffold_df: pd.DataFrame,
    output_path: Path | str,
    target_class: str,
    top_n: int = DEFAULT_TOP_N_SCAFFOLDS,
) -> Path:
    """Write a static scaffold coverage bar plot."""
    plt = _load_pyplot()
    target_class_label = format_target_class_label(target_class)

    output_path = Path(output_path)
    plot_df = scaffold_df.head(int(top_n)).iloc[::-1].copy()
    labels = _shorten_scaffold_labels(plot_df["murcko_scaffold"])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(plot_df))), constrained_layout=True)
    y_positions = np.arange(len(plot_df))
    ax.barh(
        y_positions,
        plot_df["scaffold_library_count"],
        color="#d9d9d9",
        label="Library compounds",
    )
    ax.barh(
        y_positions,
        plot_df["scaffold_assayed_count"],
        color="#0072B2",
        label="Assayed compounds",
    )
    for y_pos, row in zip(y_positions, plot_df.to_dict(orient="records")):
        ax.text(
            row["scaffold_library_count"],
            y_pos,
            f"  {row['scaffold_assayed_fraction']:.0%}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Compound count")
    ax.set_title(f"{target_class_label} scaffold assay coverage")
    ax.legend(loc="best", fontsize=8, frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def write_scaffold_coverage_interactive_plot(
    scaffold_df: pd.DataFrame,
    output_path: Path | str,
    target_class: str,
    top_n: int = DEFAULT_TOP_N_SCAFFOLDS,
) -> Path:
    """Write an interactive scaffold coverage bar plot."""
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "plotly is required to write interactive scaffold coverage plots."
        ) from error

    output_path = Path(output_path)
    target_class_label = format_target_class_label(target_class)
    plot_df = scaffold_df.head(int(top_n)).iloc[::-1].copy()
    plot_df["assayed_targets_display"] = plot_df["assayed_targets"].map(
        format_target_set_label
    )
    labels = _shorten_scaffold_labels(plot_df["murcko_scaffold"])
    customdata = plot_df.loc[
        :,
        [
            "murcko_scaffold",
            "scaffold_type",
            "scaffold_library_count",
            "scaffold_assayed_count",
            "scaffold_assayed_fraction",
            "n_assays",
            "assayed_targets_display",
        ],
    ].fillna("").to_numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["scaffold_library_count"],
            y=labels,
            orientation="h",
            name="Library compounds",
            marker={"color": "#d9d9d9"},
            customdata=customdata,
            hovertemplate=(
                "Scaffold=%{customdata[0]}<br>"
                "Type=%{customdata[1]}<br>"
                "Library=%{customdata[2]}<br>"
                "Assayed=%{customdata[3]}<br>"
                "Fraction=%{customdata[4]:.1%}<br>"
                "Assays=%{customdata[5]}<br>"
                "Targets=%{customdata[6]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=plot_df["scaffold_assayed_count"],
            y=labels,
            orientation="h",
            name="Assayed compounds",
            marker={"color": "#0072B2"},
            customdata=customdata,
            hovertemplate=(
                "Scaffold=%{customdata[0]}<br>"
                "Type=%{customdata[1]}<br>"
                "Library=%{customdata[2]}<br>"
                "Assayed=%{customdata[3]}<br>"
                "Fraction=%{customdata[4]:.1%}<br>"
                "Assays=%{customdata[5]}<br>"
                "Targets=%{customdata[6]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{target_class_label} scaffold assay coverage",
        xaxis_title="Compound count",
        yaxis_title="Scaffold",
        template="plotly_white",
        barmode="overlay",
        legend_title_text="Coverage",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    return output_path


def write_target_class_outputs(
    compound_df: pd.DataFrame,
    scaffold_df: pd.DataFrame,
    output_dir: Path | str,
    target_class: str,
    top_n_scaffolds: int = DEFAULT_TOP_N_SCAFFOLDS,
    write_plots: bool = True,
) -> dict[str, Path]:
    """Write CSV and figure outputs for one target class."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "chemical_space_csv": output_dir / f"chemical_space_{target_class}.csv",
        "chemical_space_png": output_dir / f"chemical_space_{target_class}.png",
        "chemical_space_html": output_dir / f"chemical_space_{target_class}.html",
        "scaffold_coverage_csv": output_dir / f"scaffold_coverage_{target_class}.csv",
        "scaffold_coverage_png": output_dir / f"scaffold_coverage_{target_class}.png",
        "scaffold_coverage_html": output_dir / f"scaffold_coverage_{target_class}.html",
    }
    compound_df.to_csv(paths["chemical_space_csv"], index=False)
    scaffold_df.to_csv(paths["scaffold_coverage_csv"], index=False)

    if write_plots:
        write_chemical_space_static_plot(
            compound_df=compound_df,
            output_path=paths["chemical_space_png"],
            target_class=target_class,
        )
        write_chemical_space_interactive_plot(
            compound_df=compound_df,
            output_path=paths["chemical_space_html"],
            target_class=target_class,
        )
        write_scaffold_coverage_static_plot(
            scaffold_df=scaffold_df,
            output_path=paths["scaffold_coverage_png"],
            target_class=target_class,
            top_n=top_n_scaffolds,
        )
        write_scaffold_coverage_interactive_plot(
            scaffold_df=scaffold_df,
            output_path=paths["scaffold_coverage_html"],
            target_class=target_class,
            top_n=top_n_scaffolds,
        )
    return paths


def create_chemical_space_diagnostics(
    data_dir: Path | str = "data",
    results_dir: Path | str = "results",
    round_ids: Iterable[str] | None = None,
    target_classes: Iterable[str] | None = None,
    top_n_scaffolds: int = DEFAULT_TOP_N_SCAFFOLDS,
    fingerprint_radius: int = DEFAULT_FINGERPRINT_RADIUS,
    fingerprint_bits: int = DEFAULT_FINGERPRINT_BITS,
    umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS,
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
    umap_random_state: int = DEFAULT_UMAP_RANDOM_STATE,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Create chemical-space and scaffold diagnostics for target classes."""
    resolved_round_ids = _resolve_round_ids(data_dir=data_dir, round_ids=round_ids)
    resolved_target_classes = _validate_target_classes(
        target_classes or TARGET_CLASS_REGISTRY.keys()
    )
    output_dir = resolve_output_dir(
        results_dir=results_dir,
        round_ids=resolved_round_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    target_outputs: dict[str, dict[str, Path]] = {}
    summary_rows = []
    for target_class in resolved_target_classes:
        compound_df = build_target_class_chemical_space_table(
            data_dir=data_dir,
            round_ids=resolved_round_ids,
            target_class=target_class,
            fingerprint_radius=fingerprint_radius,
            fingerprint_bits=fingerprint_bits,
            umap_n_neighbors=umap_n_neighbors,
            umap_min_dist=umap_min_dist,
            umap_random_state=umap_random_state,
        )
        scaffold_df = build_scaffold_coverage_table(compound_df=compound_df)
        target_outputs[target_class] = write_target_class_outputs(
            compound_df=compound_df,
            scaffold_df=scaffold_df,
            output_dir=output_dir,
            target_class=target_class,
            top_n_scaffolds=top_n_scaffolds,
            write_plots=write_plots,
        )
        summary_rows.append(
            summarize_target_class(
                target_class=target_class,
                compound_df=compound_df,
                scaffold_df=scaffold_df,
            )
        )

    summary_df = pd.DataFrame.from_records(summary_rows, columns=SUMMARY_COLUMNS)
    summary_path = output_dir / "chemical_space_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return {
        "round_ids": resolved_round_ids,
        "target_classes": resolved_target_classes,
        "output_dir": output_dir,
        "summary_path": summary_path,
        "summary_df": summary_df,
        "target_outputs": target_outputs,
    }
