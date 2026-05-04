"""Chemical descriptor extraction from SMILES strings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DESCRIPTOR_COLUMNS = [
    "exact_mol_wt",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "heavy_atom_count",
    "fraction_csp3",
    "qed",
]


def _load_rdkit_modules():
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    except ImportError as error:
        raise ImportError(
            "RDKit is required to generate chemical descriptors. "
            "Install the repo requirements in the target environment first."
        ) from error

    return Chem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


def _build_descriptor_record(mol, compound_id, descriptor_modules) -> dict:
    _, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors = descriptor_modules

    return {
        "compound_id": compound_id,
        "exact_mol_wt": Descriptors.ExactMolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "ring_count": Lipinski.RingCount(mol),
        "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atom_count": Lipinski.HeavyAtomCount(mol),
        "fraction_csp3": Lipinski.FractionCSP3(mol),
        "qed": QED.qed(mol),
    }


def build_chemical_descriptor_features(
    df: pd.DataFrame,
    smiles_column: str = "isomeric_smiles",
) -> pd.DataFrame:
    required_columns = ["compound_id", smiles_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Input table is missing required column(s): {missing_columns}"
        )

    descriptor_modules = _load_rdkit_modules()
    Chem = descriptor_modules[0]

    records = []
    input_records = df.loc[:, ["compound_id", smiles_column]].to_dict(orient="records")

    for record in input_records:
        compound_id = record["compound_id"]
        smiles = record[smiles_column]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES for compound_id={compound_id}")

        records.append(
            _build_descriptor_record(
                mol=mol,
                compound_id=compound_id,
                descriptor_modules=descriptor_modules,
            )
        )

    return pd.DataFrame.from_records(records, columns=["compound_id", *DESCRIPTOR_COLUMNS])


def write_chemical_descriptor_features(
    input_path: Path | str,
    output_path: Path | str,
    smiles_column: str = "isomeric_smiles",
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    input_df = pd.read_csv(input_path)
    descriptor_df = build_chemical_descriptor_features(
        input_df,
        smiles_column=smiles_column,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_df.to_csv(output_path, index=False)
    return output_path
