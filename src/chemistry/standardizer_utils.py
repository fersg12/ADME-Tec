# Chemical standardization
# Based on HarmonSmiles for molecular harmonization
# and RDKit for validation and additional curation rules.

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm.auto import tqdm

from harmonsmile import PubChemIngest, PubChemConfig, save_table

tqdm.pandas()


# ---------------------------------------------------------
# Allowed elements
# ---------------------------------------------------------

ALLOWED_ELEMENTS = {
    "H", "B", "C", "N", "O", "F",
    "Si", "P", "S", "Cl", "Se", "Br", "I"
}


# ---------------------------------------------------------
# Additional RDKit validation
# ---------------------------------------------------------

def validate_molecule(mol):
    """
    Applies additional curation rules after HarmonSmiles.

    Returns
    -------
    str or None
        Error type if the molecule fails a criterion.
    """

    if mol is None:
        return "ParsingError"

    # Check allowed elements
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}

    if not elements <= ALLOWED_ELEMENTS:
        return "DisallowedElements"

    # Check whether the molecule contains carbon
    if not any(atom.GetSymbol() == "C" for atom in mol.GetAtoms()):
        return "NotOrganic"

    return None


# ---------------------------------------------------------
# Process one molecule
# ---------------------------------------------------------

def process_molecule_row(row):
    """
    Processes one molecular entry using HarmonSmiles
    followed by additional RDKit-based validation.

    Parameters
    ----------
    row : pandas.Series
        DataFrame row containing a 'smiles' column.

    Returns
    -------
    dict
        Original molecular information plus curated_smiles
        and error information.
    """

    smiles = row["smiles"]
    result = row.to_dict()

    try:

        # -------------------------------------------------
        # 1. Parse original SMILES
        # -------------------------------------------------

        mol = Chem.MolFromSmiles(smiles, sanitize=True)

        if mol is None:
            result["error"] = "ParsingError"
            return result

        # -------------------------------------------------
        # 2. Harmonize SMILES
        # -------------------------------------------------

        # TODO:
        # Replace this section with the HarmonSmiles API
        # used by your installed version.
        #
        # harmonized_smiles = harmonize(smiles)

        # Temporary RDKit representation
        harmonized_smiles = Chem.MolToSmiles(
            mol,
            canonical=True
        )

        # -------------------------------------------------
        # 3. Reconstruct harmonized molecule
        # -------------------------------------------------

        mol = Chem.MolFromSmiles(
            harmonized_smiles,
            sanitize=True
        )

        if mol is None:
            result["error"] = "HarmonizationError"
            return result

        # -------------------------------------------------
        # 4. Additional validation
        # -------------------------------------------------

        error = validate_molecule(mol)

        if error is not None:
            result["error"] = error
            return result

        # -------------------------------------------------
        # 5. Generate final canonical SMILES
        # -------------------------------------------------

        curated_smiles = Chem.MolToSmiles(
            mol,
            canonical=True
        )

        result.update({
            "curated_smiles": curated_smiles,
            "error": None
        })

    except Exception as e:

        result["error"] = str(e)

    return result