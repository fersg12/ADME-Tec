#chemical standardization
"Based on code from: [DIFACQUIM GitHub] (https://github.com/DIFACQUIM/Cursos/blob/main/5_3_Curado_de_bases_de_datos.ipynb) and [Oxford Protein Informatics Group](https://www.blopig.com/blog/2024/09/out-of-the-box-rdkit-valid-is-an-imperfect-metric-a-review-of-the-kekulizeexception-and-nitrogen-protonation-to-correct-this/).<br>"
### Import libraries
#from joblib import Parallel, delayed
#from rdkit.Chem.rdmolops import GetFormalCharge, RemoveStereochemistry
from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from molvs.standardize import Standardizer
from molvs.charge import Uncharger, Reionizer
from molvs.fragment import LargestFragmentChooser
from molvs.tautomer import TautomerCanonicalizer
from molecular_rectifier import Rectifier
from tqdm.auto import tqdm
tqdm.pandas()


# --- Standardization and cleaning tools initialization ---
STD = Standardizer()
LFC = LargestFragmentChooser()
UC = Uncharger()
RI = Reionizer()
TC = TautomerCanonicalizer()


def process_molecule_row(row):
    """
    Processes a molecular entry from a DataFrame row.
    - Parses and sanitizes the SMILES string.
    - Rectifies valence issues.
    - Removes forbidden elements.
    - Verifies if the molecule is organic (at least one carbon atom).
    - Neutralizes, ionizes, and standardizes the molecule.

    Parameters:
    row (pd.Series): A row from a DataFrame containing molecular data in the 'smiles' column.

    Returns:
    dict: A dictionary containing the processed molecular data, including curated SMILES, and any errors encountered during processing.
    """
    smiles = row["smiles"]
    result = row.to_dict()

    try:
        # Check validity of SMILES
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            result.update({"error": "ParsingError"})
            return result

        # Rectify valence issues
        rectified = Rectifier(mol, valence_correction="charge")
        rectified.fix_issues()
        mol = rectified.mol

        # Full sanitization and hydrogen removal
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        mol = Chem.RemoveHs(mol)

        # Standardize (instead of using rdMolStandardize)
        mol = STD.standardize(mol)

        # Keep only the largest fragment
        mol = LFC(mol)

        # Neutralize and ionize
        mol = UC(mol)
        mol = RI(mol)
        mol = TC(mol)

        # Allowed elements verification -- pasar a json
        allowed_elements = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
        if not {atom.GetSymbol() for atom in mol.GetAtoms()} <= allowed_elements:
            result.update({"error": "DisallowedElements"})
            return result

        # Organic molecule verification
        if sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C") < 1:
            result.update({"error": "NotOrganic"})
            return result

        curated_smiles = Chem.MolToSmiles(mol, canonical=True)

        result.update({
            "curated_smiles": curated_smiles,
            "error": None
        })

    except Exception as e:
        result.update({"error": str(e)})

    return result


# Process the DataFrame parallelly through the rows
#def process_dataframe(df, n_jobs = -1):
  #  results = Parallel(n_jobs = n_jobs)(delayed(process_molecule_row)(row) for _, row in df.iterrows())
    
   # return pd.DataFrame(results)