import os
import pandas as pd

# Path to the local DrugBank-derived CSV file containing ADME data
DATA_PATH = os.path.join(os.path.dirname(__file__), "adme_ApprovedDrugs.csv")

# Delimiter used inside multi-value DrugBank cells
DRUGBANK_DELIMITER = ";"


def get_drugbank_by_atc(atc_code: str) -> pd.DataFrame:
    """
    Filter DrugBank compounds by ATC code or ATC name.

    This function searches for matches in multiple possible columns:
    - 'atc'
    - 'atc_name1', 'atc_name2', 'atc_name3', 'atc_name4'

    Parameters
    ----------
    atc_code : str
        ATC code or partial name to search for (case-insensitive).

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame containing compounds associated with the
        provided ATC code or name. Returns an empty DataFrame if no
        matches are found.

    Raises
    ------
    FileNotFoundError
        If the local ApprovedDrugs CSV file is not found.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Please upload ApprovedDrugs.csv first."
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Normalize search term
    atc_code = str(atc_code).strip().lower()

    # --- Search in 'atc' column ---
    if "atc" in df.columns:
        mask_atc = df["atc"].astype(str).str.lower().str.contains(atc_code, na=False)
        filtered = df[mask_atc]

        if not filtered.empty:
            return filtered.reset_index(drop=True)

    # --- Search in ATC name columns ---
    atc_name_cols = [c for c in df.columns if c.lower().startswith("atc_name")]

    if atc_name_cols:
        mask_names = pd.Series(False, index=df.index)

        for col in atc_name_cols:
            mask_names |= df[col].astype(str).str.lower().str.contains(atc_code, na=False)

        filtered = df[mask_names]

        if not filtered.empty:
            return filtered.reset_index(drop=True)

    # No matches found
    print(f" No compounds found for ATC '{atc_code}'")
    return pd.DataFrame()


def get_atc_adme_properties(atc_code: str) -> pd.DataFrame:
    """
    Retrieve compounds associated with an ATC code and prepare ADME properties.

    This function:
    1. Calls `get_drugbank_by_atc` to obtain the subset of compounds.
    2. Verifies the presence of a SMILES column required for ADME analysis.

    Parameters
    ----------
    atc_code : str
        ATC code or partial ATC name used to retrieve compounds.

    Returns
    -------
    pd.DataFrame
        DataFrame containing compounds linked to the ATC code.
        Returns an empty DataFrame if no compounds are found.

    Raises
    ------
    ValueError
        If the SMILES column is missing from the dataset.
    """
    df_atc = get_drugbank_by_atc(atc_code)

    # Return empty DataFrame if no matches
    if df_atc.empty:
        return pd.DataFrame()

    # Validate presence of SMILES column (case-insensitive check)
    if "SMILES" not in [c.upper() for c in df_atc.columns]:
        raise ValueError("'smiles' column not found in ApprovedDrugs.csv")

    return df_atc
