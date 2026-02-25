import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import streamlit as st
import matplotlib.pyplot as plt


def calcular_similitud(
    input_smiles: str,
    df_ref: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col: str | None = None,
    radius: int = 2,
    n_bits: int = 2048,
):
    """
    Compute Tanimoto similarity between an input compound and a reference dataset.

    Parameters
    ----------
    input_smiles : str
        SMILES string of the query compound.
    df_ref : pd.DataFrame
        DataFrame containing reference compounds.
    smiles_col : str
        Column name in df_ref that contains SMILES strings.
    id_col : str | None
        Optional identifier column (e.g., ChEMBL ID, ATC code).
    radius : int
        Radius for Morgan fingerprint (ECFP4 corresponds to radius=2).
    n_bits : int
        Length of the fingerprint bit vector.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by descending similarity.
    """

    # Convert query SMILES into RDKit molecule object
    mol_q = Chem.MolFromSmiles(input_smiles)
    if mol_q is None:
        raise ValueError("Invalid input SMILES")

    # Generate Morgan fingerprint (circular fingerprint) for the query molecule
    fp_q = AllChem.GetMorganFingerprintAsBitVect(
        mol_q, radius, nBits=n_bits
    )

    rows = []

    # Iterate through each reference compound
    for _, row in df_ref.iterrows():
        smi = row[smiles_col]

        # Convert reference SMILES to molecule
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # Skip invalid SMILES

        # Generate fingerprint for reference molecule
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits
        )

        # Compute Tanimoto similarity between query and reference
        sim = DataStructs.TanimotoSimilarity(fp_q, fp)

        # Store results
        rows.append(
            {
                "Reference_ID": row[id_col] if id_col else None,
                "Reference_SMILES": smi,
                "Tanimoto_Similarity": sim,
            }
        )

    # Create DataFrame, sort by similarity (highest first), reset index
    df_sim = (
        pd.DataFrame(rows)
        .sort_values("Tanimoto_Similarity", ascending=False)
        .reset_index(drop=True)
    )

    return df_sim


def visualizar_top_similares(
    input_smiles: str,
    df_sim: pd.DataFrame,
    top_n: int = 5,
    mols_per_row: int = 6,
):
    """
    Display the top-N most similar compounds in a grid format using Streamlit.
    """

    # Section title in Streamlit app
    st.markdown("### Most Similar Compounds")

    # Select top-N most similar compounds
    df_top = df_sim.head(top_n)

    mols = []
    legends = []

    # Add query molecule as the first molecule in the grid
    mol_q = Chem.MolFromSmiles(input_smiles)
    if mol_q:
        mols.append(mol_q)
        legends.append("Input molecule")

    # Add top reference molecules
    for _, row in df_top.iterrows():
        mol = Chem.MolFromSmiles(row["Reference_SMILES"])
        if mol:
            mols.append(mol)
            ref_id = row["Reference_ID"]

            # Handle case where ID may be stored as a pandas Series
            if isinstance(ref_id, pd.Series):
                ref_id = ref_id.iloc[0]

            ref_id = str(ref_id).strip()
            sim = float(row["Tanimoto_Similarity"])

            # Legend includes ID and similarity value
            legends.append(f"{ref_id} | Sim: {sim:.2f}")

    # Generate grid image of molecules
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(250, 250),
        legends=legends,
        useSVG=False,
    )

    # Display image in Streamlit
    st.image(img, use_container_width=True)


def plot_similarity_bars(df_sim, top_n=5):
    """
    Plot a horizontal bar chart showing similarity scores for the top-N compounds.
    """

    # Select and reverse top-N entries for better visual ordering
    df_top = df_sim.head(top_n).copy()
    df_top = df_top.iloc[::-1]

    clean_labels = []

    # Clean and standardize reference IDs for display
    for x in df_top["Reference_ID"]:
        if isinstance(x, pd.Series):
            x = x.iloc[0]
        x = str(x).strip().replace("\n", "").replace("\t", "")
        clean_labels.append(x)

    # Extract similarity values
    values = df_top["Tanimoto_Similarity"].astype(float).values.tolist()

    # Create large figure for publication-quality visualization
    fig, ax = plt.subplots(figsize=(22, 14))

    # Horizontal bar plot
    ax.barh(clean_labels, values)

    # Define similarity thresholds
    med_thr = 0.4
    high_thr = 0.6

    # Light shaded background regions for similarity interpretation
    ax.axvspan(0, med_thr, alpha=0.04)
    ax.axvspan(med_thr, high_thr, alpha=0.06)
    ax.axvspan(high_thr, 1, alpha=0.08)

    # Vertical lines indicating similarity thresholds
    ax.axvline(med_thr, linestyle="--", linewidth=2.2, label="Moderate similarity (≥0.4)")
    ax.axvline(high_thr, linestyle="--", linewidth=2.2, label="High similarity (≥0.6)")

    # Axis configuration
    ax.set_xlim(0, 1)
    ax.set_xlabel("Tanimoto similarity", fontsize=26, labelpad=12)
    ax.set_title("Chemical similarity to reference compounds", fontsize=34, pad=18)

    # Increase tick label size
    ax.tick_params(axis="both", labelsize=20)

    # Add similarity values next to bars
    for i, v in enumerate(values):
        ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=18)

    # Add legend
    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=18,
        title="Thresholds",
        title_fontsize=25
    )

    plt.tight_layout()

    return fig
