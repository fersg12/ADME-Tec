import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import streamlit as st
import seaborn as sns
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt 
import numpy as np


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


def generar_fps(df, smiles_col="SMILES", id_col=None, radius=2, n_bits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    ids = []

    for idx, row in df.iterrows():
        smi = row[smiles_col]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits
        )

        fps.append(fp)
        
        if id_col and id_col in df.columns:
            val = row[id_col]

            if isinstance(val, pd.Series):
                val = val.iloc[0]

            ids.append(str(val))
        else:
            ids.append(str(idx))

    return fps, ids

def construir_matriz_similitud(query_fps, ref_fps, query_ids, ref_ids):
    sim_matrix = []

    for fp_q in query_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp_q, ref_fps)
        sim_matrix.append(sims)

    df_sim = pd.DataFrame(sim_matrix, index=query_ids, columns=ref_ids)
    return df_sim


def tanimoto_distance_matrix(fps):
    n = len(fps)
    dists = []

    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    return np.array(dists)

def highlight_max_ref(row):
    max_val = row[sim_cols].max()
    return [
        "background-color: #2E7D32; color: white; font-weight: bold;"
        if (col in sim_cols and val == max_val) else ""
        for col, val in row.items()
    ]

def plot_heatmap_similitud(
    df_query, 
    df_ref, 
    smiles_col="SMILES", 
    id_col_query=None,
    id_col_ref=None
):

    # ===== fingerprints =====
    query_fps, query_ids = generar_fps(df_query, smiles_col, id_col=id_col_query)
    ref_fps, ref_ids = generar_fps(df_ref, smiles_col, id_col=id_col_ref)

    if len(query_fps) == 0 or len(ref_fps) == 0:
        st.warning("No valid molecules for heatmap")
        return

    # ===== matriz similitud =====
    df_sim = construir_matriz_similitud(query_fps, ref_fps, query_ids, ref_ids)

    # ===== mean similarity =====
    df_sim["Mean_Similarity"] = df_sim.mean(axis=1)


    sim_cols = [c for c in df_sim.columns if c != "Mean_Similarity"]

    # ===== clustering =====
    row_dist = tanimoto_distance_matrix(query_fps)
    col_dist = tanimoto_distance_matrix(ref_fps)

    row_linkage = linkage(row_dist, method="average")
    col_linkage = linkage(col_dist, method="average")
    
    
    sns.set(style="white")

     # ===== plot =====
    g = sns.clustermap(
        df_sim,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        cmap="viridis",
        vmin=0, vmax=1,
        figsize=(10, 8),
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Tanimoto similarity"}
    )

    # ===== labels =====
    g.ax_heatmap.set_xlabel("Reference compounds", fontsize=10)
    g.ax_heatmap.set_ylabel("Query compounds", fontsize=10)

    # ===== ticks =====
    g.ax_heatmap.tick_params(axis='x', labelsize=7)
    g.ax_heatmap.tick_params(axis='y', labelsize=7)

    # Optional: Rotate x-axis labels for better readability
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")

    st.pyplot(g.fig)


    return df_sim
