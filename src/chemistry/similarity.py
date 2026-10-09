import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
import streamlit as st
import seaborn as sns
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt 
import numpy as np
from molraptor import MorganFingerprintProfile, encode_fingerprints, FINGERPRINT_TYPES



# ============================================================
# MOLRAPTOR DESCRIPTOR CONFIGURATION
# ============================================================

import pandas as pd

DESCRIPTOR_LABELS = {
    "morgan": "Morgan",
    "featmorgan": "FeatMorgan",
    "atompair": "Atom Pair",
    "rdk": "RDKit",
    "torsion": "Topological Torsion",
    "layered": "Layered",
    "maccs": "MACCS",
}

def _normalize_descriptor(value):
    """Normaliza nombres para comparar etiquetas y tipos de MolRaptor."""
    return "".join(
        char for char in str(value).lower()
        if char.isalnum()
    )

DESCRIPTOR_OPTIONS = {
    DESCRIPTOR_LABELS.get(
        _normalize_descriptor(fp_type), str(fp_type)
    ): fp_type
    for fp_type in FINGERPRINT_TYPES
}

def _resolver_fingerprint(descriptor="Morgan", radius=2, n_bits=2048):
    """Acepta tanto la etiqueta visible como el tipo interno de MolRaptor."""

    # Si recibe la etiqueta visible, obtiene el tipo interno.
    if descriptor in DESCRIPTOR_OPTIONS:
        fingerprint_type = DESCRIPTOR_OPTIONS[descriptor]

    # Si recibe directamente el tipo interno, también lo acepta.
    elif descriptor in FINGERPRINT_TYPES:
        fingerprint_type = descriptor

    else:
        # Último intento: comparar nombres normalizados.
        normalized = _normalize_descriptor(descriptor)
        matches = [
            fp_type for fp_type in FINGERPRINT_TYPES
            if _normalize_descriptor(fp_type) == normalized
        ]

        if not matches:
            raise ValueError(
                f"Fingerprint desconocido: {descriptor}. "
                f"Opciones disponibles: {list(DESCRIPTOR_OPTIONS.keys())}"
            )

        fingerprint_type = matches[0]

    profile = None

    if _normalize_descriptor(fingerprint_type) == "morgan":
        profile = MorganFingerprintProfile(
            radius=radius,
            fp_size=n_bits,
            include_chirality=False,
        )

    return fingerprint_type, profile

def calcular_similitud(
    input_smiles: str,
    df_ref: pd.DataFrame,
    smiles_col: str = "SMILES",
    id_col: str | None = None,
    descriptor: str = "Morgan",
    radius: int = 2,
    n_bits: int = 2048,
):
    """
    Compute Tanimoto similarity between an input compound and a
    reference dataset using a selected MolRaptor fingerprint type.
    """

    # ---------------------------------------------------------
    # 1. Resolve descriptor
    # ---------------------------------------------------------

    if descriptor not in DESCRIPTOR_OPTIONS:
        raise ValueError(
            f"Unknown descriptor '{descriptor}'. "
            f"Available descriptors: "
            f"{list(DESCRIPTOR_OPTIONS.keys())}"
        )

    
    fingerprint_type, profile = _resolver_fingerprint(
        descriptor=descriptor,
        radius=radius,
        n_bits=n_bits,
    )

    if fingerprint_type not in FINGERPRINT_TYPES:
        raise ValueError(
            f"Unsupported MolRaptor fingerprint type: "
            f"{fingerprint_type}"
        )

    # ---------------------------------------------------------
    # 2. Build fingerprint parameters
    # ---------------------------------------------------------

    # MolRaptor v0.4.0 currently allows custom profile
    # parameters only for Morgan.
    profile = None

    if fingerprint_type == "morgan":
        profile = MorganFingerprintProfile(
            radius=radius,
            fp_size=n_bits,
            include_chirality=False,
        )

    # ---------------------------------------------------------
    # 3. Encode query molecule
    # ---------------------------------------------------------

    query_result = encode_fingerprints(
        [input_smiles],
        profile=profile,
        fingerprint_type=fingerprint_type,
    )

    if query_result.valid_count == 0:
        raise ValueError("Invalid input SMILES")

    fp_q = query_result.fingerprints[0]

    # ---------------------------------------------------------
    # 4. Prepare reference SMILES
    # ---------------------------------------------------------

    ref_smiles = (
        df_ref[smiles_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # ---------------------------------------------------------
    # 5. Encode reference molecules
    # ---------------------------------------------------------

    ref_result = encode_fingerprints(
        ref_smiles,
        profile=profile,
        fingerprint_type=fingerprint_type,
    )

    fingerprints = ref_result.fingerprints

    valid_indices = np.asarray(
        ref_result.valid_indices,
        dtype=int,
    )

    # ---------------------------------------------------------
    # 6. Calculate Tanimoto similarity
    # ---------------------------------------------------------

    intersection = fingerprints @ fp_q

    fp_counts = fingerprints.sum(axis=1)
    query_count = fp_q.sum()

    union = (
        fp_counts
        + query_count
        - intersection
    )

    similarities = np.divide(
        intersection,
        union,
        out=np.zeros_like(
            intersection,
            dtype=float,
        ),
        where=union != 0,
    )

    # ---------------------------------------------------------
    # 7. Build results
    # ---------------------------------------------------------

    rows = []

    for i, original_idx in enumerate(valid_indices):

        row = df_ref.iloc[original_idx]

        rows.append(
            {
                "Reference_ID":
                    row[id_col] if id_col else None,

                "Reference_SMILES":
                    row[smiles_col],

                "Tanimoto_Similarity":
                    similarities[i],
            }
        )

    # ---------------------------------------------------------
    # 8. Sort by similarity
    # ---------------------------------------------------------

    result_df = pd.DataFrame(
        rows,
        columns=["Reference_ID", "Reference_SMILES", "Tanimoto_Similarity"],
    )
    return result_df.sort_values(
        "Tanimoto_Similarity", ascending=False
    ).reset_index(drop=True)


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


def _resolver_fingerprint(descriptor="Morgan", radius=2, n_bits=2048):
    """Resuelve el tipo de fingerprint y sus parámetros compatibles con MolRaptor."""
    if descriptor not in DESCRIPTOR_OPTIONS:
        raise ValueError(
            f"Fingerprint desconocido: {descriptor}. "
            f"Opciones disponibles: {list(DESCRIPTOR_OPTIONS.keys())}"
        )

    fingerprint_type = DESCRIPTOR_OPTIONS[descriptor]
    profile = None
    if fingerprint_type == "morgan":
        profile = MorganFingerprintProfile(
            radius=radius,
            fp_size=n_bits,
            include_chirality=False,
        )
    return fingerprint_type, profile


def _codificar_fingerprints(df, smiles_col="SMILES", descriptor="Morgan",
                            radius=2, n_bits=2048, id_col=None):
    """Codifica moléculas con el fingerprint elegido y conserva sus IDs válidos."""
    fingerprint_type, profile = _resolver_fingerprint(descriptor, radius, n_bits)
    smiles = df[smiles_col].fillna("").astype(str).tolist()
    result = encode_fingerprints(
        smiles,
        profile=profile,
        fingerprint_type=fingerprint_type,
    )

    valid_indices = np.asarray(result.valid_indices, dtype=int)
    fps = result.fingerprints
    if hasattr(fps, "toarray"):
        fps = fps.toarray()
    fps = np.asarray(fps)

    if id_col and id_col in df.columns:
        ids_df = df.iloc[valid_indices][id_col]

        if isinstance(ids_df, pd.DataFrame):
            ids = ids_df.iloc[:, 0].astype(str).tolist()
        else:
            ids = ids_df.astype(str).tolist()
    else:
        ids = df.index.to_numpy()[valid_indices].astype(str).tolist()

    return fps, ids, valid_indices


def generar_fps(df, smiles_col="SMILES", id_col=None, radius=2, n_bits=2048,
                descriptor="Morgan"):
    """Genera fingerprints del tipo seleccionado (no fuerza Morgan)."""
    fps, ids, _ = _codificar_fingerprints(
        df, smiles_col=smiles_col, descriptor=descriptor,
        radius=radius, n_bits=n_bits, id_col=id_col,
    )
    return fps, ids


def construir_matriz_similitud(query_fps, ref_fps, query_ids, ref_ids):
    """Calcula similitud de Tanimoto entre matrices binarias de fingerprints."""
    query_fps = np.asarray(query_fps, dtype=bool)
    ref_fps = np.asarray(ref_fps, dtype=bool)
    intersection = query_fps.astype(np.int64) @ ref_fps.astype(np.int64).T
    q_counts = query_fps.sum(axis=1)[:, None]
    r_counts = ref_fps.sum(axis=1)[None, :]
    union = q_counts + r_counts - intersection
    similarities = np.divide(
        intersection, union,
        out=np.zeros_like(intersection, dtype=float),
        where=union != 0,
    )
    return pd.DataFrame(similarities, index=query_ids, columns=ref_ids)


def tanimoto_distance_matrix(fps):
    """Devuelve distancias condensadas 1 - Tanimoto para clustering jerárquico."""
    fps = np.asarray(fps, dtype=bool)
    n = len(fps)
    dists = []
    for i in range(1, n):
        intersection = np.logical_and(fps[:i], fps[i]).sum(axis=1)
        union = np.logical_or(fps[:i], fps[i]).sum(axis=1)
        sims = np.divide(
            intersection, union,
            out=np.zeros(i, dtype=float),
            where=union != 0,
        )
        dists.extend(1.0 - sims)
    return np.asarray(dists, dtype=float)


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
    id_col_ref=None,
    descriptor="Morgan",
    radius=2,
    n_bits=2048,
):
    """Dibuja un mapa de calor usando el fingerprint seleccionado."""
    query_fps, query_ids, _ = _codificar_fingerprints(
        df_query, smiles_col=smiles_col, id_col=id_col_query,
        descriptor=descriptor, radius=radius, n_bits=n_bits,
    )
    ref_fps, ref_ids, _ = _codificar_fingerprints(
        df_ref, smiles_col=smiles_col, id_col=id_col_ref,
        descriptor=descriptor, radius=radius, n_bits=n_bits,
    )

    if len(query_fps) == 0 or len(ref_fps) == 0:
        st.warning("No valid molecules for heatmap")
        return None

    df_sim = construir_matriz_similitud(query_fps, ref_fps, query_ids, ref_ids)
    df_sim["Mean_Similarity"] = df_sim.mean(axis=1)
    sim_cols = [c for c in df_sim.columns if c != "Mean_Similarity"]

    # Se necesita más de una molécula para calcular el clustering de cada eje.
    row_linkage = linkage(tanimoto_distance_matrix(query_fps), method="average") if len(query_fps) > 1 else None
    col_linkage = linkage(tanimoto_distance_matrix(ref_fps), method="average") if len(ref_fps) > 1 else None

    sns.set(style="white")
    # La columna Mean_Similarity es un resumen, no una molécula de referencia;
    # se excluye del clustering y de la matriz visualizada.
    df_plot = df_sim[sim_cols]
    g = sns.clustermap(
        df_plot,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        cmap="viridis",
        vmin=0, vmax=1,
        figsize=(10, 10),
        xticklabels=True,
        yticklabels=True,
        cbar_pos=(0.99, 0.2, 0.015, 0.6),
        cbar_kws={"label": f"Tanimoto similarity ({descriptor})"},
    )
    g.ax_heatmap.set_xlabel("Reference compounds", fontsize=10)
    g.ax_heatmap.set_ylabel("Query compounds", fontsize=10)
    g.ax_heatmap.tick_params(axis="x", labelsize=6)
    g.ax_heatmap.tick_params(axis="y", labelsize=6)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(g.fig)
    return df_sim
