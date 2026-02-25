import time
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import streamlit as st


GLORYX_BASE_URL = "https://nerdd.univie.ac.at/api/gloryx"
JOBS_URL = "https://nerdd.univie.ac.at/api/jobs"


def calcular_metabolitos(
    smiles: str,
    phase: str = "phase_1_and_2",
    poll_interval: int = 5,
    timeout: int = 300,
):
    """
    Query the GLORYx API to predict metabolites for a given compound.

    This function:
    1. Submits a prediction job to the GLORYx REST API.
    2. Polls the server until the job is completed or times out.
    3. Retrieves and processes predicted metabolites.
    4. Returns RDKit molecule objects, labels, and a structured DataFrame.

    Parameters
    ----------
    smiles : str
        SMILES string of the input compound.

    phase : str, optional (default="phase_1_and_2")
        Metabolic phase to predict.
        Options:
        - "phase_1"
        - "phase_2"
        - "phase_1_and_2"

    poll_interval : int, optional (default=5)
        Number of seconds between job status checks.

    timeout : int, optional (default=300)
        Maximum waiting time in seconds before raising a TimeoutError.

    Returns
    -------
    mols : list[rdkit.Chem.Mol]
        List of RDKit molecule objects for valid predicted metabolites.

    labels : list[str]
        List of labels formatted as:
        "<Reaction_Type>\nScore: <Priority_Score>"

    df_met : pandas.DataFrame
        DataFrame containing predicted metabolite information with columns:
        - Input_SMILES
        - Metabolite_SMILES
        - Reaction_Type
        - Priority_Score

    Raises
    ------
    requests.HTTPError
        If the API request fails.

    TimeoutError
        If the prediction job exceeds the allowed waiting time.
    """

    # -------------------------
    # Create GLORYx job
    # -------------------------
    resp = requests.post(
        f"{GLORYX_BASE_URL}/jobs",
        data={
            "inputs": [smiles],
            "metabolism_phase": phase,
        },
        timeout=30,
    )
    resp.raise_for_status()

    job_id = resp.json()["id"]

    # -------------------------
    # Wait for completion
    # -------------------------
    elapsed = 0
    while elapsed < timeout:
        status = requests.get(
            f"{JOBS_URL}/{job_id}", timeout=30
        ).json()

        if status.get("status") == "completed":
            break

        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        raise TimeoutError("GLORYx job timed out")

    # -------------------------
    # Retrieve results
    # -------------------------
    results_resp = requests.get(
        f"{JOBS_URL}/{job_id}/results", timeout=30
    )
    results_resp.raise_for_status()

    results = results_resp.json().get("data", [])

    if not results:
        return [], [], pd.DataFrame()

    # -------------------------
    # Process results
    # -------------------------
    rows, mols, labels = [], [], []

    # Sort metabolites by priority score (descending)
    results_sorted = sorted(
        results,
        key=lambda x: x.get("priority_score", 0.0),
        reverse=True,
    )

    for r in results_sorted:
        smi_met = r.get("metabolite_smiles")
        reaction = r.get("reaction_type", "NA")
        score = float(r.get("priority_score", 0.0))

        mol = Chem.MolFromSmiles(smi_met)
        if mol is None:
            continue

        mols.append(mol)
        labels.append(f"{reaction}\nScore: {score:.2f}")

        rows.append(
            {
                "Input_SMILES": smiles,
                "Metabolite_SMILES": smi_met,
                "Reaction_Type": reaction,
                "Priority_Score": score,
            }
        )

    df_met = pd.DataFrame(rows)

    return mols, labels, df_met


def visualizar_metabolitos(
    df_met,
    default_score_thr: float = 0.5,
    mols_per_row: int = 4,
    img_size: tuple = (250, 250),
):
    """
    Visualize predicted metabolites in a Streamlit interface.

    This function:
    1. Displays a priority score threshold slider.
    2. Filters metabolites based on the selected score.
    3. Groups metabolites by reaction type.
    4. Displays RDKit grid images sorted by priority score.
    5. Allows users to download the complete metabolite table.

    Parameters
    ----------
    df_met : pandas.DataFrame
        DataFrame containing metabolite predictions.
        Required columns:
        - Input_SMILES
        - Metabolite_SMILES
        - Reaction_Type
        - Priority_Score

    default_score_thr : float, optional (default=0.5)
        Initial threshold value for filtering metabolites
        based on Priority_Score.

    mols_per_row : int, optional (default=4)
        Number of molecules displayed per row in the grid.

    img_size : tuple[int, int], optional (default=(250, 250))
        Size (width, height) of each molecule image in pixels.

    Returns
    -------
    None
        Displays interactive components directly in Streamlit.
    """

    if df_met is None or df_met.empty:
        return

    st.markdown("### Predicted Metabolites (GLORYx)")

    # -----------------------------
    # 1️Score threshold selector
    # -----------------------------
    score_thr = st.slider(
        "Minimum Priority Score",
        min_value=0.0,
        max_value=1.0,
        value=default_score_thr,
        step=0.05,
    )

    df_filt = df_met[df_met["Priority_Score"] >= score_thr].copy()

    if df_filt.empty:
        st.warning("No metabolites above the selected priority score.")
        return

    # -----------------------------
    # Group by reaction type
    # -----------------------------
    reaction_order = (
        df_filt.groupby("Reaction_Type")["Priority_Score"]
        .max()
        .sort_values(ascending=False)
    )

    for rxn_type in reaction_order.index:

        df_rxn = df_filt[df_filt["Reaction_Type"] == rxn_type]

        with st.expander(
            f"{rxn_type} ({len(df_rxn)} metabolite(s))",
            expanded=True,
        ):

            mols = []
            legends = []

            df_rxn = df_rxn.sort_values(
                "Priority_Score", ascending=False
            )

            for _, row in df_rxn.iterrows():
                mol = Chem.MolFromSmiles(row["Metabolite_SMILES"])
                if mol:
                    mols.append(mol)
                    legends.append(
                        f"Score: {row['Priority_Score']:.2f}"
                    )

            if mols:
                img = Draw.MolsToGridImage(
                    mols,
                    molsPerRow=mols_per_row,
                    subImgSize=img_size,
                    legends=legends,
                    useSVG=False,
                )
                st.image(img, use_container_width=True)

    # -----------------------------
    # Full table and download
    # -----------------------------
    with st.expander("📊 Show full metabolite table"):
        st.dataframe(
            df_met.sort_values(
                "Priority_Score", ascending=False
            ),
            use_container_width=True,
        )

        csv = df_met.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download metabolite table (CSV)",
            data=csv,
            file_name="gloryx_metabolites.csv",
            mime="text/csv",
        )