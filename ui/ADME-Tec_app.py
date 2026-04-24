
"""
ADME‑TEC Streamlit Application
==============================


This module implements an interactive web interface for:
- ADME prediction using ADMET‑AI
- ChEMBL / DrugBank reference retrieval
- Metabolite prediction (GLORYx)
- Chemical similarity analysis
- ADME radar visualization
- Desirability‑based compound prioritization


The code is organized in logical sections following the Streamlit
execution flow to ensure clarity and reproducibility.


Author: Fernanda Saldivar
"""

# ============================== IMPORTS ==============================
import pandas as pd
import streamlit as st
from admet_ai import ADMETModel
from PIL import Image
import os
import torch, argparse
import sys
from pathlib import Path
import base64
from io import BytesIO
from rdkit import Chem

# --- Add project root to Python path ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --- Internal project modules ---
from src.utils.inputs import molecule_input
from src.utils.visualization import show_molecules
from src.target.chembl_utils import get_mechanism_metadata, retrieve_chembl_data
from src.chemistry.standardizer_utils import process_molecule_row
from src.admet.radar_plot import plot_radar_with_min_max_df
from src.admet.adme_mappings import categories_adme, map_columns_perc, map_columns
from src.target.ATC_utils import get_drugbank_by_atc
from src.admet.GloryX import calcular_metabolitos, visualizar_metabolitos
from src.chemistry.similarity import calcular_similitud, visualizar_top_similares, plot_similarity_bars
from src.admet.ranges_utils import prepare_ranges_from_reference
from src.admet.desirability_score import normalize_weights, compute_desirability, compute_desirability_geometric
from src.admet.desirability_conf import PROPERTY_CONFIG 
#from src.admet.plot_utils import plot_desirability_with_uncertainty 

# ========================= ADMET MODEL LOADING ==========================
@st.cache_resource
def load_admet_model():
    return ADMETModel()
# ============================== TORCH PATCH =============================
# Fix compatibility with PyTorch ≥ 2.6 when loading serialized objects
_original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    with torch.serialization.safe_globals([argparse.Namespace]):
        return _original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

# ============================= PAGE CONFIG ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


ICON_PATH = PROJECT_ROOT / "src" / "assets" / "IOR1.png"

icon = Image.open(ICON_PATH)

st.set_page_config(
    page_title="ADME-TEC",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================== UI STYLING ==============================
st.markdown(
    """
    <style>
    .block-container { margin: auto; max-width: 1100px; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True
)

def icon_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

banner_url = "https://tec.mx/es/investigacion/instituto-de-investigacion-sobre-obesidad"
col1, col2, col3 = st.columns([0.3, 3, 0.3])

with col2:
    st.markdown(
        f"""
        <a href="{banner_url}" target="_blank">
            <img src="data:image/png;base64,{icon_to_base64(icon)}" style="width:100%; height:auto;">
        </a>
        """,
        unsafe_allow_html=True
    )
    st.title("ADME-TEC")
    st.markdown(
        """
        <div style="text-align: justify; font-size:18px; line-height:2.5;">
            Interactive platform for <strong>ADME</strong> analysis, prediction,
            and visualization of chemical compound properties.
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================= SESSION STATE =============================
#"""Initialize persistent variables used across Streamlit reruns."""

default_states = {
    "adme_df": pd.DataFrame(),
    "adme_chembl_df": pd.DataFrame(),
    "adme_atc_df": pd.DataFrame(),
    "results": [],
    "selected_actions": None,
    "metabolites_df": pd.DataFrame(),
    "selected_adme_props": [],
    "adme_weights": {},
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================== SIDEBAR INPUTS ==============================

st.sidebar.header("Molecule Metadata")

# --- Example loader for demonstration purposes ---
if st.sidebar.button("🔹Load Example"):
    st.session_state["use_example"] = True
    st.session_state["example_smiles"] = ["CC(C)NCC(O)COc1ccccc1"]  # Propranolol
    st.session_state["example_chembl"] = "CHEMBL235"
    st.session_state["example_atc"] = ""
    st.session_state["selected_actions"] = None
    st.session_state["auto_run"] = True

if st.session_state.get("use_example", False):
    smiles_list = st.session_state["example_smiles"]
    chembl_target = st.session_state["example_chembl"]
    atc_code = st.session_state["example_atc"]
    show_molecules(smiles_list)
    st.sidebar.text_area("Enter SMILES", value="\n".join(smiles_list), disabled=True)
    st.sidebar.text_input("CHEMBL target ID", value=chembl_target, disabled=True)
    st.sidebar.text_input("ATC code", value=atc_code, disabled=True)
    input_df = None 

# ============================== INPUT HANDLING ==============================
else:
    with st.sidebar:
        result = molecule_input()

    # --- Handle return safely (supports old or new version) ---
    if isinstance(result, tuple) and len(result) == 2:
        smiles_list, input_df = result
    else:
        smiles_list = result
        input_df = None

    # --- Store full dataframe if available (CSV case) ---
    if input_df is not None and isinstance(input_df, pd.DataFrame):
        st.session_state.input_df = input_df

    # --- Show molecules ---
    if smiles_list and st.session_state.get("show_mols", True):
        show_molecules(smiles_list)

    # --- Sidebar metadata inputs ---
    chembl_target = st.sidebar.text_input(
        "CHEMBL target ID",
        placeholder="e.g., CHEMBL235",
        disabled=bool(st.session_state.get("atc_code"))
    )

    atc_code = st.sidebar.text_input(
        "ATC code",
        placeholder="e.g., N02BE01",
        disabled=bool(chembl_target)
    )

# ================= Metadata selectors =================

design_phase = st.sidebar.selectbox(
    "Drug design phase",
    ['Hit identification','Lead optimization','Candidate selection']
)

target_location = st.sidebar.selectbox(
    "Target location",
    ['Extracellular','Intracellular','Crosses BBB']
)

# Store metadata in session
st.session_state['chembl_target'] = chembl_target
st.session_state['atc_code'] = atc_code
st.session_state['design_phase'] = design_phase
st.session_state['target_location'] = target_location

# ================= Run control =================

if st.session_state.get("auto_run", False):
    run_analysis = True
    st.session_state["auto_run"] = False
else:
    run_analysis = st.sidebar.button("▶ Run ADME Analysis")


# ============================== ADME PREDICTION ==============================
if smiles_list:
    if st.session_state.adme_df.empty:
        with st.spinner("Calculating ADME for input molecules..."):
            model = load_admet_model()
            adme_results = model.predict(smiles_list)

            st.session_state.adme_df = pd.DataFrame(adme_results)

            # Insert structural columns immediately
            st.session_state.adme_df.insert(0, "smiles", smiles_list)

            if input_df is not None and "ID" in input_df.columns:
                st.session_state.adme_df["ID"] = input_df["ID"].values

    st.markdown("### ADME Properties of Input Molecules")
    st.dataframe(st.session_state.adme_df, use_container_width=True)


# =========================== METABOLITE PREDICTION ============================


if smiles_list:

    st.markdown("### Metabolite Prediction (GLORYx)")

    # CASE 1: CSV uploaded with ID column
    if "input_df" in st.session_state and \
       isinstance(st.session_state.input_df, pd.DataFrame) and \
       {"ID", "smiles"}.issubset(st.session_state.input_df.columns):

        df_input = st.session_state.input_df

        selected_id = st.selectbox(
            "Select compound ID to predict metabolites:",
            options=df_input["ID"].unique(),
            key="met_select_id"
        )

        smi_for_met = df_input.loc[
            df_input["ID"] == selected_id,
            "smiles"
        ].values[0]

    # CASE 2: Manual SMILES (no ID available)
    else:

        selected_index = st.selectbox(
            "Select input molecule:",
            options=list(range(len(smiles_list))),
            format_func=lambda x: f"Molecule {x+1}",
            key="met_select_index"
        )

        smi_for_met = smiles_list[selected_index]

    # Predict metabolites
    if st.button("Predict Metabolites (GLORYx)"):

        with st.spinner("Predicting metabolites with GLORYx..."):
            mols, labels, df_met = calcular_metabolitos(smi_for_met)

        st.session_state.metabolites_df = df_met


# Display results
if "metabolites_df" in st.session_state:
    visualizar_metabolitos(st.session_state.metabolites_df)


# ============================== REFERENCE RETRIEVAL ============================== 
#"""Retrieve ChEMBL or DrugBank reference compounds and compute ADME."""

# =========================================================
# -------------------- ChEMBL SECTION ---------------------
# =========================================================

if chembl_target:

    chembl_preview = get_mechanism_metadata(chembl_target)

    if not chembl_preview.empty:

        # ---------------------------
        # Select action types for ChEMBL compounds if not already selected and cache in session state
        # ---------------------------
        if st.session_state.get("selected_actions") is None:

            st.markdown("#### Select Action Type(s) for ChEMBL Compounds")

            with st.form("action_form"):

                available_actions = (
                    chembl_preview['action_type']
                    .dropna()
                    .unique()
                    .tolist()
                )

                selected = []

                for act in available_actions:
                    if st.checkbox(act, key=f"act_{act}"):
                        selected.append(act)

                submitted = st.form_submit_button("Confirm Selection")

                if submitted and selected:
                    st.session_state.selected_actions = selected

        # ---------------------------
        # Calculate ADME for ChEMBL compounds based on selected action types and cache results
        # ---------------------------
        if (
            st.session_state.get("selected_actions") is not None
            and st.session_state.adme_chembl_df.empty
        ):

            try:
                with st.spinner("Calculating ADME for ChEMBL compounds..."):

                    chembl_df, adme_chembl_df = retrieve_chembl_data(
                        chembl_target,
                        st.session_state.selected_actions
                    )

                    st.session_state.chembl_df = chembl_df
                    st.session_state.adme_chembl_df = adme_chembl_df

            except Exception as e:
                st.error(f"Error retrieving ChEMBL or ADME data: {e}")

        # ---------------------------
        # Display ChEMBL data and ADME predictions if available
        # ---------------------------
        if not st.session_state.get("chembl_df", pd.DataFrame()).empty:

            st.markdown("#### Retrieved Data from ChEMBL")
            st.dataframe(
                st.session_state.chembl_df,
                use_container_width=True
            )

        if not st.session_state.adme_chembl_df.empty:

            st.markdown("#### ADME Properties of ChEMBL Compounds")
            st.dataframe(
                st.session_state.adme_chembl_df,
                use_container_width=True
            )


# =========================================================
# ----------------------- ATC SECTION ---------------------
# =========================================================

if atc_code:

    # ---------------------------
    # Reset ATC-related session state if code changes
    # ---------------------------
    if "last_atc" not in st.session_state:
        st.session_state.last_atc = None

    if atc_code != st.session_state.last_atc:
        st.session_state.adme_atc_df = pd.DataFrame()
        st.session_state.last_atc = atc_code

    # ---------------------------
    # obtain DrugBank compounds for ATC code and cache in session state
    # ---------------------------
    df_drugbank_atc = get_drugbank_by_atc(atc_code)
    st.session_state.df_drugbank_atc = df_drugbank_atc

    if not df_drugbank_atc.empty:

        st.markdown(
            f"### DrugBank Compounds for ATC {atc_code.upper()}"
        )

        st.dataframe(
            df_drugbank_atc,
            use_container_width=True
        )

    # STORE ALL NON-ATC COLUMNS AS ADME DATA
    # -------------------------------------------------
    exclude_cols = [
        "atc",
        "atc_name_1",
        "atc_name_2",
        "atc_name_3",
        "atc_name_4",
    ]

    admet_cols = [
        c for c in df_drugbank_atc.columns
        if c not in exclude_cols
    ]

    if admet_cols:

        st.session_state.adme_atc_df = df_drugbank_atc[admet_cols].copy()

    

#============================= CHEMICAL SIMILARITY =============================

#=========== Reference dataset selection (ChEMBL or DrugBank ATC)  ===========
# This block determines which reference chemical dataset
# is available in the Streamlit session state and sets:
#   - df_ref: dataframe containing reference molecules
#   - id_col: column with compound identifier
#   - smiles_col: column containing SMILES strings
# Priority is given to ChEMBL if both are present.

# ==============================
# CHEMICAL SIMILARITY
# ==============================

df_ref = None
id_col = None
smiles_col = None
current_source = None


if atc_code and (
    "df_drugbank_atc" in st.session_state
    and not st.session_state.df_drugbank_atc.empty
):
    df_ref = st.session_state.df_drugbank_atc.copy()
    id_col = "name"
    smiles_col = "smiles"
    current_source = "atc"

elif chembl_target and (
    "chembl_df" in st.session_state
    and not st.session_state.chembl_df.empty
):
    df_ref = st.session_state.chembl_df.copy()
    id_col = "molecule_chembl_id"
    smiles_col = "smiles"
    current_source = "chembl"

# -----------------------------
# Reset cache for similarity calculations if reference source changes
# -----------------------------

if st.session_state.get("similarity_source") != current_source:
    st.session_state.pop("similarity_df", None)
    st.session_state["similarity_source"] = current_source

# ==========================================================
# Execute chemical similarity analysis if reference dataset and input SMILES are available
# ==========================================================

if df_ref is not None and smiles_list:

    st.markdown("## Chemical Similarity (ChEMBL / ATC)")

    # -------------------------
    # Process reference dataset: standardize SMILES, curate structures, and prepare for similarity calculations
    # -------------------------

    df_proc = df_ref[[smiles_col, id_col]].dropna().copy()

    processed = df_proc.apply(process_molecule_row, axis=1, result_type="expand")
    df_proc = pd.concat([df_proc, processed], axis=1)

    df_proc = df_proc[df_proc["error"].isna()].copy()

    df_proc["curated_smiles"] = df_proc["curated_smiles"].astype(str)
    df_proc[id_col] = df_proc[id_col].astype(str)

    df_proc = df_proc.drop_duplicates(
        subset=[id_col, "curated_smiles"]
    ).reset_index(drop=True)

    # -------------------------
    # Process input SMILES: standardize and curate the first input molecule for similarity comparison
    # -------------------------

    input_df = pd.DataFrame({"smiles": [smiles_list[0]]})
    input_proc = input_df.apply(process_molecule_row, axis=1, result_type="expand")

    if not input_proc["error"].isna().iloc[0]:

        st.error("Input SMILES could not be standardized.")

    else:

        curated_input = str(input_proc["curated_smiles"].iloc[0])

        # -------------------------
        # Calculate chemical similarity between the curated input molecule and the reference dataset
        # -------------------------

        if "similarity_df" not in st.session_state:

            with st.spinner("Calculating chemical similarity..."):
                st.session_state.similarity_df = calcular_similitud(
                    input_smiles=curated_input,
                    df_ref=df_proc,
                    smiles_col="curated_smiles",
                    id_col=id_col
                )

        # -------------------------
        # Visualize top similar compounds and their similarity scores using bar plots
        # -------------------------

        visualizar_top_similares(
            input_smiles=curated_input,
            df_sim=st.session_state.similarity_df,
            top_n=5,
        )

        fig = plot_similarity_bars(
            st.session_state.similarity_df,
            top_n=5
        )

        st.pyplot(fig)

        st.markdown(
            "Similarity computed using Morgan fingerprints (2048 bits) "
            "and Tanimoto coefficient."
        )


# ============================================================
# ============================================================
# ADME PROPERTY SELECTION, WEIGHTING & RADAR VISUALIZATION
# ============================================================
# This block manages interactive ADME profiling in three stages: 
# # 1) Detection of available ADME reference dataset 
# # 2) User-driven selection of ADME properties and weight assignment 
# # 3) Radar plot comparison between input compound(s) and reference space  
# # The workflow is executed only when: 
# # - ADME predictions exist for the input compound(s) 
# # - At least one reference ADME dataset (ChEMBL or DrugBank ATC) is available 

# ------------------------------------------------------------
# Initialize session state safely
# ------------------------------------------------------------
if "selected_adme_props" not in st.session_state:
    st.session_state.selected_adme_props = []

if "adme_weights" not in st.session_state:
    st.session_state.adme_weights = {}


# ------------------------------------------------------------
# Detect available ADME reference dataset
# ------------------------------------------------------------
ref_key = None
ref_label = None

if (
    "adme_chembl_df" in st.session_state
    and not st.session_state.adme_chembl_df.empty
):
    ref_key = "adme_chembl_df"
    ref_label = "ChEMBL compounds"

elif (
    "adme_atc_df" in st.session_state
    and not st.session_state.adme_atc_df.empty
):
    ref_key = "adme_atc_df"
    ref_label = "DrugBank (ATC) compounds"


# ------------------------------------------------------------
# Display ADME profiling only if input + reference exist
# ------------------------------------------------------------
if (
    "adme_df" in st.session_state
    and not st.session_state.adme_df.empty
    and ref_key is not None
):

    ref_df = st.session_state[ref_key]
    input_adme_df = st.session_state.adme_df

    st.markdown("---")
    st.markdown(f"## ADME Profiling vs {ref_label}")

    is_single_molecule = len(input_adme_df) == 1


    # ========================================================
    # STEP 1 — PROPERTY SELECTION
    # ========================================================
    with st.form("select_adme_props_form"):

        st.markdown("### 1. Select ADME properties")

        selected_tmp = []

        for category, props in categories_adme.items():
            with st.expander(category, expanded=False):
                for prop in props:
                    if st.checkbox(
                        prop,
                        key=f"sel_{prop}",
                        value=prop in st.session_state.selected_adme_props,
                    ):
                        selected_tmp.append(prop)

        submitted_props = st.form_submit_button("Confirm property selection")

    if submitted_props:
        if not selected_tmp:
            st.warning("Please select at least one ADME property.")
        else:
            st.session_state.selected_adme_props = selected_tmp

            # Reset weights ONLY for selected properties
            st.session_state.adme_weights = {
                prop: st.session_state.adme_weights.get(prop, 1.0)
                for prop in selected_tmp
            }

            st.success("ADME properties selected.")


    # ========================================================
    # STEP 2 — WEIGHTS (ONLY FOR MULTIPLE MOLECULES)
    # ========================================================
    if not is_single_molecule and st.session_state.selected_adme_props:

        with st.form("assign_adme_weights_form"):

            st.markdown("### 2. Assign weights to selected properties")

            weights_tmp = {}

            for prop in st.session_state.selected_adme_props:

                weight = st.slider(
                    f"Weight for {prop}",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.adme_weights.get(prop, 1.0),
                    step=0.05,
                    key=f"w_{prop}",
                )

                weights_tmp[prop] = weight

            submitted_weights = st.form_submit_button("Confirm weights")

        if submitted_weights:
            st.session_state.adme_weights = weights_tmp
            st.success("ADME weights saved successfully.")


    # ========================================================
    # STEP 3 — RADAR VISUALIZATION
    # ========================================================
    show_radar = False

    if is_single_molecule and st.session_state.selected_adme_props:
        show_radar = True

    elif (
        not is_single_molecule
        and st.session_state.selected_adme_props
        and st.session_state.adme_weights
    ):
        show_radar = True


    if show_radar:

        st.markdown("### ADME radar comparison")

        selected_cols = [
            map_columns_perc[p]
            for p in st.session_state.selected_adme_props
            if p in map_columns_perc
        ]

        missing_cols = [
            col for col in selected_cols
            if col not in input_adme_df.columns or col not in ref_df.columns
        ]

        if missing_cols:
            st.warning(f"Missing required ADME columns: {missing_cols}")

        else:
            min_df = pd.DataFrame([ref_df[selected_cols].min()])
            max_df = pd.DataFrame([ref_df[selected_cols].max()])
            compuestos_df = input_adme_df[selected_cols].reset_index(drop=True)

            fig = plot_radar_with_min_max_df(
                min_df=min_df,
                max_df=max_df,
                compuestos_df=compuestos_df,
                title=f"ADME Profile Comparison vs {ref_label}",
            )

            st.pyplot(fig, clear_figure=True)


# ========================================================
# DESIRABILITY SCORING DEPENDING ON DESIGN PHASE
# ========================================================

if (
    ref_key is not None
    and "adme_df" in st.session_state
    and not st.session_state.adme_df.empty
):

    ref_df = st.session_state[ref_key]
    input_adme_df = st.session_state.adme_df

    is_single_molecule = len(input_adme_df) == 1
    is_hit_phase = design_phase == "Hit identification"
    is_geo_phase = design_phase in ["Lead optimization", "Candidate selection"]

    # -----------------------------
    # UI selections
    # -----------------------------
    selected_ui_props = st.session_state.get("selected_adme_props", [])
    all_ui_weights = st.session_state.get("adme_weights", {})

    # -----------------------------
    # Convert UI → internal keys
    # -----------------------------
    selected_internal_props = [
        map_columns[p]
        for p in selected_ui_props
        if p in map_columns
    ]

    # -----------------------------
    # Convert weights to internal keys
    # -----------------------------
    internal_weights = {
        map_columns[k]: v
        for k, v in all_ui_weights.items()
        if k in map_columns
    }

    # -----------------------------
    # Filter only selected properties
    # -----------------------------
    filtered_weights = {
        prop: internal_weights[prop]
        for prop in selected_internal_props
        if prop in internal_weights
    }

    enable_desirability = (
        not is_single_molecule
        and selected_internal_props
        and filtered_weights
        and (is_hit_phase or is_geo_phase)
    )

    if enable_desirability:

        st.markdown("### ADME desirability scoring")

        try:
            weights = normalize_weights(filtered_weights)
            
            st.write("Normalized weights:", weights)
            st.write("Sum normalized weights:", sum(weights.values()))

            # -----------------------------------------
            # Filter PROPERTY_CONFIG only for selected
            # -----------------------------------------
            filtered_config = {
                k: v
                for k, v in PROPERTY_CONFIG.items()
                if k in selected_internal_props
            }
            # -----------------------------------------
            # Build ranges from reference dataset
            # -----------------------------------------
            ranges = prepare_ranges_from_reference(ref_df, filtered_config)
            
            #-------------------------------------------
            if is_hit_phase:

                st.markdown("#### Linear desirability (Hit identification)")

                desirability_df = compute_desirability(
                    inputs=input_adme_df,
                    ranges=ranges,
                    weights=weights,
                    config=filtered_config,
                )

                st.session_state.desirability_df = desirability_df
                st.dataframe(desirability_df, use_container_width=True)

            elif is_geo_phase:

                st.markdown("#### Geometric desirability (Lead/Candidate stage)")

                desirability_geo_df = compute_desirability_geometric(
                    inputs=input_adme_df,
                    ranges=ranges,
                    weights=weights,
                    config=filtered_config,
                )

                st.session_state.desirability_geo_df = desirability_geo_df
                st.dataframe(desirability_geo_df, use_container_width=True)

        except Exception as e:
            st.error(f"Error computing desirability: {e}")


# ---------------------- Contact ----------------------
st.markdown("---")
st.markdown("For questions or support, contact: **Fernanda Saldivar** – [fer.saldivarg@tec.mx](mailto:fernanda.saldivarg@tec.mx)")