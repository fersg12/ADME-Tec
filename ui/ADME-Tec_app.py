
"""
ADME‑TEC Streamlit Application
==============================


This module implements an interactive web interface for:
- ADMET prediction using ADMET‑AI
- ChEMBL / DrugBank reference retrieval
- Metabolite prediction (GLORYx)
- Desirability‑based compound prioritization
- Chemical similarity analysis
- ADMET radar visualization

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
import plotly.graph_objects as go
import numpy as np 
import pickle
import time 
import matplotlib.pyplot as plt

# --- Add project root to Python path ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# --- Internal project modules ---
from src.utils.inputs import molecule_input
from src.utils.visualization import show_molecules
from src.target.chembl_utils import get_mechanism_metadata, retrieve_chembl_data
from src.chemistry.standardizer_utils import process_molecule_row
from src.admet.radar_plot import plot_radar_with_min_max_df, plot_desirability_heatmap, render_heatmap
from src.admet.adme_mappings import categories_adme, map_columns_perc, map_columns
from src.target.ATC_utils import get_drugbank_by_atc
from src.admet.GloryX import calcular_metabolitos, visualizar_metabolitos
from src.chemistry.similarity import calcular_similitud, plot_heatmap_similitud, visualizar_top_similares, plot_similarity_bars, highlight_max_ref
from src.admet.ranges_utils import prepare_ranges_from_reference
from src.admet.desirability_score import normalize_weights, compute_desirability, compute_desirability_geometric
from src.admet.desirability_conf import PROPERTY_CONFIG 
from pathlib import Path
from src.admet.reference_analysis import compute_reference_variability, compute_reference_correlation, plot_reference_correlation
from src.admet.Sensitive_test import perturb_weights_sensitivity


# Robust TOML loader: prefer stdlib tomllib (Python 3.11+), fall back to the 'toml' package.
_load_toml = None
try:
    import tomllib as _tomllib  # type: ignore
    def _load_toml(path):
        with open(path, "rb") as f:
            return _tomllib.load(f)
except Exception:
    try:
        import toml as _toml  # type: ignore
        def _load_toml(path):
            return _toml.load(str(path))
    except Exception:
        _load_toml = None

config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"

config = {}
if _load_toml is None:
    print("Warning: no TOML loader available (neither tomllib nor toml). Skipping .streamlit/config.toml parsing.")
else:
    try:
        if config_path.exists():
            config = _load_toml(config_path)
        else:
            print(f"Warning: .streamlit/config.toml not found at {config_path}")
    except Exception as e:
        print(f"Warning: failed to load {config_path}: {e}")

# If a [theme] section exists in config.toml, apply a minimal CSS fallback so
# the app visually matches the configured theme even when Streamlit itself
# didn't pick up the file (e.g., when running the script directly).
try:
    theme = config.get("theme", {}) if isinstance(config, dict) else {}
    if theme:
        css_rules = []
        bg = theme.get("backgroundColor")
        if bg:
            css_rules.append(f".stApp {{ background-color: {bg} !important; }}")
        sec_bg = theme.get("secondaryBackgroundColor")
        if sec_bg:
            css_rules.append(f".stSidebar {{ background-color: {sec_bg} !important; }}")
        text = theme.get("textColor")
        if text:
            css_rules.append(f".stApp, .stApp * {{ color: {text} !important; }}")
        primary = theme.get("primaryColor")
        if primary:
            # Attempt to colour major accents (buttons, links)
            css_rules.append(f".stButton>button, a {{ background-color: {primary} !important; border-color: {primary} !important; }}")
        if css_rules:
            # Do not call Streamlit APIs before set_page_config; store CSS to apply later
            THEME_CSS = "<style>" + "\n".join(css_rules) + "</style>"
        else:
            THEME_CSS = None
except Exception as _e:
    # Silently ignore any styling errors to avoid breaking the app UI
    THEME_CSS = None
    print(f"Warning: failed to prepare theme CSS from {config_path}: {_e}")

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

logo = Image.open("src/assets/logo.png")

st.set_page_config(
    page_title="ADMETec",
    page_icon=logo,
    layout="wide"
)

col_logo, col_titulo = st.columns([1, 8], gap="medium")

with col_logo:
    st.image("src/assets/logo.png", use_container_width=True)

with col_titulo:
    st.title("ADMETec")
    st.caption("Context-aware ADMET analysis and compound prioritization for drug discovery")


# ==============================
# REQUIRED INPUT (EXPANDER)
# ==============================
with st.expander("ℹ️ Required Input", expanded=False):
    st.markdown("""
    <div style="text-align: justify;">
    <ul>
        <li><strong>Molecules (required):</strong>  Compound name or SMILES strings (single entry or CSV upload).</li>
        <li><strong>ChEMBL ID (optional)::</strong> Identifier from the ChEMBL database used to retrieve bioactivity data, target information, and mechanism of action for a specific protein.<br>
    <em>Examples:</em> CHEMBL235 (PPARγ), CHEMBL204 (EGFR). <br>
                Learn how to find CHEMBL target IDs:  
    <a href="https://www.ebi.ac.uk/chembl/explore/targets/" target="_blank">
    ChEMBL Targets
    </a>
        </li>
        <li><strong>ATC Code (optional):</strong> Classification system that groups drugs by therapeutic indication.  
    Useful to contextualize compounds within a pharmacological class.<br>
    <em>Examples:</em> <code>N02</code> → Analgesics,<code>A10</code> → Antidiabetics,<code>C08</code> → Calcium Channel Blockers <br>
    Learn how to find ATC codes:
        <a href="https://atcddd.fhi.no/atc_ddd_index/" target="_blank">
    ATC/DDD Index
    </a>
    </li>
        <li><strong>Drug Design Phase:</strong> Indicates the stage of the drug discovery pipeline, which influences how compounds are evaluated:<br>
    <ul>
        <li><strong>Hit identification:</strong> Early stage → prioritize broad exploration and moderate ADME constraints. 
    A <strong>linear desirability function</strong> is applied, allowing partial satisfaction of multiple properties and promoting chemical diversity.
    </li>
        <li><strong>Lead optimization:</strong> Balance potency, ADME, and safety. 
    A <strong>geometric desirability function</strong> is used, increasing sensitivity to poorly optimized properties and enforcing a more balanced profile.
    </li>
        <li><strong>Candidate selection:</strong> Strict optimization → high ADME, safety, and developability requirements. 
    A <strong>geometric desirability function</strong> is also applied, strongly penalizing any suboptimal property to ensure robust drug-like behavior.
    </li>
    </ul>
    </li>
        <li><strong>Target Location:</strong>     Biological location of the target, which defines ADME requirements:<br>
    <ul>
        <li><strong>Extracellular:</strong> Limited permeability required</li>
        <li><strong>Intracellular:</strong> Cell permeability is important</li>
        <li><strong>Crosses BBB:</strong> CNS drugs → must cross the blood-brain barrier (BBB), often requiring specific physicochemical properties (e.g., bRo5 space)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


with st.expander("⚙️ Functionalities", expanded=False):

    st.markdown("""
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">

    <style>
    .card {
        background-color: #e8f1faff;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
        transition: 0.2s;
        margin-bottom: 15px;
    }

    .card:hover {
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        transform: translateY(-3px);
    }

    .title {
        font-weight: 600;
        font-size: 16px;
        color: #1E3A8A;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .material-icons {
        font-size: 20px;
        color: #1E3A8A;
    }

    .text {
        font-size: 14px;
        color: #475569;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="title">
                <span class="material-icons">hub</span>
                Data Retrieval & Standardization
            </div>
            <div class="text">
                Retrieve approved drugs and clinical compounds from ChEMBL.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="title">
                <span class="material-icons">science</span>
                ADMET prediction
            </div>
            <div class="text">
                Compute physicochemical and ADMET properties using ADMET-AI.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="title">
                <span class="material-icons">analytics</span>
                Desirability Score
            </div>
            <div class="text">
                Prioritize compounds based on customizable ADMET desirability.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="title">
                <span class="material-icons">hub</span>
                Expert modulation
            </div>
            <div class="text">
                Select and weight ADMET properties based on project needs.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================= SESSION STATE =============================
# Initialize persistent variables used across Streamlit reruns.

def save_session_state(file_path= "session_state.pkl"):
    with open(file_path, 'wb') as f:
            pickle.dump(st.session_state.to_dict(), f)


def load_session_state(file_path = "session_state.pkl"):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
                loaded_state = pickle.load(f)
                #print("##### LOADED SESSION ####")
                values = []
                for k, v in loaded_state.items():
                    if k != "FormSubmitter:select_adme_props_form-Confirm property selection" and k != "FormSubmitter:assign_adme_weights_form-Confirm weights":
                        st.session_state[k] = v
                        values.append(k)
                        #print(f"Loaded {k} into session state = {v}")
                #values.sort()
                #for v in values:
                #    print(v)
    else:
        print("File not found.")
default_states = {
    "adme_df": pd.DataFrame(),
    "adme_chembl_df": pd.DataFrame(),
    "adme_atc_df": pd.DataFrame(),
    "results": [],
    "selected_actions": None,
    "metabolites_df": pd.DataFrame(),
    "selected_adme_props": [],
    "adme_weights": {},
    "weights_confirmed": False,
    #Datos de diego para cargar ejemplo
    "result_molecule_input": ([], None),
    "df_sim_matrix": None,
    "use_loaded_values": False
}
def asignar_session():
    chembl_target = st.session_state['chembl_target']
    atc_code = st.session_state['atc_code']
    design_phase = st.session_state['design_phase']
    target_location = st.session_state['target_location']
    result = st.session_state["result_molecule_input"]
    input_df = st.session_state.input_df
    selected = st.session_state.selected_actions
    df_input = st.session_state.input_df
    df_met = st.session_state.metabolites_df
    chembl_df = st.session_state.chembl_df
    adme_chembl_df = st.session_state.adme_chembl_df
    #atc_code = st.session_state.last_atc
    #df_drugbank_atc = st.session_state.df_drugbank_atc
    weights_tmp = st.session_state.adme_weights
    input_adme_df = st.session_state.adme_df
    selected_ui_props = st.session_state.get("selected_adme_props", [])
    all_ui_weights = st.session_state.get("adme_weights", {})
    weights_ready = st.session_state.get("weights_confirmed", False)
    desirability_df = st.session_state.desirability_df
    desirability_ref_df = st.session_state.desirability_ref_df
    #desirability_geo_df = st.session_state.desirability_geo_df
    df_ref_plot = st.session_state.desirability_ref_df.copy()
    current_source = st.session_state["similarity_source"]
    df_query = st.session_state.input_df.copy()
use_loaded_values = False
if  "use_loaded_values" in st.session_state:
    use_loaded_values = st.session_state["use_loaded_values"]
# una vez que ya hayas guardado los datos cambia a True
if use_loaded_values:
    load_session_state()
    asignar_session()
else:
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================== SIDEBAR INPUTS ==============================
with st.sidebar:
    #if use_loaded_values : st.text("Usando valores pre cargados")
    #st.button("Save Session State", on_click=save_session_state )
    st.title("Input Parameters")

# ================= SIDEBAR =================
    if st.session_state["result_molecule_input"] == ([], None): 
        result = molecule_input()
    else:
        result = st.session_state["result_molecule_input"]
    chembl_target = st.text_input(
        "CHEMBL target ID",
        placeholder="e.g., CHEMBL235",
        disabled=bool(st.session_state.get("atc_code"))
    )

    atc_code = st.text_input(
        "ATC code",
        placeholder="e.g., N02",
        disabled=bool(chembl_target)
    )

    # ================= Metadata selectors =================

    design_phase = st.selectbox(
        "Drug design phase",
        ['Hit identification','Lead optimization','Candidate selection']
    )

    target_location = st.selectbox(
        "Target location",
        ['Extracellular','Intracellular','Crosses BBB', 'bR05 target']
    )

    # ================= Run control + Example button =================
    if st.session_state.get("auto_run", False):
        run_analysis = True
        st.session_state["auto_run"] = False
        load_example_btn = False  # 
    else:
        run_analysis = st.button("▶ Run ADME Analysis")
        load_example_btn = st.button("🔹 Load Example")  

# ================= Store metadata =================
if not use_loaded_values:
    st.session_state['chembl_target'] = chembl_target
    st.session_state['atc_code'] = atc_code
    st.session_state['design_phase'] = design_phase
    st.session_state['target_location'] = target_location

# ================= LOGIC ========================
if load_example_btn:
    load_session_state()
    asignar_session()
    use_loaded_values = True
    # mark that we are intentionally using the provided example dataset
    st.session_state["use_loaded_values"] = True
    st.session_state["use_example"] = True
    st.rerun()

# ================= Handle input =================

if isinstance(result, tuple) and len(result) == 2:
    smiles_list, input_df = result
    st.session_state["result_molecule_input"] = result
else:
    smiles_list = result
    input_df = None

if (smiles_list or input_df is not None) and not load_example_btn:
    st.session_state["use_example"] = False

if input_df is not None and isinstance(input_df, pd.DataFrame):
    st.session_state.input_df = input_df

# ================= Example mode =================
if st.session_state.get("use_example", False):
    smiles_list = st.session_state["example_smiles"]
    chembl_target = st.session_state["example_chembl"]
    atc_code = st.session_state["example_atc"]

    show_molecules(smiles_list)

    input_df = None

# ================= DISPLAY NORMAL =================
elif smiles_list and st.session_state.get("show_mols", True):
    ids = None

    if input_df is not None and "ID" in input_df.columns:
        ids = input_df["ID"].astype(str).tolist()

    show_molecules(smiles_list, ids=ids)
    

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


# ============================== ADMET PREDICTION ==============================
if smiles_list:
    if st.session_state.adme_df.empty:
        with st.spinner("Calculating ADMET for input molecules..."):
            model = load_admet_model()
            adme_results = model.predict(smiles_list)

            adme_df = pd.DataFrame(adme_results)

            # Create complementary BBB score (high = safer, low BBB penetration)
            if "BBB_Martins" in adme_df.columns:
                adme_df["BBB_Martins_Safe"] = 1 - adme_df["BBB_Martins"]

            # Insert structural columns
            adme_df.insert(0, "smiles", smiles_list)

            if input_df is not None and "ID" in input_df.columns:
                adme_df["ID"] = input_df["ID"].values

            st.session_state.adme_df = adme_df

    st.markdown("## ADMET Properties of Input Molecules")
    st.dataframe(st.session_state.adme_df, use_container_width=True)


# ==================== REFERENCE RETRIEVAL =======================
# Calculate ADMET for ChEMBL compounds based on selected action types and cache results
# ---------------------------
        
    if (
        st.session_state.get("selected_actions") is not None
        and st.session_state.adme_chembl_df.empty
        ):

        try:
            with st.spinner("Calculating ADMET for ChEMBL compounds..."):

                    chembl_df, adme_chembl_df = retrieve_chembl_data(
                    chembl_target,
                    st.session_state.selected_actions
                    )
                    # Create complementary BBB score (high = safer, low BBB penetration)
                    if "BBB_Martins" in adme_chembl_df.columns:
                        adme_chembl_df["BBB_Martins_Safe"] = 1 - adme_chembl_df["BBB_Martins"]
                    st.session_state.chembl_df = chembl_df
                    st.session_state.adme_chembl_df = adme_chembl_df

        except Exception as e:
                st.error(f"Error retrieving ChEMBL or ADMET data: {e}")

    # ---------------------------
    # Display ADMET predictions if available
    # ---------------------------

    if not st.session_state.adme_chembl_df.empty:
        st.markdown("---")
        st.markdown("## ADMET Properties of ChEMBL Compounds")
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
    # Obtain DrugBank compounds for ATC code and cache in session state
    # ---------------------------
        df_drugbank_atc = get_drugbank_by_atc(atc_code)
        st.session_state.df_drugbank_atc = df_drugbank_atc

        if not df_drugbank_atc.empty:
            st.markdown("---")
            st.markdown(
                f"## ADMET Properties of Compounds for ATC {atc_code.upper()}"
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

            if "BBB_Martins" in df_drugbank_atc.columns and "BBB_Martins_Safe" not in df_drugbank_atc.columns:
                df_drugbank_atc["BBB_Martins_Safe"] = 1 - df_drugbank_atc["BBB_Martins"]
            st.session_state.adme_atc_df = df_drugbank_atc[admet_cols].copy()
    

    # =========================================================
    # REFERENCE SET VARIABILITY ANALYSIS
    # =========================================================

    reference_df = pd.DataFrame()

    # ---------------------------------------------------------
    # Select reference set automatically
    # ---------------------------------------------------------

    if (
        "adme_chembl_df" in st.session_state
        and not st.session_state.adme_chembl_df.empty
    ):

        reference_df = st.session_state.adme_chembl_df.copy()

        reference_source = "ChEMBL"

    elif (
        "adme_atc_df" in st.session_state
        and not st.session_state.adme_atc_df.empty
    ):

        reference_df = st.session_state.adme_atc_df.copy()

        reference_source = "DrugBank"


    # ---------------------------------------------------------
    # Run variability analysis
    # ---------------------------------------------------------

    if not reference_df.empty:

        variability_df = compute_reference_variability(
            reference_df
        )

        st.markdown(
            f"#### Variability Analysis — {reference_source}"
        )

        st.markdown(
            "Properties are ranked from lowest to highest "
            "relative variability in the reference set."
        )
        st.info(
            """
            **How to interpret this analysis**

            • **Variability:** describes how much a property varies among the reference compounds.
            Lower variability means that the reference compounds have more similar values for that property.

            • **Reference range:** represents the P10–P90 range of the reference compounds.
            A narrow range indicates that the property is more constrained within the reference set.

            • **CV (Coefficient of Variation):** measures relative variability.
            Lower CV values indicate greater consistency among the reference compounds.

            • **IQR:** measures the spread of the middle 50% of the reference compounds.
            Lower IQR values indicate a narrower distribution.

            **Important:** Low variability does not necessarily mean that a property is more biologically important.
            These results should be used as a guide for identifying potential constraints and should be combined
            with biological knowledge and expert judgment.
            """
        )

        st.dataframe(
            variability_df[
                [
                    "property",
                    "Variability",
                    "Reference range",
                    "CV",
                    "IQR",
                    "normalized_IQR"
                ]
            ],
            use_container_width=True
        )
        
        st.caption(
        "A narrow reference range may indicate a potential constraint, "
        "but property selection should also consider the biological target, "
        "therapeutic indication, and expert knowledge."
        )
        # =====================================================
        # CORRELATION ANALYSIS
        # =====================================================

        st.markdown(
            "#### Property Correlation"
        )

        st.markdown(
            """
            Spearman correlation is used to identify relationships
            between ADMET properties. Strong correlations may indicate
            redundant information when selecting constraints.
            """
        )

        corr_df, high_corr_df = compute_reference_correlation(
            reference_df,
            threshold=0.80
        )

        # -----------------------------------------------------
        # Two-column layout
        # -----------------------------------------------------

        col1, col2 = st.columns(2)

        # -----------------------------------------------------
        # LEFT COLUMN — Highly correlated pairs
        # -----------------------------------------------------

        with col1:

            st.markdown(
                "##### Highly Correlated Properties"
            )

            if not high_corr_df.empty:

                st.dataframe(
                    high_corr_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.success(
                    "No highly correlated property pairs were "
                    "detected at |ρ| ≥ 0.80."
                )


        # -----------------------------------------------------
        # RIGHT COLUMN — Correlation heatmap
        # -----------------------------------------------------

        with col2:

            st.markdown(
                "##### Correlation Matrix"
            )

            fig = plot_reference_correlation(
                corr_df
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

# ============================================================
# ADMET PROPERTY SELECTION, WEIGHTING AND DESIRABILITY SCORING
# ============================================================

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
# Display ADMET profiling only if input + reference exist
# ------------------------------------------------------------
if (
    "adme_df" in st.session_state
    and not st.session_state.adme_df.empty
    and ref_key is not None
):

    ref_df = st.session_state[ref_key]
    input_adme_df = st.session_state.adme_df

    st.markdown("---")
    st.markdown(f"## Context-Aware Prioritization Using {ref_label}")

    is_single_molecule = len(input_adme_df) == 1
    # ========================================================
    # PROPERTY SELECTION
    # ========================================================

    with st.form("select_adme_props_form"):

        st.markdown("### 1. Select ADMET properties")

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

        submitted_props = st.form_submit_button(
            "Confirm property selection"
        )


    # --------------------------------------------------------
    # SAVE PROPERTY SELECTION
    # --------------------------------------------------------

    if submitted_props:

        if not selected_tmp:

            st.warning(
                "Please select at least one ADMET property."
            )

        else:

            st.session_state.selected_adme_props = selected_tmp

            # New property selection requires new weights
            st.session_state.weights_confirmed = False

            # Keep existing weights when possible
            st.session_state.adme_weights = {
                prop: st.session_state.adme_weights.get(
                    prop,
                    1.0
                )
                for prop in selected_tmp
            }

            # Clear previous desirability results
            st.session_state.pop(
                "desirability_df",
                None
            )

            st.session_state.pop(
                "desirability_geo_df",
                None
            )

            st.session_state.pop(
                "desirability_ref_df",
                None
            )

            st.success(
                "ADMET properties selected."
            )


    # ========================================================
    # WEIGHTS
    # ========================================================

    if st.session_state.selected_adme_props:

        st.markdown(
            "### 2. Assign weights to selected properties"
        )

        if not st.session_state.get(
            "weights_confirmed",
            False
        ):

            st.info(
                "Assign the relative importance of each property."
            )

            with st.form("assign_adme_weights_form"):

                weights_tmp = {}

                for prop in st.session_state.selected_adme_props:

                    weight = st.slider(
                        f"Weight for {prop}",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(
                            st.session_state.adme_weights.get(
                                prop,
                                1.0
                            )
                        ),
                        step=0.05,
                        key=f"w_{prop}",
                    )

                    weights_tmp[prop] = weight

                submitted_weights = st.form_submit_button(
                    "Confirm weights"
                )


            # ----------------------------------------------------
            # SAVE WEIGHTS
            # ----------------------------------------------------

            if submitted_weights:

                # Prevent all weights = 0
                if sum(weights_tmp.values()) <= 0:

                    st.error(
                        "At least one property must have a weight greater than 0."
                    )

                else:

                    st.session_state.adme_weights = weights_tmp

                    st.session_state.weights_confirmed = True

                    # Clear previous results
                    st.session_state.pop(
                        "desirability_df",
                        None
                    )

                    st.session_state.pop(
                        "desirability_geo_df",
                        None
                    )

                    st.session_state.pop(
                        "desirability_ref_df",
                        None
                    )

                    st.success(
                        "ADMET weights saved successfully."
                    )

        else:

            st.success(
                "ADMET weights confirmed."
            )


    # ========================================================
    # DESIRABILITY SCORING
    # ========================================================

    selected_ui_props = st.session_state.get(
        "selected_adme_props",
        []
    )

    all_ui_weights = st.session_state.get(
        "adme_weights",
        {}
    )

    weights_ready = st.session_state.get(
        "weights_confirmed",
        False
    )


    # --------------------------------------------------------
    # Convert UI properties → internal property names
    # --------------------------------------------------------

    selected_internal_props = [
        map_columns[p]
        for p in selected_ui_props
        if p in map_columns
    ]


    # --------------------------------------------------------
    # Convert UI weights → internal property names
    # --------------------------------------------------------

    internal_weights = {
        map_columns[k]: v
        for k, v in all_ui_weights.items()
        if k in map_columns
    }


    # --------------------------------------------------------
    # Keep only selected properties
    # --------------------------------------------------------

    filtered_weights = {
        prop: internal_weights[prop]
        for prop in selected_internal_props
        if prop in internal_weights
    }


    # --------------------------------------------------------
    # Design phase
    # --------------------------------------------------------

    is_single_molecule = (
        len(input_adme_df) == 1
    )

    is_hit_phase = (
        design_phase == "Hit identification"
    )

    is_geo_phase = (
        design_phase
        in [
            "Lead optimization",
            "Candidate selection"
        ]
    )


    # --------------------------------------------------------
    # Enable desirability
    # --------------------------------------------------------

    enable_desirability = (
        bool(selected_internal_props)
        and bool(filtered_weights)
        and weights_ready
        and (is_hit_phase or is_geo_phase)
    )


    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    with st.expander(
        "Desirability calculation status",
        expanded=False
    ):

        st.write(
            "Selected properties:",
            selected_ui_props
        )

        st.write(
            "Internal properties:",
            selected_internal_props
        )

        st.write(
            "Weights:",
            filtered_weights
        )

        st.write(
            "Weights confirmed:",
            weights_ready
        )

        st.write(
            "Design phase:",
            design_phase
        )

        st.write(
            "Input molecules:",
            len(input_adme_df)
        )

        st.write(
            "Desirability enabled:",
            enable_desirability
        )


    # ========================================================
    # RUN DESIRABILITY
    # ========================================================

    if enable_desirability:

        try:

            # ------------------------------------------------
            # Normalize weights
            # ------------------------------------------------

            weights = normalize_weights(
                filtered_weights
            )


            # ------------------------------------------------
            # Filter PROPERTY_CONFIG
            # ------------------------------------------------

            filtered_config = {
                k: v
                for k, v in PROPERTY_CONFIG.items()
                if k in selected_internal_props
            }


            # ------------------------------------------------
            # Check configuration
            # ------------------------------------------------

            missing_config = [
                p
                for p in selected_internal_props
                if p not in filtered_config
            ]

            if missing_config:

                st.error(
                    f"Properties missing from PROPERTY_CONFIG: "
                    f"{missing_config}"
                )

            else:

                # --------------------------------------------
                # Build ranges from reference set
                # --------------------------------------------

                ranges = prepare_ranges_from_reference(
                    ref_df,
                    filtered_config
                )


                # =================================================
                # HIT IDENTIFICATION
                # =================================================

                if is_hit_phase:

                    if not use_loaded_values:

                        desirability_df = compute_desirability(
                            inputs=input_adme_df,
                            ranges=ranges,
                            weights=weights,
                            config=filtered_config,
                        )

                        desirability_ref_df = compute_desirability(
                            inputs=ref_df,
                            ranges=ranges,
                            weights=weights,
                            config=filtered_config,
                        )

                        st.session_state.desirability_df = (
                            desirability_df
                        )

                        st.session_state.desirability_ref_df = (
                            desirability_ref_df
                        )


                # =================================================
                # LEAD / CANDIDATE
                # =================================================

                elif is_geo_phase:

                    if not use_loaded_values:

                        desirability_geo_df = (
                            compute_desirability_geometric(
                                inputs=input_adme_df,
                                ranges=ranges,
                                weights=weights,
                                config=filtered_config,
                            )
                        )

                        desirability_ref_df = (
                            compute_desirability_geometric(
                                inputs=ref_df,
                                ranges=ranges,
                                weights=weights,
                                config=filtered_config,
                            )
                        )

                        st.session_state.desirability_geo_df = (
                            desirability_geo_df
                        )

                        st.session_state.desirability_ref_df = (
                            desirability_ref_df
                        )


                st.success(
                    "Desirability analysis completed."
                )


        except Exception as e:

            st.error(
                f"Error computing desirability: {e}"
            )
        # ========================================================
        # SUMMARY DASHBOARD
        # ========================================================
        if (
            ref_key is not None
            and not st.session_state.adme_df.empty
        ):

            ref_df = st.session_state[ref_key]
            input_adme_df = st.session_state.adme_df

            st.markdown("### Summary Dashboard")

            # -----------------------------
            # DESIRABILITY DETECTION
            # -----------------------------
            df_des = None

            if "desirability_df" in st.session_state and not st.session_state.desirability_df.empty:
                df_des = st.session_state.desirability_df

            elif "desirability_geo_df" in st.session_state and not st.session_state.desirability_geo_df.empty:
                df_des = st.session_state.desirability_geo_df


            # -----------------------------
            # SCORE COLUMN
            # -----------------------------
            avg_score = None
            ref_avg_score = None
            ref_q1_score = None
            score_col = None

            if df_des is not None:

                score_col = next(
                    (c for c in df_des.columns if "Desirability" in c),
                    None
                )

                if score_col:

                    # Input compounds
                    avg_score = df_des[score_col].mean()

                    # Reference compounds
                    if (
                        "desirability_ref_df" in st.session_state
                        and not st.session_state.desirability_ref_df.empty
                        and score_col in st.session_state.desirability_ref_df.columns
                    ):

                        ref_scores = st.session_state.desirability_ref_df[score_col]

                        ref_avg_score = ref_scores.mean()
                        ref_q1_score = ref_scores.quantile(0.25)

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric("Input compounds", len(input_adme_df))
            col2.metric("Reference compounds", len(ref_df))

            col3.metric(
                "Input Avg",
                round(avg_score, 3) if avg_score is not None else "N/A"
            )

            col4.metric(
                "Reference Avg",
                round(ref_avg_score, 3) if ref_avg_score is not None else "N/A"
            )

            col5.metric(
                "Reference Q1",
                round(ref_q1_score, 3) if ref_q1_score is not None else "N/A"
            )

            # ========================================================
            # PLOT
            # ========================================================
            if df_des is not None and score_col:

                st.markdown("#### Desirability scores")


                fig = go.Figure()

                # ==============================
                # INPUT COMPOUNDS
                # ==============================
                df_input_plot = df_des.copy().sort_values(by=score_col, ascending=False)

                if "ID" in df_input_plot.columns:
                    x_input = df_input_plot["ID"].astype(str)
                else:
                    x_input = df_input_plot.index.astype(str)

                y_input = df_input_plot[score_col]

                fig.add_trace(go.Scatter(
                    x=x_input,
                    y=y_input,
                    mode='lines+markers',
                    name='Input compounds',
                    line=dict(color='royalblue', width=3),
                    marker=dict(size=8, color='royalblue'),
                    fill='tozeroy',
                    fillcolor='rgba(65,105,225,0.15)'
                ))

                # ==============================
                # REFERENCE COMPOUNDS
                # ==============================
                if "desirability_ref_df" in st.session_state:

                    df_ref_plot = st.session_state.desirability_ref_df.copy()

                    df_ref_plot = df_ref_plot.sort_values(by=score_col, ascending=False)

                    if "ID" in df_ref_plot.columns:
                        x_ref = df_ref_plot["ID"].astype(str)
                    else:
                        x_ref = df_ref_plot.index.astype(str)

                    y_ref = df_ref_plot[score_col]

                    fig.add_trace(go.Scatter(
                        x=x_ref,
                        y=y_ref,
                        mode='lines+markers',
                        name='Reference compounds',
                        line=dict(color='firebrick', width=2, dash='dash'),
                        marker=dict(size=6, color='firebrick'),
                        fill='tozeroy',
                        fillcolor='rgba(178,34,34,0.10)'
                    ))

                # ==============================
                # MERGE INPUT + REFERENCE
                # ==============================
                df_input_plot = df_des.copy()
                df_input_plot["Dataset"] = "Input"

                df_ref_plot = None
                if "desirability_ref_df" in st.session_state:
                    df_ref_plot = st.session_state.desirability_ref_df.copy()
                    df_ref_plot["Dataset"] = "Reference"

                # ------------------------------
                # CONCAT
                # ------------------------------
                df_all = pd.concat(
                    [df_input_plot, df_ref_plot],
                    ignore_index=True
                )

                # ------------------------------
                # Name X
                # ------------------------------
                def get_name(row):
                    if row["Dataset"] == "Reference":
                        if pd.notna(row.get("name")):
                            return str(row["name"])
                        elif pd.notna(row.get("molecule_name")):
                            return str(row["molecule_name"])
                        elif pd.notna(row.get("ID")):
                            return str(row["ID"])
                        else:
                            return str(row.name)

                    # Input compounds
                    if pd.notna(row.get("ID")):
                        return str(row["ID"])
                    else:
                        return str(row.name)


                df_all["Compound_name"] = df_all.apply(get_name, axis=1)

                # ==============================
                # ORDER BY SCORE
                # ==============================
                df_all = df_all.sort_values(by=score_col, ascending=False).reset_index(drop=True)

                # eje X numérico común
                df_all["x_pos"] = np.arange(len(df_all))

                # ==============================
                # SPLIT DATASETS
                # ==============================
                df_input_final = df_all[df_all["Dataset"] == "Input"]
                df_ref_final = df_all[df_all["Dataset"] == "Reference"]

                fig = go.Figure()

                # ------------------------------
                # INPUT
                # ------------------------------
                fig.add_trace(go.Scatter(
                    x=df_input_final["x_pos"],
                    y=df_input_final[score_col],
                    mode='lines+markers',
                    name='Input compounds',
                    line=dict(color='royalblue', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(65,105,225,0.15)'
                ))

                # ------------------------------
                # REFERENCE
                # ------------------------------
                fig.add_trace(go.Scatter(
                    x=df_ref_final["x_pos"],
                    y=df_ref_final[score_col],
                    mode='lines+markers',
                    name='Reference compounds',
                    line=dict(color='firebrick', width=2, dash='dash'),
                    marker=dict(size=6),
                    fill='tozeroy',
                    fillcolor='rgba(178,34,34,0.10)'
                ))

                # ==============================
                # REFERENCE Q1 THRESHOLD
                # ==============================
                if ref_q1_score is not None:

                    fig.add_hline(
                        y=ref_q1_score,
                        line_dash="dot",
                        line_color="darkgreen",
                        line_width=2,
                        annotation_text=f"Reference Q1 = {ref_q1_score:.3f}",
                        annotation_position="top right"
                    )

                # ==============================
                # X and Y AXIS CONFIGURATION
                # ==============================
                fig.update_layout(
                    xaxis=dict(
                        title="Compound",
                        tickmode='array',
                        tickvals=df_all["x_pos"],
                        ticktext=df_all["Compound_name"],
                        tickangle=45
                    ),
                    yaxis_title="Desirability",
                    template="simple_white",
                    hovermode="x unified",
                    legend=dict(title="Dataset")
                )

                st.plotly_chart(fig, use_container_width=True) 

                            # -------- pie de figura --------
                st.markdown(
                    """
                    *Desirability scores for the input compounds (blue) and the reference compounds (red), 
                    ranked from highest to lowest score. The dotted horizontal line indicates the first quartile (Q1) of the reference compounds. 
                    Using this value as the prioritization threshold would recover approximately 75% of the reference compounds.*
                    """
                )

            # ========================================================
            # SHOW DESIRABILITY TABLE (FINAL OUTPUT)
            # ========================================================
            if df_des is not None:

                # ----------------------------------------------------
                # Identify desirability type
                # ----------------------------------------------------

                if (
                    "desirability_df" in st.session_state
                    and df_des is st.session_state.desirability_df
                ):
                    desirability_title = (
                        "Linear desirability results (Hit identification)"
                    )

                elif (
                    "desirability_geo_df" in st.session_state
                    and df_des is st.session_state.desirability_geo_df
                ):
                    desirability_title = (
                        "Geometric desirability results (Lead/Candidate stage)"
                    )

                else:
                    desirability_title = "Desirability results"

                # ====================================================
                # TWO-COLUMN LAYOUT
                # ====================================================

                col1, col2 = st.columns(2)

                # ====================================================
                # COLUMN 1 → DESIRABILITY TABLE
                # ====================================================

                with col1:

                    st.markdown(
                        f"#### {desirability_title}"
                    )

                    st.dataframe(
                        df_des,
                        use_container_width=True,
                        hide_index=True
                    )

                # ====================================================
                # COLUMN 2 → DESIRABILITY HEATMAP
                # ====================================================

                with col2:

                    st.markdown("#### Desirability Profile")

                    if (
                        st.session_state.get("use_example", False)
                        and os.path.exists(
                            "example/Linear_desirability_heatmap.png"
                        )
                    ):

                        st.image(
                            "example/Linear_desirability_heatmap.png",
                            use_container_width=True
                        )

                    else:

                        # --------------------------------------------
                        # Generate heatmap
                        # --------------------------------------------

                        df_heatmap = plot_desirability_heatmap(
                            df_des
                        )

                        fig = render_heatmap(
                            df_heatmap
                        )

                        st.pyplot(
                            fig,
                            clear_figure=True
                        )

                        plt.close(fig)

            else:

                st.info(
                    "Provide a reference dataset (ChEMBL or ATC) "
                    "and input molecules to compute similarity."
                )



        #========================== CHEMICAL SIMILARITY =============================

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
        # Execute chemical similarity workflow only if reference dataset and input molecules are available
        # ==========================================================

        if df_ref is not None and smiles_list:

            st.markdown("## Chemical Similarity (ChEMBL / ATC)")

            df_proc = df_ref[[smiles_col, id_col]].dropna().copy()

            processed = df_proc.apply(process_molecule_row, axis=1, result_type="expand")
            df_proc = pd.concat([df_proc, processed], axis=1)

            df_proc = df_proc[df_proc["error"].isna()].copy()

            df_proc["curated_smiles"] = df_proc["curated_smiles"].astype(str)
            df_proc[id_col] = df_proc[id_col].astype(str)

            df_proc = df_proc.drop_duplicates(
                subset=[id_col, "curated_smiles"]
            ).reset_index(drop=True)

        # ==========================================================
        # Choose similarity workflow based on input type:
        # ==========================================================

        # -------- CASE 1: Multiple molecules (CSV input) --------
        if (
            "input_df" in st.session_state
            and isinstance(st.session_state.input_df, pd.DataFrame)
        ):

            df_query = st.session_state.input_df.copy()

            processed_q = df_query.apply(
                process_molecule_row,
                axis=1,
                result_type="expand"
            )

            df_query = pd.concat(
                [df_query, processed_q],
                axis=1
            )

            df_query = df_query[
                df_query["error"].isna()
            ].copy()

            df_query["curated_smiles"] = (
                df_query["curated_smiles"].astype(str)
            )

            # ==========================================================
            # SIMILARITY ANALYSIS
            # ==========================================================

            if len(df_query) > 1:

                # ======================================================
                # TWO-COLUMN LAYOUT
                # ======================================================

                col1, col2 = st.columns(2)

                # ======================================================
                # COLUMN 1 → HEATMAP
                # ======================================================

                with col1:

                    st.markdown("### Similarity Heatmap")

                    if not use_loaded_values:

                        # plot_heatmap_similitud calculates and displays
                        # the heatmap and returns the similarity matrix
                        df_sim_matrix = plot_heatmap_similitud(
                            df_query,
                            df_proc,
                            smiles_col="curated_smiles",
                            id_col_query="ID",
                            id_col_ref=id_col
                        )

                    else:

                        df_sim_matrix = st.session_state["df_sim_matrix"]

                        st.image(
                            "example/heatmap.png",
                            use_container_width=True
                        )

                    st.markdown(
                        """
                        *Structural similarity between query and reference
                        compounds based on Morgan fingerprints (radius = 2,
                        2048 bits) and the Tanimoto coefficient.*
                        """
                    )

                # ======================================================
                # COLUMN 2 → SIMILARITY MATRIX
                # ======================================================

                with col2:

                    st.markdown("### Similarity Matrix")

                    # ------------------------------------------
                    # Reference similarity columns
                    # ------------------------------------------

                    sim_cols = [
                        c for c in df_sim_matrix.columns
                        if c != "Mean_Similarity"
                    ]

                    # ------------------------------------------
                    # Highlight maximum similarity
                    # ------------------------------------------

                    def highlight_max_ref(row):

                        max_val = row[sim_cols].max()

                        return [
                            (
                                "background-color: #2E7D32; "
                                "color: white; "
                                "font-weight: bold;"
                            )
                            if (
                                col in sim_cols
                                and val == max_val
                            )
                            else ""
                            for col, val in row.items()
                        ]

                    styled_df = df_sim_matrix.style.apply(
                        highlight_max_ref,
                        axis=1
                    )

                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown(
                        """
                        *The highlighted value represents the most structurally
                        similar reference compound for each input molecule.*
                        """
                    )

            else:

                st.info(
                    "Similarity clustering requires at least two input molecules."
                )
        # -------- CASE 2: Single molecule (manual input) --------
        else:

            input_df = pd.DataFrame({"smiles": [smiles_list[0]]})

            input_proc = input_df.apply(process_molecule_row, axis=1, result_type="expand")

            if not input_proc["error"].isna().iloc[0]:

                st.error("Input SMILES could not be standardized.")

            else:

                curated_input = str(input_proc["curated_smiles"].iloc[0])

                if "similarity_df" not in st.session_state:

                    with st.spinner("Calculating chemical similarity..."):
                        st.session_state.similarity_df = calcular_similitud(
                            input_smiles=curated_input,
                            df_ref=df_proc,
                            smiles_col="curated_smiles",
                            id_col=id_col
                        )

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
        # RADAR PLOTS 
        # ============================================================

        st.markdown("---")
        st.markdown("## Compound Profile")

        if (
            ref_key is not None
            and not input_adme_df.empty
            and st.session_state.selected_adme_props
        ):

            # ========================================================
            # SELECT ADME COLUMNS
            # ========================================================

            selected_cols = [
                map_columns_perc[p]
                for p in st.session_state.selected_adme_props
                if p in map_columns_perc
            ]

            # ========================================================
            # VALIDATE REQUIRED COLUMNS
            # ========================================================

            missing_cols = [
                col
                for col in selected_cols
                if (
                    col not in input_adme_df.columns
                    or col not in ref_df.columns
                )
            ]

            if missing_cols:

                st.warning(
                    f"Missing required ADME columns: {missing_cols}"
                )

            else:

                # ====================================================
                # REFERENCE MIN / MAX
                # ====================================================

                min_df = pd.DataFrame(
                    [ref_df[selected_cols].min()]
                )

                max_df = pd.DataFrame(
                    [ref_df[selected_cols].max()]
                )

                # ====================================================
                # MOLECULE PAGINATION
                # ====================================================

                n = len(input_adme_df)

                molecules_per_page = 1

                total_pages = (
                    (n - 1) // molecules_per_page + 1
                )

                pages = list(range(total_pages))

                # IDs of input molecules
                molecule_ids = [
                    str(x)
                    for x in input_adme_df["ID"].tolist()
                ] if "ID" in input_adme_df.columns else [
                    f"Compound {i+1}"
                    for i in range(n)
                ]

                # ====================================================
                # PAGE SELECTOR
                # ====================================================

                page = st.pills(
                    "Select molecule",
                    options=pages,
                    default=0,
                    selection_mode="single",
                    format_func=lambda x: molecule_ids[x]
                )

                # Safety check
                if page is None:
                    page = 0

                # ====================================================
                # SELECT CURRENT MOLECULE
                # ====================================================

                start_idx = page * molecules_per_page
                end_idx = start_idx + molecules_per_page

                page_df = input_adme_df.iloc[
                    start_idx:end_idx
                ]

                # ====================================================
                # DISPLAY CURRENT MOLECULE
                # ====================================================

                for i, (_, row) in enumerate(
                    page_df.iterrows()
                ):

                    compound_name = str(
                        row.get(
                            "ID",
                            f"Compound {start_idx + i + 1}"
                        )
                    )

                    col1, col2 = st.columns(
                        [1, 1.8]
                    )

                    # ==================================================
                    # COLUMN 1 → MOLECULE
                    # ==================================================

                    with col1:

                        st.markdown(
                            f"### {compound_name}"
                        )

                        image_name = (
                            "comp_"
                            + compound_name.replace(".", "_")
                            + ".jpg"
                        )

                        example_path = (
                            f"example/{image_name}"
                        )

                        # ------------------------------
                        # Example image
                        # ------------------------------

                        if (
                            st.session_state.get(
                                "use_example",
                                False
                            )
                            and os.path.exists(
                                example_path
                            )
                        ):

                            st.image(
                                example_path,
                                use_container_width=True
                            )

                        # ------------------------------
                        # Generate molecule
                        # ------------------------------

                        else:

                            if "smiles" in input_adme_df.columns:

                                mol = Chem.MolFromSmiles(
                                    row["smiles"]
                                )

                                if mol:

                                    mol_img = (
                                        Chem.Draw.MolToImage(
                                            mol,
                                            size=(500, 500)
                                        )
                                    )

                                    st.image(
                                        mol_img,
                                        use_container_width=True
                                    )

                                else:

                                    st.error(
                                        "Could not generate "
                                        "molecule structure."
                                    )

                            else:

                                st.warning(
                                    "SMILES column not available."
                                )

                    # ==================================================
                    # COLUMN 2 → RADAR
                    # ==================================================

                    with col2:

                        st.markdown(
                            "### ADMET Profile"
                        )

                        radar_name = (
                            compound_name.replace(
                                ".", "_"
                            )
                            + ".png"
                        )

                        radar_path = (
                            f"example/{radar_name}"
                        )

                        # ------------------------------
                        # Create compound dataframe
                        # ------------------------------

                        comp_df = pd.DataFrame(
                            [row[selected_cols]]
                        )

                        comp_df["ID"] = (
                            compound_name
                        )

                        # ------------------------------
                        # Example radar
                        # ------------------------------

                        if (
                            st.session_state.get(
                                "use_example",
                                False
                            )
                            and os.path.exists(
                                radar_path
                            )
                        ):

                            st.image(
                                radar_path,
                                use_container_width=True
                            )

                        # ------------------------------
                        # Generate radar
                        # ------------------------------

                        else:

                            try:

                                fig = (
                                    plot_radar_with_min_max_df(
                                        min_df=min_df,
                                        max_df=max_df,
                                        compuestos_df=comp_df,
                                        selected_properties=(
                                            st.session_state
                                            .selected_adme_props
                                        )
                                    )
                                )

                                st.pyplot(
                                    fig,
                                    clear_figure=True
                                )

                                plt.close(fig)

                            except Exception as _e:

                                st.warning(
                                    f"Failed to generate radar "
                                    f"for {compound_name}: {_e}"
                                )
            # ============================================================
            # GLORYx METABOLITE PREDICTION
            # ============================================================

            st.markdown("#### Metabolite Prediction (GLORYx)")

            # ------------------------------------------------------------
            # Use the compound currently selected in Compound Profile
            # ------------------------------------------------------------

            if "smiles" in row.index and pd.notna(row["smiles"]):

                smi_for_met = str(row["smiles"])

                # --------------------------------------------------------
                # Initialize metabolite storage
                # --------------------------------------------------------

                if "metabolites_by_compound" not in st.session_state:
                    st.session_state.metabolites_by_compound = {}

                # --------------------------------------------------------
                # Check whether metabolites were already calculated
                # --------------------------------------------------------

                has_metabolites = (
                    compound_name
                    in st.session_state.metabolites_by_compound
                )

                # --------------------------------------------------------
                # Prediction button
                # --------------------------------------------------------

                if not has_metabolites:

                    if st.button(
                        "Predict Metabolites (GLORYx)",
                        key=f"predict_met_{compound_name}"
                    ):

                        with st.spinner(
                            f"Predicting metabolites for {compound_name}..."
                        ):

                            try:

                                mols, labels, df_met = (
                                    calcular_metabolitos(
                                        smi_for_met
                                    )
                                )

                                st.session_state.metabolites_by_compound[
                                    compound_name
                                ] = df_met

                                st.success(
                                    f"Metabolite prediction completed "
                                    f"for {compound_name}."
                                )

                            except Exception as e:

                                st.error(
                                    f"Error predicting metabolites: {e}"
                                )

                else:

                    st.info(
                        f"Metabolites already predicted for "
                        f"**{compound_name}**."
                    )

                # --------------------------------------------------------
                # Display results for current compound
                # --------------------------------------------------------

                if (
                    "metabolites_by_compound"
                    in st.session_state
                    and compound_name
                    in st.session_state.metabolites_by_compound
                ):

                    df_met_current = (
                        st.session_state
                        .metabolites_by_compound[
                            compound_name
                        ]
                    )

                    visualizar_metabolitos(
                        df_met_current
                    )

            else:

                st.warning(
                    "SMILES not available for metabolite prediction."
                )

    # ============================================================
    # WEIGHT PERTURBATION DEBUG (FOR TESTING PURPOSES)
    st.markdown("---")
    st.markdown("### Weight perturbation debug")

    if st.button("Run sensitivity test"):
        perturbed_sets, debug_df = perturb_weights_sensitivity(
            filtered_weights,
            perturbation=0.1
        )

        st.write("debug_df shape:", debug_df.shape)
        st.dataframe(debug_df)

# ---------------------- Contact ----------------------
st.markdown("---")
st.markdown("© 2026 ADME-Tec · Developed by Nano]°[Biostructures RG · Tecnologico de Monterrey | [GitHub Repository](https://github.com/NanoBiostructuresRG/NanoBiostructuresRG.github.io)")

