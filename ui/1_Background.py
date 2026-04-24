import streamlit as st

st.set_page_config(page_title="Background", layout="wide")

st.title("Background")

st.markdown(
"""
        <div style="text-align: justify;">

        <h4>Required Input</h4>
        <ul>
            <li><strong>SMILES strings:</strong> Input compounds in SMILES format (single or multiple compounds via CSV upload).</li>
            <li><strong>CHEMBL ID:</strong> Optional, to retrieve mechanism of action and biological target information.</li>
            <li><strong>ATC Code:</strong> Optional, to relate compounds to specific therapeutic areas.</li>
            <li><strong>Drug Design Phase:</strong> Optional, to indicate project stage (Hit identification, Lead optimization, Candidate selection).</li>
            <li><strong>Target Location:</strong> Optional, to indicate the site of action (Extracellular, Intracellular, Crosses BBB).</li>
        </ul>

        <h4>Functionalities</h4>
        <ol>
            <li><strong>ADME Prediction:</strong> Computes physicochemical and ADME properties for input compounds using the ADMET-AI model.</li>
            <li><strong>ChEMBL Integration:</strong> Retrieves compounds and action types for a selected target, predicts ADME properties, and compares with DrugBank references.</li>
            <li><strong>Radar Plot Visualization:</strong> Interactive comparison of ADME profiles for selected properties.</li>
            <li><strong>Metabolite Prediction (GLORYx):</strong> Predicts Phase I and II metabolites for selected compounds and displays them as structures and tables.</li>
            <li><strong>Data Export:</strong> Download predicted ADME properties, metadata, and metabolite results as CSV files.</li>
            <li><strong>Session Management:</strong> Tracks user inputs and predictions for interactive exploration without re-running calculations.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
)