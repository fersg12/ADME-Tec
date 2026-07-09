import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("About ADMETEC")

# --- Custom CSS ---
st.markdown("""
<style>
.card {
    background-color: #e8f1faff;  
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 16px;
    border: 1px solid #E2E8F0;
}

.card h4 {
    margin-top: 0;
    color: #1E3A5F;  /* azul marino */
}

.card p, .card li {
    text-align: justify;
    color: #334155;
}
</style>
""", unsafe_allow_html=True)

# --- Cards ---
st.markdown("""
<div class="card">
    <h4>Overview</h4>
    <p>
    <strong>ADMETEC</strong> is a free, open-access web server for ADMET evaluation and compound prioritization. 
    It implements a context-aware approach that integrates target biology, subcellular localization, and therapeutic indication, 
    supporting more informed decision-making during early-stage drug discovery.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h4>Third Parties</h4>
    <p>
    <strong>Data handling:</strong> Uploaded files, including SMILES records, are used exclusively during request processing 
    and are not stored or retained after execution.<br>
    <strong>Toolkits and libraries:</strong><br>
        - ADME-AI (MIT License)<br>
        - GloryX (BSD 3-Clause License)<br>
        - RDKit (BSD 3-Clause License)<br>
        - Streamlit (BSD 3-Clause License)
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <h4>Software</h4>
    <p>
    F.I. Saldivar-Gonzáez. <em>LigandHub-API</em>. Version v0.1.14.<br>
    https://doi.org/10.5281/zenodo.20065340

</div>
""", unsafe_allow_html=True)