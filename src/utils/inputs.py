import streamlit as st
import pandas as pd
import cirpy
from src.utils.processing import process_single_smiles, process_smiles_list

def molecule_input():
    """
    Handles molecule input either from a single entry or a CSV file.
    Returns:
        smiles_list (list)
        input_df (DataFrame or None)
    """

    input_mode = st.radio(
        "Select input mode", 
        ['Single molecule', 'From CSV file']
    )

    smiles_list = []
    input_df = None  # ← importante

    if input_mode == 'Single molecule':
        input_type = st.selectbox("Input type", ['SMILES', 'Name'])
        
        if input_type == 'SMILES':
            smiles = st.text_input(
                "Enter SMILES", 
                placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
            )
            if smiles:
                smiles_list = process_single_smiles(smiles)

        else:  # Input by molecule name
            name = st.text_input(
                "Enter molecule name", 
                placeholder="e.g., paracetamol"
            )
            if name:
                resolved = cirpy.resolve(name, 'smiles')
                if resolved:
                    smiles_list = process_single_smiles(resolved)
                else:
                    st.warning("Could not resolve name to SMILES.")

    else:  # From CSV
        uploaded_file = st.file_uploader(
            "Upload CSV with 'SMILES' and optional 'ID' column", 
            type="csv"
        )

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)

                if "smiles" not in df.columns:
                    st.error("CSV must contain a 'smiles' column.")
                    return [], None

                smiles_list = process_smiles_list(df)
                input_df = df  

            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    return smiles_list, input_df

