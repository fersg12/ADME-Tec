import pandas as pd
import streamlit as st
from src.chemistry.standardizer_utils import process_molecule_row

def process_single_smiles(smiles):
    processed = process_molecule_row(pd.Series({"smiles": smiles}))
    if processed.get("error") is None:
        return [processed["curated_smiles"]]
    else:
        st.warning(f"SMILES input error: {processed.get('error')}")
        return []

def process_smiles_list(df):
    if 'smiles' not in df.columns:
        st.error("CSV must contain a 'smiles' column.")
        return []
    raw_df = df[['smiles']].dropna()
    processed_df = raw_df.apply(process_molecule_row, axis=1, result_type='expand')
    valid_df = processed_df[processed_df['error'].isna()]
    smiles_list = valid_df['curated_smiles'].tolist()
    st.success(f"Loaded {len(smiles_list)} molecules.")
    return smiles_list
