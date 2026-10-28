"""ADMET prediction utilities for ADME-Tec."""

import argparse
import pandas as pd
import streamlit as st
import torch
from admet_ai import ADMETModel
from prop_profiler import configure_logging, profile_molecules



# Compatibility patch for PyTorch >= 2.6.
# Preserve the original loader to avoid wrapping it repeatedly.
if not getattr(torch.load, "_admetec_patched", False):
    _original_torch_load = torch.load

    def safe_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        with torch.serialization.safe_globals([argparse.Namespace]):
            return _original_torch_load(*args, **kwargs)

    safe_torch_load._admetec_patched = True
    torch.load = safe_torch_load


@st.cache_resource
def load_admet_model():
    """Load and cache the ADMET-AI model."""
    return ADMETModel()


def predict_admet(smiles_list, input_df=None):
    """
    Predict ADMET properties and calculate molecular properties
    using Prop_profile.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to evaluate.
    input_df : pandas.DataFrame, optional
        Original input dataframe. If it contains an ID column,
        the identifiers are retained in the prediction results.

    Returns
    -------
    pandas.DataFrame
        ADMET predictions and Prop_profile molecular properties,
        with SMILES and optional IDs.
    """
    if not smiles_list:
        return pd.DataFrame()

    # 1. Predict ADMET properties using ADMET-AI.
    model = load_admet_model()
    adme_results = model.predict(smiles_list)
    adme_df = pd.DataFrame(adme_results)

    # Complementary BBB score: higher values indicate lower
    # predicted BBB penetration.
    if "BBB_Martins" in adme_df.columns:
        adme_df["BBB_Martins_Safe"] = 1 - adme_df["BBB_Martins"]

    # Preserve the correspondence between molecules and predictions.
    adme_df.insert(0, "smiles", smiles_list)

    # 2. Calculate complementary properties using Prop_profile.
    configure_logging()
    profile_results = profile_molecules(smiles_list)

    if isinstance(profile_results, pd.DataFrame):
        profile_df = profile_results.copy()
    else:
        profile_df = pd.DataFrame(profile_results)

    # Reset indices to preserve row correspondence.
    adme_df = adme_df.reset_index(drop=True)
    profile_df = profile_df.reset_index(drop=True)

    # Validate that both tools returned the same number of molecules.
    if len(profile_df) != len(adme_df):
        raise ValueError(
            "The number of Prop_profile results does not match "
            "the number of ADMET-AI predictions."
        )

    # Remove duplicate identifiers from Prop_profile.
    if "smiles" in profile_df.columns:
        profile_df = profile_df.drop(columns=["smiles"])

    if "ID" in profile_df.columns:
        profile_df = profile_df.drop(columns=["ID"])

    # Prefix all Prop_profile columns to prevent naming conflicts.
    profile_df = profile_df.rename(
        columns={
            col: f"prop_profile_{col}"
            for col in profile_df.columns
        }
    )

    # Combine ADMET-AI predictions with Prop_profile properties.
    adme_df = pd.concat([adme_df, profile_df], axis=1)

    # Preserve original IDs when available.
    if (
        input_df is not None
        and "ID" in input_df.columns
        and len(input_df) == len(adme_df)
    ):
        if "ID" in adme_df.columns:
            adme_df = adme_df.drop(columns=["ID"])

        adme_df["ID"] = input_df["ID"].values

    # Verify that no duplicate column names remain.
    if adme_df.columns.duplicated().any():
        duplicates = adme_df.columns[
            adme_df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate columns remain in prediction results: {duplicates}"
        )

    return adme_df
