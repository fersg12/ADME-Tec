
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, QED
from rdkit.Chem.FilterCatalog import (
    FilterCatalog,
    FilterCatalogParams
)


def calculate_lipinski(mol):

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)

    violations = 0

    if mw > 500:
        violations += 1

    if logp > 5:
        violations += 1

    if hba > 10:
        violations += 1

    if hbd > 5:
        violations += 1

    return {
        "MW": mw,
        "logP": logp,
        "HBA": hba,
        "HBD": hbd,
        "violations": violations,
        "pass": violations == 0
    }


def calculate_qed(mol):

    return QED.qed(mol)


def calculate_structural_alerts(mol):

    catalogs = {
        "PAINS": FilterCatalogParams.FilterCatalogs.PAINS,
        "Brenk": FilterCatalogParams.FilterCatalogs.BRENK,
        "NIH": FilterCatalogParams.FilterCatalogs.NIH,
    }

    results = {}

    for name, catalog_type in catalogs.items():

        params = FilterCatalogParams()
        params.AddCatalog(catalog_type)

        catalog = FilterCatalog(params)

        matches = catalog.GetMatches(mol)

        results[name] = {
            "count": len(matches),
            "alerts": [
                match.GetDescription()
                for match in matches
            ]
        }

    results["Total"] = sum(
        item["count"]
        for item in results.values()
    )

    return results


def render_molecular_features(mol):

    if mol is None:
        return

    lipinski = calculate_lipinski(mol)
    qed = calculate_qed(mol)
    alerts = calculate_structural_alerts(mol)

    # --------------------------------------------------
    # FEATURE CARDS
    # --------------------------------------------------

    st.markdown(
        """
        <style>

        .mf-card {
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            padding: 10px 6px;
            text-align: center;
            min-height: 95px;
            background-color: white;
        }

        .mf-icon {
            font-size: 28px;
            line-height: 32px;
        }

        .mf-label {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }

        .mf-value {
            font-size: 15px;
            font-weight: 600;
            margin-top: 3px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    features = [

        (
            "✏",
            "Rule of Five",
            "Yes" if lipinski["pass"] else "No"
        ),

        (
            "⌖",
            "QED",
            f"{qed:.2f}"
        ),

        (
            "⚠",
            "Structural alerts",
            str(alerts["Total"])
        )
    ]

    for col, (icon, label, value) in zip(cols, features):

        with col:

            st.markdown(
                f"""
                <div class="mf-card">
                    <div class="mf-icon">{icon}</div>
                    <div class="mf-label">{label}</div>
                    <div class="mf-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------------------------------
    # LEARN MORE
    # --------------------------------------------------

    st.markdown("")

    if st.button(
        "LEARN MORE",
        key="molecular_features_learn_more"
    ):
        st.session_state["show_molecular_features"] = not st.session_state.get(
            "show_molecular_features",
            False
        )

    # --------------------------------------------------
    # DETAILS
    # --------------------------------------------------

    if st.session_state.get(
        "show_molecular_features",
        False
    ):

        st.markdown("#### Molecular details")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("**Lipinski Rule of Five**")

            st.write(
                f"MW: {lipinski['MW']:.2f} Da"
            )

            st.write(
                f"logP: {lipinski['logP']:.2f}"
            )

            st.write(
                f"HBA: {lipinski['HBA']}"
            )

            st.write(
                f"HBD: {lipinski['HBD']}"
            )

            st.write(
                f"Violations: {lipinski['violations']}"
            )

        with col2:

            st.markdown("**Structural alerts**")

            for catalog, data in alerts.items():

                if catalog == "Total":
                    continue

                st.write(
                    f"**{catalog}:** {data['count']}"
                )

                if data["count"] > 0:

                    for alert in data["alerts"]:
                        st.caption(
                            f"• {alert}"
                        )

            st.write(
                f"**Total alerts: {alerts['Total']}**"
            )