import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import src.utils.molecule_icon_generator as mig


def show_molecules(smiles_list, ids=None):
    """Displays molecules in grid (2D) and optional 3D only for single molecule."""

    st.markdown("### Molecule Visualization")

    n = len(smiles_list)

    # ========================= SINGLE MOLECULE =========================
    if n == 1:
        smiles = smiles_list[0]

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.warning(f"Invalid SMILES: {smiles}")
                return

            cols = st.columns(2)

            # -------- 2D --------
            with cols[0]:
                st.markdown("**2D Structure**")
                Chem.rdDepictor.Compute2DCoords(mol)
                st.image(Draw.MolToImage(mol, size=(300, 300)))

                if ids is not None:
                    st.markdown(f"**ID:** {ids[0]}")

            # -------- 3D --------
            with cols[1]:
                st.markdown("**3D Structure**")
                show_3d = st.checkbox("Show 3D Structure", value=True)

                if show_3d:
                    try:
                        mol3d = mig.parse_structure(
                            smiles,
                            nice_conformation=True,
                            dimension_3=True
                        )
                        fig = mig.graph_3d(mol3d)
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Could not render 3D structure: {e}")

        except Exception as e:
            st.error(f"Error processing molecule: {e}")

    # ========================= MULTIPLE MOLECULES =========================
    else:

        molecules_per_page = 10
        cols_per_row = 5

        total_pages = (n - 1) // molecules_per_page + 1

        pages = list(range(1, total_pages + 1))

        page = st.pills(
            "Page",
            options=pages,
            default=1,
            selection_mode="single"
        )

        if page is None:
            page = 1

        start = (page - 1) * molecules_per_page
        end = min(start + molecules_per_page, n)

        page_smiles = smiles_list[start:end]

        for i in range(0, len(page_smiles), cols_per_row):

            row = page_smiles[i:i + cols_per_row]
            cols = st.columns(cols_per_row)

            for j, smiles in enumerate(row):

                with cols[j]:

                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            st.warning("Invalid SMILES")
                            continue

                        Chem.rdDepictor.Compute2DCoords(mol)
                        img = Draw.MolToImage(mol, size=(300, 300))

                        st.image(img, use_container_width=True)

                        idx = start + i + j

                        if ids is not None:
                            st.markdown(
                                f"""
                                <div style="text-align:center;
                                            font-size:0.85rem;
                                            font-weight:600;
                                            margin-top:-8px;">
                                    {ids[idx]}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption(f"Mol {idx + 1}")

                    except Exception as e:
                        st.error(str(e))