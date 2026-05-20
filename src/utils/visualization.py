import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import src.utils.molecule_icon_generator as mig

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import src.utils.molecule_icon_generator as mig

def show_molecules(smiles_list):
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

            st.markdown("#### Molecule")
            st.code(smiles)

            cols = st.columns(2)

            # -------- 2D --------
            with cols[0]:
                st.markdown("**2D Structure**")
                Chem.rdDepictor.Compute2DCoords(mol)
                st.image(Draw.MolToImage(mol, size=(300, 300)))

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
        
        cols_per_row = 5

        for i in range(0, n, cols_per_row):
            row_smiles = smiles_list[i:i + cols_per_row]
            cols = st.columns(cols_per_row)

            for j, smiles in enumerate(row_smiles):
                with cols[j]:
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is None:
                            st.warning("Invalid SMILES")
                            continue

                        Chem.rdDepictor.Compute2DCoords(mol)
                        img = Draw.MolToImage(mol, size=(300, 300))

                        # -------- DISPLAY --------
                        st.image(img)
                        st.caption(f"Mol {i + j + 1}")
                        st.code(smiles)  
                    
                    except Exception as e:
                        st.error(f"Error: {e}")