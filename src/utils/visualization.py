import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import src.utils.molecule_icon_generator as mig

def show_molecules(smiles_list):
    """Displays 2D and optional 3D structures for each molecule."""
    st.markdown("### Molecule Visualization")
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.warning(f"Invalid SMILES: {smiles}")
                continue

            st.markdown(f"#### Molecule {i+1}")
            st.code(smiles)

            cols = st.columns(2)
            with cols[0]:
                st.markdown("**2D Structure**")
                Chem.rdDepictor.Compute2DCoords(mol)
                st.image(Draw.MolToImage(mol, size=(300, 300)))

            with cols[1]:
                st.markdown("**3D Structure**")
                show_3d = True
                if len(smiles_list) > 1:
                    show_3d = st.checkbox("Show 3D Structure", value=False, key=f"3d_{i}")
                if show_3d:
                    try:
                        mol3d = mig.parse_structure(smiles, nice_conformation=True, dimension_3=True)
                        fig = mig.graph_3d(mol3d)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not render 3D structure: {e}")
        except Exception as e:
            st.error(f"Error processing molecule {i+1}: {e}")
