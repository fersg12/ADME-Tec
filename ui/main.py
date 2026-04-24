import streamlit as st
ICON_RED = "images/icon_red.png"

pages = [
    st.Page("ADME-Tec_app.py", title="ADME-TEC"),
    st.Page("1_Backgroung.py", title="Background"),
    st.Page("2_Contact.py", title="Contact"),
]

pg = st.navigation(pages, position="top")
pg.run()