import streamlit as st

pages = [
    st.Page("ADME-Tec_app.py", title="ADME-TEC", icon=":material/lab_research:"),
    st.Page("1_Background.py", title="About", icon=":material/menu_book:"),
    st.Page("2_Contact.py", title="Contact", icon=":material/email:"),
]

pg = st.navigation(pages, position="top")
pg.run()