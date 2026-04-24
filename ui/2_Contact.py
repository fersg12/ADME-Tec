import streamlit as st
import smtplib
from email.mime.text import MIMEText

st.set_page_config(layout="wide")

# ===================== EMAIL FUNCTION =====================
def send_email(name, email, message):
    sender = "tu_email@gmail.com"
    password = "TU_APP_PASSWORD"  # ⚠️  App Password

    msg = MIMEText(f"""
    New message from ADME-Tec:

    Name: {name}
    Email: {email}

    Message:
    {message}
    """)

    msg["Subject"] = "ADME-Tec Contact Form"
    msg["From"] = sender
    msg["To"] = sender

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.send_message(msg)

# ===================== LAYOUT =====================
col1, col2 = st.columns([1, 1])

# -------- LEFT: CONTACT FORM --------
with col1:
    st.title("Contact")

    st.markdown(
        "Reach out for collaborations, questions, or feedback about **ADME-Tec**."
    )
    st.markdown("🌐 **Website:** [Nanobiostructures Research Group](https://nanobiostructuresrg.github.io/)")
    st.markdown("📞 **Phone:** +52 (81) 8358 2000")
    


    # ---- CARD STYLE ----
    with st.container():
        st.markdown("**Send us a message**")

        with st.form("contact_form"):
            name = st.text_input("Name")
            email = st.text_input("Email")
            message = st.text_area("Message")

            submitted = st.form_submit_button("Send message")

            if submitted:
                if name and email and message:
                    try:
                        send_email(name, email, message)
                        st.success("Message sent successfully!")
                    except Exception as e:
                        st.error("Error sending message. Check configuration.")
                else:
                    st.warning("Please fill all fields.")



# -------- RIGHT: LOCATION + IMAGE --------
with col2:
    st.subheader("Location")
    st.write("Tecnológico de Monterrey, Monterrey, Nuevo León, México")

    # MAP
    map_html = """
    <iframe
        src="https://www.google.com/maps?q=Tecnologico+de+Monterrey+Monterrey&output=embed"
        width="100%"
        height="250"
        style="border:0;"
        loading="lazy">
    </iframe>
    """
    st.components.v1.html(map_html, height=250)

    # IMAGE
    st.image(
        "https://cdn-dynmedia-1.microsoft.com/is/image/microsoftcorp/Tecnologico%20de%20Monterrey%20Header%20Image?resMode=sharp2&op_usm=1.5,0.65,15,0&wid=1900&hei=550&qlt=100&fmt=png-alpha&fit=crop",
        caption="Tecnológico de Monterrey - Monterrey Campus",
        use_container_width=True
    )