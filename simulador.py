import streamlit as st

# Configuración de la página
st.set_page_config(page_title="PÁGINA PARA PRUEBAS- Sitio en Construcción", page_icon="🚧")

# CSS personalizado para centrar y estilizar
st.markdown("""
    <style>
    .main {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 80vh;
        text-align: center;
    }
    .title {
        font-size: 3rem;
        color: #FFD700;
    }
    </style>
""", unsafe_allow_html=True)

# Contenido de la página
st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown("# 🚧 ¡Estamos trabajando!")
st.subheader("Esta página se encuentra actualmente en construcción.")
st.write("Estamos puliendo los detalles.")

# Puedes agregar una barra de progreso o un spinner
st.progress(65)
st.caption("Progreso del desarrollo: 65%")

st.markdown('</div>', unsafe_allow_html=True)
