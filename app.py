import streamlit as st
from streamlit_app.ui.sidebar import render_sidebar
from streamlit_app.utils.session import initialize_session

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Privacy Expert",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session initialization ────────────────────────────────────────────────────
initialize_session()

# ── Sidebar (siempre visible) ─────────────────────────────────────────────────
render_sidebar()

# ── Flujo principal: carga del agente -> chat ──────────────────────────────────
if "agent_bundle" not in st.session_state:
    # Primera carga: mostrar pantalla de inicialización
    from streamlit_app.ui.loading import render_loading_screen
    render_loading_screen()
else:
    # Agente listo: mostrar interfaz de chat
    from streamlit_app.ui.chat_interface import render_chat_interface
    render_chat_interface()
