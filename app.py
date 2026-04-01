import streamlit as st
from src.ui.chat_interface import render_chat_interface
from src.ui.sidebar import render_sidebar
from src.utils.session import initialize_session

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Expert Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session initialization ───────────────────────────────────────────────────
initialize_session()

# ── Render sidebar ───────────────────────────────────────────────────────────
render_sidebar()

# ── Render main chat interface ───────────────────────────────────────────────
render_chat_interface()
