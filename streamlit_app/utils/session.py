import streamlit as st
import uuid
from datetime import datetime
from .summarize_title import generate_chat_title
from .database import init_db, save_conversation, save_message, load_all_conversations, delete_conversation_db

def initialize_session():
    """Initialize all session state variables and load history from DB."""
    init_db()

    # ── Conversations ──────────────────────────────────────────
    if "conversations" not in st.session_state:
        st.session_state.conversations = load_all_conversations()

    if "current_conv_id" not in st.session_state:
        # Siempre iniciar con una conversación nueva y limpia al abrir la app
        _new_conv()

    # ── Settings ───────────────────────────────────────────────
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "RAG Expert (default)"

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2

    if "top_p" not in st.session_state:
        st.session_state.top_p = 1.0

    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1024

    if "streaming" not in st.session_state:
        st.session_state.streaming = True


def _new_conv():
    """Create a new empty conversation only in session state (not in DB yet)."""
    conv_id = str(uuid.uuid4())[:8]
    title = "Nueva conversación"
    st.session_state.conversations[conv_id] = {
        "title": title,
        "messages": [],
        "created_at": datetime.now(),
    }
    st.session_state.current_conv_id = conv_id
    return conv_id


def new_conversation():
    """Public wrapper to start a new conversation."""
    return _new_conv()


def get_current_messages():
    """Return the message list for the active conversation."""
    cid = st.session_state.current_conv_id
    return st.session_state.conversations[cid]["messages"]


def add_message(role: str, content: str, sources: list = None, steps: list = None):
    """Append a message and persist to DB only if the conversation has content."""
    cid = st.session_state.current_conv_id
    conv = st.session_state.conversations[cid]
    
    # Añadir a la memoria RAM
    conv["messages"].append({
        "role": role,
        "content": content,
        "sources": sources or [],
        "steps": steps or [],
        "timestamp": datetime.now(),
    })
    
    # Persistencia perezosa: 
    # Si es el primer mensaje (lo envía el usuario), guardamos la cabecera de la conversación
    if len(conv["messages"]) == 1:
        save_conversation(cid, conv["title"])
    
    # Guardar mensaje en DB
    save_message(cid, role, content, sources, steps)

    # Auto-title: resumir y actualizar título en DB
    if role == "user" and conv["title"] == "Nueva conversación":
        title = generate_chat_title(content)
        conv["title"] = title
        save_conversation(cid, title)


def switch_conversation(conv_id: str):
    """Switch to an existing conversation."""
    st.session_state.current_conv_id = conv_id


def delete_conversation(conv_id: str):
    """Remove a conversation from session state and DB."""
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
        
        # Eliminar de DB
        delete_conversation_db(conv_id)

        # Si borramos la actual, necesitamos saltar a otra o crear una nueva
        if st.session_state.current_conv_id == conv_id:
            if st.session_state.conversations:
                # Elegir la última disponible
                st.session_state.current_conv_id = list(st.session_state.conversations.keys())[-1]
            else:
                _new_conv()
