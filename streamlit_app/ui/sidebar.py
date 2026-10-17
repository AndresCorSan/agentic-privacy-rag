import streamlit as st
from streamlit_app.utils.session import new_conversation, switch_conversation, delete_conversation
from datetime import datetime


def render_sidebar():
    with st.sidebar:
        # Estilo para evitar que el texto de los botones salte de línea
        st.markdown("""
            <style>
                /* Solo para los botones de la lista de chats (que están en columnas) */
                [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button div p {
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    display: block;
                }
            </style>
        """, unsafe_allow_html=True)

        st.title("📑 Agentic Privacy RAG")

        # ── New conversation ───────────────────────────────────────────
        if st.button("✏️ Nueva conversación", use_container_width=True):
            new_conversation()
            st.rerun()

        st.divider()

        # ── Conversation history ───────────────────────────────────────
        convs = st.session_state.conversations
        current = st.session_state.current_conv_id

        if convs:
            st.caption("CONVERSACIONES RECIENTES")
            for cid, data in reversed(list(convs.items())):
                is_active = cid == current
                
                # Layout: Boton de chat + boton borrar (extra estrecho)
                col_btn, col_menu = st.columns([0.85, 0.15], gap="xxsmall")
                
                with col_btn:
                    if st.button(
                        f"{data['title']}",
                        key=f"btn_{cid}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        switch_conversation(cid)
                        st.rerun()
                
                with col_menu:
                    if st.button(
                        "✕", 
                        key=f"del_{cid}", 
                        help="Eliminar", 
                        use_container_width=True,
                        type="secondary"
                    ):
                        delete_conversation(cid)
                        st.rerun()


        st.divider()

        # ── Parametros del LLM ───────────────────────────────────────────────────────────
        st.caption("PARÁMETROS DEL MODELO")
        st.slider(
            "temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="temperature",
            help="Controla la creatividad de la respuesta. Recomendado: 0.2"
        )
        st.slider(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="top_p",
            help="Controla la diversidad de las palabras elegidas. Recomendado: 1.0"
        )
        st.number_input(
            "max_tokens",
            min_value=64,
            max_value=6000,
            step=64,
            key="max_tokens",
            help="Limita la longitud máxima de la respuesta. Recomendado: 1024 (límite 6000 TPM)."
        )
        st.toggle(
            "Streaming de respuesta",
            key="streaming",
            help="Muestra la respuesta del agente palabra por palabra en tiempo real."
        )

        st.divider()

        # ── Sistema RAG info ───────────────────────────────────────────────────
        st.caption("INFORMACIÓN DEL SISTEMA")
        
        with st.expander("🛠️ Especificaciones del Agente"):
            st.markdown("""
- **LLM:** `GPT-OSS 120B (Groq)`
- **Embeddings:** `E5-Base`
- **Sparse:** `BM25`
- **Reranker:** `BGE-v2-M3`
- **Vector DB:** `Qdrant`
- **Search:** `Tavily`
            """)

        # ── Corpus info ──────────────────────────────────────────────────────
        with st.expander("📚 Contenido del Corpus"):
            st.markdown("""
- **Políticas de Privacidad:** Google, Spotify
- **Términos de Servicio:** Google, Spotify
- **Guías:** GDPR, Cookies y Derechos
            """)

        
