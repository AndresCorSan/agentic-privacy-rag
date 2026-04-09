import streamlit as st
import time

from streamlit_app.utils.session import get_current_messages, add_message

# ── Suggestion prompts ────────────────────────────────────────────────────────
SUGGESTIONS = [
    {
        "icon": "🔒",
        "title": "Política de privacidad de Spotify",
        "subtitle": "¿Qué datos recopila Spotify sobre ti?",
        "prompt": "¿Qué datos personales recopila Spotify y cómo los utiliza?",
    },
    {
        "icon": "📋",
        "title": "Términos de servicio de Google",
        "subtitle": "¿Puede Google bloquear tu cuenta?",
        "prompt": "¿Puede Google bloquear o suspender una cuenta de usuario? ¿En qué casos?",
    },
    {
        "icon": "🇪🇺",
        "title": "Derechos GDPR",
        "subtitle": "Conoce tus derechos sobre tus datos",
        "prompt": "¿Qué derechos tienen los usuarios sobre sus datos personales según el GDPR?",
    },
    {
        "icon": "🍪",
        "title": "Qué son las cookies",
        "subtitle": "Para qué se utilizan realmente",
        "prompt": "¿Qué son las cookies y para qué se utilizan en los sitios web?",
    },
]


# ── Helpers para el agente ────────────────────────────────────────────────────

def _run_agent(question: str, history: list) -> tuple:
    """Lee el bundle del agente desde session_state y ejecuta la consulta."""
    from streamlit_app.agent import run_agent
    bundle = st.session_state["agent_bundle"]
    
    # Aplicar ajustes del modelo (temperatura, top_p, max_tokens)
    settings = {
        "temperature": st.session_state.get("temperature", 0.2),
        "top_p": st.session_state.get("top_p", 1.0),
        "max_tokens": st.session_state.get("max_tokens", 1024),
    }
    
    return run_agent(bundle, question, history, model_settings=settings)


# ── Render ────────────────────────────────────────────────────────────────────

def render_chat_interface():
    # ── Header ────────────────────────────────────────────────────────
    conv = st.session_state.conversations[st.session_state.current_conv_id]
    model_name = "Qwen3-32B · RAG + Qdrant"

    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.subheader(conv["title"])
    with col_badge:
        st.caption(f"⚡ {model_name}")

    st.divider()

    messages = get_current_messages()
    triggered_prompt = None

    # ── Welcome / empty state ──────────────────────────────────────────
    if not messages:
        st.markdown("## 👋 ¿En qué puedo ayudarte hoy?")
        st.write(
            "Soy tu asistente experto en **privacidad y protección de datos**, "
            "potenciado por un sistema RAG con búsqueda híbrida en documentos oficiales "
            "de Google, Spotify y guías sobre el GDPR."
        )

        st.write("")
        col1, col2 = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            col = col1 if i % 2 == 0 else col2
            with col:
                label = f"{s['icon']} **{s['title']}**\n\n_{s['subtitle']}_"
                if st.button(label, key=f"sug_{i}", use_container_width=True):
                    triggered_prompt = s["prompt"]

    else:
        # ── Render mensajes existentes ─────────────────────────────────
        for msg in messages:
            role = msg["role"]
            avatar = "streamlit_app/assets/agent_avatar.png" if role == "assistant" else "streamlit_app/assets/user_avatar.png"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])

                # Fuentes
                if msg.get("sources"):
                    with st.expander("📎 Fuentes consultadas", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(f"- `{src}`")

                # Pasos del agente (trazas)
                if msg.get("steps"):
                    _render_agent_steps(msg["steps"])

    # ── Chat input ─────────────────────────────────────────────────────
    user_input = st.chat_input("Escribe tu pregunta sobre privacidad…")
    final_input = triggered_prompt or user_input

    if final_input:
        add_message("user", final_input)
        with st.chat_message("user", avatar="streamlit_app/assets/user_avatar.png"):
            st.markdown(final_input)

        with st.chat_message("assistant", avatar="streamlit_app/assets/agent_avatar.png"):
            _stream_agent_response(final_input, messages)

        st.rerun()


def _render_agent_steps(steps: list):
    """Render colapsable con las trazas del agente."""
    if not steps:
        return

    with st.expander("🔍 Trazas del agente", expanded=False):
        for step in steps:
            t = step["type"]
            name = step.get("name", "")
            content = step.get("content", "")

            if t == "thinking":
                st.markdown(f"**💭 Razonando…**")
                # Limpiamos espacios extra y usamos white-space: normal para evitar doble interlineado
                clean_content = content.strip()
                st.markdown(
                    f'<style>div.thinking-box p:last-child {{ margin-bottom: 0 !important; }}</style>'
                    f'<div class="thinking-box" style="font-size:0.82rem;color:#8e8ea0;border-left:3px solid #3a3a3a;'
                    f'padding:10px;margin:4px 0 0 0;white-space: normal;overflow-wrap: break-word;'
                    f'background-color: rgba(255,255,255,0.03);border-radius:0 4px 4px 0;">{clean_content}</div>',
                    unsafe_allow_html=True,
                )
            elif t == "tool_call":
                # Determinar emoji según herramienta
                emoji = {"search_documents": "🗂️", "web_search": "🌐",
                         "get_current_date": "📅", "calculator": "🔢"}.get(name, "🔧")
                st.markdown(f"**{emoji} Acción → `{name}`**")
                if content:
                    st.code(content, language="json")
            elif t == "tool_return":
                emoji = {"search_documents": "📄", "web_search": "🔗",
                         "get_current_date": "🗓️", "calculator": "🧮"}.get(name, "📥")
                st.markdown(f"**{emoji} Resultado de `{name}`**")
                if content:
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#b4b4b4;background:#1a1a1a;'
                        f'border-radius:6px;padding:10px 14px;margin:4px 0;border:1px solid #333;'
                        f'max-height:250px;overflow-y:auto;">{content}</div>',
                        unsafe_allow_html=True
                    )


def _stream_agent_response(question: str, prior_messages: list):
    """Llama al agente y hace streaming chunked de la respuesta."""
    # Historial previo (solo user/assistant, sin pasos ni fuentes)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in prior_messages
        if m["role"] in ("user", "assistant")
    ]

    # Placeholder de escritura progresiva (simula stream)
    placeholder = st.empty()
    with st.spinner("Pensando y consultando fuentes. Un momento..."):
        try:
            answer, sources, steps = _run_agent(question, history)
            st.session_state["agent_ready"] = True
        except Exception as e:
            answer = f"❌ Error al ejecutar el agente: {e}"
            sources, steps = [], []

    # Mostrar respuesta
    if st.session_state.get("streaming", True):
        # Efecto de escritura progresiva
        displayed = ""
        words = answer.split(" ")
        for i, word in enumerate(words):
            displayed += word + (" " if i < len(words) - 1 else "")
            placeholder.markdown(displayed + "▌")
            time.sleep(0.012)
        placeholder.markdown(answer)
    else:
        # Mostrar de golpe
        placeholder.markdown(answer)

    # Fuentes
    if sources:
        with st.expander("📎 Fuentes consultadas", expanded=True):
            for src in sources:
                st.markdown(f"- `{src}`")

    # Trazas
    _render_agent_steps(steps)

    # Guardar en sesión
    add_message("assistant", answer, sources, steps)
