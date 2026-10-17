import streamlit as st
import time

from streamlit_app.utils.session import get_current_messages, add_message

# ── Suggestion prompts ────────────────────────────────────────────────────────
SUGGESTIONS = [
    {
        "icon": "🔍",
        "title": "Comparativa de políticas de privacidad: Google vs. Spotify",
        "subtitle": "Para responder, el agente debe buscar en diferentes documentos del corpus",
        "prompt": "¿Qué empresa es más privacy-friendly: Google o Spotify? Responde en formato tabla.",
    },
    {
        "icon": "🍪",
        "title": "Funcionamiento y tipos de cookies",
        "subtitle": "El agente devuelve imágenes en su respuesta",
        "prompt": "How do cookies work? What types are there?",
    },
    {
        "icon": "🌐",
        "title": "Datos recopilados por Netflix al usar sus servicios",
        "subtitle": "Ejemplo de uso de la tool web_search",
        "prompt": "¿Qué datos recopila Netflix cuando usamos sus servicios?",
    },
    {
        "icon": "🔢",
        "title": "Calcular multa por incumplimiento del GDPR",
        "subtitle": "Requiere búsqueda en documentos y uso de la tool calculator",
        "prompt": "Teniendo en cuenta que una empresa factura 2 millones de euros anuales, ¿cuánto tendría que pagar de multa si incumple el GDPR?",
    },
]


# ── Content Rendering (Multi-modal) ───────────────────────────────────────────

import re
import os

def render_image(img_filename: str):
    """Renderiza una imagen individual del corpus."""
    img_path = os.path.join("corpus", "images", img_filename.strip())
    if os.path.exists(img_path):
        st.image(
            img_path, 
            caption=f"{img_filename}", 
            width="stretch"
        )
    else:
        st.caption(f"🖼️ _(Imagen no encontrada: {img_filename})_")

def render_assistant_multimodal(content: str, steps: list = None):
    """
    Renderiza el contenido del asistente y, proactivamente, 
    una galería de imágenes encontradas en sus trazas.
    """
    # 1. Texto principal
    st.markdown(content)

    # 2. Galería de imágenes proactivas
    if steps:
        found_images = []
        for step in steps:
            if step["type"] == "tool_return":
                matches = re.findall(r"\[IMAGE:\s*(.*?)\]", str(step["content"]))
                for m in matches:
                    if m not in found_images and f"[IMAGE: {m}]" not in (content or ""):
                        found_images.append(m)
        
        # Renderizar en filas de hasta 3 imágenes (33% de ancho cada una)
        for i in range(0, len(found_images), 3):
            batch = found_images[i : i + 3]
            cols = st.columns(3) 
            for idx, img_name in enumerate(batch):
                with cols[idx]:
                    render_image(img_name)


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
    model_name = "GPT-OSS 120B · RAG + Qdrant"

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
            "Soy tu asistente experto en **políticas de privacidad, términos de "
            "servicio y protección de datos**, potenciado por un sistema RAG con "
            "búsqueda híbrida en documentos oficiales de Google, Spotify y guías "
            "sobre el GDPR."
        )

        st.write("")
        col1, col2 = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            col = col1 if i % 2 == 0 else col2
            with col:
                label = f"{s['icon']} **{s['title']}**\n\n_{s['subtitle']}_"
                if st.button(label, key=f"sug_{i}", use_container_width=True, wrap=True):
                    triggered_prompt = s["prompt"]

    else:
        # ── Render mensajes existentes ─────────────────────────────────
        for msg in messages:
            role = msg["role"]
            avatar = "streamlit_app/assets/agent_avatar.png" if role == "assistant" else "streamlit_app/assets/user_avatar.png"
            with st.chat_message(role, avatar=avatar):
                if role == "assistant":
                    render_assistant_multimodal(msg["content"], msg.get("steps"))
                else:
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
    user_input = st.chat_input("Pregunta lo que quieras...")
    final_input = triggered_prompt or user_input

    if final_input:
        add_message("user", final_input)
        with st.chat_message("user", avatar="streamlit_app/assets/user_avatar.png"):
            st.markdown(final_input)

        with st.chat_message("assistant", avatar="streamlit_app/assets/agent_avatar.png"):
            _stream_agent_response(final_input, messages)

        st.rerun()


import html

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
                    # Determinamos si escapamos el contenido o permitimos HTML puro
                    if name == "search_documents":
                        display_content = html.escape(content).replace("\n", "<br>")
                    else:
                        # Para web_search y otros, permitimos Markdown (negritas, enlaces)
                        # Reemplazamos saltos de línea por <br> manuales para asegurar consistencia dentro del div
                        display_content = content

                    st.markdown(
                        f'''
                        <div style="font-size:0.82rem;color:#d1d1d1;background:#16181e;
                        border-radius:6px;padding:12px;margin:8px 0;border:1px solid #3a3a3a;
                        max-height:350px;overflow-y:auto;white-space: normal; line-height: 1.5;">
                            {display_content}
                        </div>
                        ''',
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
        # Usamos la lógica multimodal para la respuesta final
        with st.container():
            render_assistant_multimodal(answer, steps)
    else:
        # Mostrar de golpe con multimodal
        render_assistant_multimodal(answer, steps)

    # Fuentes
    if sources:
        with st.expander("📎 Fuentes consultadas", expanded=True):
            for src in sources:
                st.markdown(f"- `{src}`")

    # Trazas
    _render_agent_steps(steps)

    # Guardar en sesión
    add_message("assistant", answer, sources, steps)
