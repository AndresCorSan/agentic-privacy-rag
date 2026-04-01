import streamlit as st
import time
import random
from src.utils.session import get_current_messages, add_message

# ── Suggestion prompts ────────────────────────────────────────────────────────
SUGGESTIONS = [
    {
        "icon": "📄",
        "title": "Analiza un documento",
        "subtitle": "Sube un PDF y hazme preguntas sobre él",
        "prompt": "¿Cómo puedo subir y analizar un documento PDF?",
    },
    {
        "icon": "🔍",
        "title": "Búsqueda semántica",
        "subtitle": "Encuentra información en tu base de conocimiento",
        "prompt": "¿Cómo funciona la búsqueda semántica en RAG?",
    },
    {
        "icon": "🧠",
        "title": "¿Qué es RAG?",
        "subtitle": "Aprende sobre Retrieval-Augmented Generation",
        "prompt": "Explícame qué es RAG y cómo funciona paso a paso.",
    },
    {
        "icon": "⚙️",
        "title": "Configurar fuentes",
        "subtitle": "Conecta tu base de datos vectorial",
        "prompt": "¿Qué bases de datos vectoriales puedo usar con este chatbot?",
    },
]

# ── Mock responses ────────────────────────────────────────────────────────────
MOCK_RESPONSES = [
    """¡Buena pregunta! Permíteme darte una respuesta detallada basada en la base de conocimiento disponible.

RAG (Retrieval-Augmented Generation) combina dos componentes principales:

1. **Recuperación (Retrieval)**: Un sistema que busca documentos o fragmentos relevantes de una base de datos vectorial, usando similitud semántica.

2. **Generación (Generation)**: Un modelo de lenguaje grande (LLM) que usa esos fragmentos recuperados como contexto adicional para generar respuestas más precisas y fundamentadas.

Esto permite que el modelo responda con información actualizada y específica de tu dominio, sin necesidad de reentrenarlo.""",

    """Basándome en los documentos disponibles, aquí tienes la información encontrada:

La arquitectura RAG consta de los siguientes pasos:
- **Indexación**: Los documentos se dividen en *chunks* y se codifican como vectores usando un modelo de embeddings.
- **Recuperación**: Cuando el usuario hace una pregunta, se genera un vector de búsqueda y se compara con los vectores almacenados (similaridad coseno).
- **Augmentación**: Los fragmentos más relevantes se añaden al prompt del LLM como contexto.
- **Generación**: El LLM produce la respuesta final.

¿Te gustaría profundizar en alguno de estos pasos?""",

    """Las bases de datos vectoriales más populares para RAG son:

| Base de datos | Tipo | Características |
|---|---|---|
| **Chroma** | Open Source | Ligera, ideal para desarrollo local |
| **Pinecone** | Cloud | Escalable, gestión automática |
| **Weaviate** | Open/Cloud | Potente filtrado híbrido |
| **FAISS** | Open Source | Ultra-rápida, de Meta AI |
| **Qdrant** | Open/Cloud | Rust nativo, muy eficiente |

Para empezar te recomendaría **Chroma** por su sencillez de integración. ¿Quieres ver un ejemplo de código?""",
]


def _mock_stream(text: str):
    """Word-by-word streaming generator."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(random.uniform(0.02, 0.05))


def _get_bot_response(user_input: str):
    """
    Placeholder — replace with your RAG pipeline.
    Returns (response_text, [source_names])
    """
    response = random.choice(MOCK_RESPONSES)
    sources = random.sample(
        ["documento_tecnico.pdf", "manual_usuario.docx", "FAQ.pdf",
         "informe_2024.pdf", "base_conocimiento.txt"],
        k=random.randint(1, 3),
    )
    return response, sources


def render_chat_interface():
    # ── Header ────────────────────────────────────────────────
    conv = st.session_state.conversations[st.session_state.current_conv_id]
    model = st.session_state.get("selected_model", "RAG Expert (default)")

    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.subheader(conv["title"])
    with col_badge:
        st.caption(f"⚡ {model}")

    st.divider()

    # ── Get messages ───────────────────────────────────────────
    messages = get_current_messages()
    triggered_prompt = None

    # ── Welcome / empty state ──────────────────────────────────
    if not messages:
        st.markdown("## 👋 ¿En qué puedo ayudarte hoy?")
        st.write(
            "Soy tu asistente experto potenciado por RAG. "
            "Puedo analizar documentos, responder preguntas sobre tu base de "
            "conocimiento y mucho más."
        )
        st.write("")

        # Suggestion cards as buttons in a 2×2 grid
        col1, col2 = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            col = col1 if i % 2 == 0 else col2
            with col:
                label = f"{s['icon']} **{s['title']}**\n\n_{s['subtitle']}_"
                if st.button(label, key=f"sug_{i}", use_container_width=True):
                    triggered_prompt = s["prompt"]

    else:
        # ── Render existing messages ───────────────────────────
        for msg in messages:
            with st.chat_message(msg["role"],
                                 avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.caption("📎 Fuentes consultadas: " +
                               " · ".join(f"`{s}`" for s in msg["sources"]))

    # ── Chat input ─────────────────────────────────────────────
    user_input = st.chat_input("Escribe tu mensaje…")
    final_input = triggered_prompt or user_input

    if final_input:
        # Save & show user message
        add_message("user", final_input)
        with st.chat_message("user", avatar="👤"):
            st.markdown(final_input)

        # Generate & show assistant response
        with st.chat_message("assistant", avatar="🤖"):
            response_text, sources = _get_bot_response(final_input)

            if st.session_state.get("streaming", True):
                displayed = st.write_stream(_mock_stream(response_text))
            else:
                st.markdown(response_text)
                displayed = response_text

            if sources:
                st.caption("📎 Fuentes consultadas: " +
                           " · ".join(f"`{s}`" for s in sources))

        add_message("assistant", displayed, sources)
        st.rerun()

    # ── Disclaimer ─────────────────────────────────────────────
    st.divider()
    st.caption(
        "⚠️ RAG Expert puede cometer errores. "
        "Verifica la información importante con las fuentes originales."
    )
