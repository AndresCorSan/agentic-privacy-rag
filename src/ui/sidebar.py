import streamlit as st
from src.utils.session import new_conversation, switch_conversation
from datetime import datetime


def render_sidebar():
    with st.sidebar:

        st.title("🤖 RAG Expert Chat")

        # ── New conversation ───────────────────────────────────
        if st.button("✏️ Nueva conversación", use_container_width=True):
            new_conversation()
            st.rerun()

        st.divider()

        # ── Conversation history ───────────────────────────────
        convs = st.session_state.conversations
        current = st.session_state.current_conv_id
        today = datetime.now().date()

        groups = {"Hoy 📅": [], "Ayer": [], "Anteriores": []}
        for cid, data in reversed(list(convs.items())):
            delta = (today - data["created_at"].date()).days
            if delta == 0:
                groups["Hoy 📅"].append((cid, data))
            elif delta == 1:
                groups["Ayer"].append((cid, data))
            else:
                groups["Anteriores"].append((cid, data))

        for group_name, items in groups.items():
            if not items:
                continue
            st.caption(group_name.upper())
            for cid, data in items:
                is_active = cid == current
                label = f"**{data['title']}**" if is_active else data["title"]
                if st.button(
                    f"💬 {data['title']}",
                    key=f"conv_{cid}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    switch_conversation(cid)
                    st.rerun()

        st.divider()

        # ── Settings ───────────────────────────────────────────
        st.caption("⚙️ CONFIGURACIÓN")

        st.selectbox(
            "Modelo",
            options=[
                "RAG Expert (default)",
                "GPT-4o",
                "GPT-3.5 Turbo",
                "Llama 3 (local)",
            ],
            key="selected_model",
        )

        st.slider(
            "Temperatura",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="temperature",
        )

        st.toggle("Streaming", key="streaming")

        st.divider()

        # ── User footer ────────────────────────────────────────
        st.caption("👤 Usuario")
