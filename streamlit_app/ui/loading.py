"""
Pantalla de carga mostrada mientras el agente RAG se inicializa.
Se llama desde app.py cuando el agente aún no está en session_state.
Usa exclusivamente elementos nativos de Streamlit (sin HTML).
"""

import time
import streamlit as st


def render_loading_screen() -> None:
    """
    Muestra la pantalla de inicialización con barra de progreso y
    estado por etapa. Al terminar guarda el bundle en session_state
    y hace st.rerun() para mostrar la interfaz de chat.
    """
    from streamlit_app.agent import initialize_agent

    # ── Layout centrado ───────────────────────────────────────────────────────
    _, col, _ = st.columns([1, 2, 1])

    with col:
        st.markdown("## 🔒 RAG Privacy Expert")
        st.caption("Inicializando los modelos de IA… por favor espere.")
        st.write("")

        # Barra de progreso con texto de estado integrado
        progress_bar = st.progress(0.0, text="Preparando el entorno…")
        st.write("")

        # st.status: muestra un panel expandido con cada paso del proceso
        with st.status("Cargando componentes…", expanded=True) as status:
            step_placeholders = {}
            current_step_msg = None

            def on_step(message: str, progress: float) -> None:
                nonlocal current_step_msg
                
                # 1. Actualizar barra de progreso principal
                progress_bar.progress(min(progress, 1.0), text=message)

                # 2. Si había un paso anterior en curso, lo marcamos como completado
                if current_step_msg in step_placeholders:
                    step_placeholders[current_step_msg].write(f"✅ {current_step_msg}")

                # 3. Si no hemos llegado al final, creamos el placeholder para el paso actual
                if progress < 1.0:
                    step_placeholders[message] = st.empty()
                    step_placeholders[message].write(f"⏳ **{message}**")
                    current_step_msg = message

            try:
                bundle = initialize_agent(on_step=on_step)
                st.session_state["agent_bundle"] = bundle
                st.session_state["agent_ready"] = True
                
                # Marcar el último paso como completado manualmente
                if current_step_msg in step_placeholders:
                    step_placeholders[current_step_msg].write(f"✅ {current_step_msg}")

                status.update(
                    label="✅ ¡Agente listo!",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:
                status.update(label=f"Error al inicializar: {e}", state="error")
                st.error(f"❌ {e}")
                st.stop()

        # Breve pausa para que el usuario vea el estado final antes del rerun
        time.sleep(0.5)

    st.rerun()
