"""
Paquete agent - expone la API pública del agente RAG.
"""
from streamlit_app.agent.runner import initialize_agent, run_agent

__all__ = ["initialize_agent", "run_agent"]
