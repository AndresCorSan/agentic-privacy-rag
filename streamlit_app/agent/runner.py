"""
Inicialización y ejecución del agente pydantic-ai RAG.
Orquesta la carga de modelos (retrieval.py) y el registro de tools (tools.py).
"""

import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT_AGENT = (
    "Eres un asistente especializado en privacidad, protección de datos y términos de servicio.\n\n"

    "Dispones de una herramienta llamada 'search_documents' que permite buscar información en documentos oficiales.\n"
    "Dispones de una herramienta llamada 'web_search' que permite buscar información en internet.\n"
    "Tambien, tienes acceso a otras herramientas auxiliares.\n\n"

    "INSTRUCCIONES DE USO DE HERRAMIENTAS:\n"
    "- Si la pregunta requiere información específica, normativa o detallada del dominio, usa la herramienta 'search_documents'.\n"
    "- Si no estas seguro de tener información oficial y verificada, usa la herramienta 'search_documents'.\n"
    "- Si puedes responder con conocimiento general fiable, responde directamente sin usar la herramienta.\n"
    "- Usa la herramienta 'web_search' SOLO cuando la herramienta 'search_documents' no devuelva resultados suficientes para responder a la pregunta del usuario.\n\n"

    "USO DE LA INFORMACIÓN:\n"
    "- Cuando uses información de la herramienta 'search_documents', intégrala de forma natural en la respuesta.\n"
    "- No menciones herramientas, contexto, fragmentos ni estructuras internas del corpus de documentos.\n"
    "- Puedes referirte a 'documentos oficiales' de forma general si es necesario cuando uses 'search_documents'.\n\n"

    "RESTRICCIONES:\n"
    "- No respondas preguntas fuera del dominio de privacidad y protección de datos.\n"
    "- No inventes información bajo ninguna circunstancia.\n\n"

    "FORMATO DE RESPUESTA:\n"
    "- Usa un tono formal.\n"
    "- Sé claro, breve y estructurado.\n"
    "- Responde en el mismo idioma que la pregunta del usuario."
)

def initialize_agent(on_step=None) -> dict:
    """
    Carga todos los modelos e inicializa el agente pydantic-ai con sus tools.

    Args:
        on_step: callable opcional ``(message: str, progress: float) -> None``
                 invocado antes de cada etapa pesada para reportar el progreso
                 (progress va de 0.0 a 1.0).

    Returns:
        dict con claves: agent, cliente_qdrant, modelo_emb, sparse_model, reranker
    """
    def _step(msg: str, pct: float) -> None:
        if on_step:
            on_step(msg, pct)

    _step("Cargando dependencias…", 0.05)
    from pydantic_ai import Agent
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider
    from tavily import TavilyClient

    from streamlit_app.agent.retrieval import (
        load_qdrant_client,
        load_embedding_model,
        load_sparse_model,
        load_reranker,
        make_detectar_categorias,
    )
    from streamlit_app.agent.tools import register_tools

    _step("Conectando con Qdrant…", 0.15)
    cliente_qdrant = load_qdrant_client()

    _step("Cargando modelo de embeddings (multilingual-e5-base)…", 0.25)
    modelo_emb = load_embedding_model()

    _step("Calculando embeddings de categorías…", 0.50)
    detectar_categorias = make_detectar_categorias(modelo_emb)

    _step("Cargando modelo sparse BM25…", 0.60)
    sparse_model = load_sparse_model()

    _step("Cargando reranker (bge-reranker-v2-m3)…", 0.72)
    reranker = load_reranker()

    _step("Inicializando LLM Qwen3-32B vía Groq…", 0.85)
    groq_model = GroqModel(
        "qwen/qwen3-32b",
        provider=GroqProvider(api_key=os.getenv("GROQ_KEY")),
        settings={"temperature": 0.2},
    )

    _step("Conectando con Tavily (búsqueda web)…", 0.90)
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    _step("Registrando herramientas del agente…", 0.95)
    agent = Agent(groq_model, system_prompt=SYSTEM_PROMPT_AGENT)
    register_tools(
        agent=agent,
        cliente_qdrant=cliente_qdrant,
        modelo_emb=modelo_emb,
        sparse_model=sparse_model,
        reranker=reranker,
        detectar_categorias=detectar_categorias,
        tavily_client=tavily_client,
    )

    _step("¡Agente listo!", 1.0)

    return {
        "agent": agent,
        "cliente_qdrant": cliente_qdrant,
        "modelo_emb": modelo_emb,
        "sparse_model": sparse_model,
        "reranker": reranker,
    }


def run_agent(
    agent_bundle: dict, 
    question: str, 
    history: list[dict] | None = None,
    model_settings: dict | None = None
) -> tuple:
    """
    Ejecuta el agente con la pregunta del usuario.

    Args:
        agent_bundle:   dict devuelto por initialize_agent()
        question:       pregunta del usuario
        history:        lista de mensajes previos
        model_settings: dict opcional con parámetros del modelo (ej: {"temperature": 0.4})

    Returns:
        (answer: str, sources: list[str], steps: list[dict])
    """
    from pydantic_ai.messages import (
        ModelMessage, 
        ModelRequest, 
        ModelResponse, 
        UserPromptPart, 
        TextPart,
        ToolCallPart, 
        ToolReturnPart, 
        ThinkingPart
    )

    agent = agent_bundle["agent"]

    # ── Gestión de Memoria (Protección de Tokens 6K TPM) ────────────────────
    # Limitamos a los últimos 6 mensajes para ahorrar tokens
    MAX_HISTORY_MESSAGES = 6
    history_window = history[-MAX_HISTORY_MESSAGES:] if history else []

    message_history: list[ModelMessage] = []
    for m in history_window:
        content = m["content"]
        # Seguridad: si un mensaje histórico es muy largo, lo recortamos
        if len(content) > 1500:
            content = content[:1500] + "... [recortado]"

        if m["role"] == "user":
            message_history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif m["role"] == "assistant":
            message_history.append(ModelResponse(parts=[TextPart(content=content)]))

    # Ejecutar con el historial optimizado para no superar TPM
    result = agent.run_sync(
        question, 
        message_history=message_history, 
        model_settings=model_settings
    )

    steps: list[dict] = []
    sources: list[str] = []

    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                steps.append({"type": "tool_call", "name": part.tool_name, "content": str(part.args)})

            elif isinstance(part, ToolReturnPart):
                content_str = str(part.content)
                steps.append({"type": "tool_return", "name": part.tool_name, "content": content_str})

                # Extraer fuentes de search_documents
                if part.tool_name == "search_documents" and content_str not in ("NO_RESULTS", "ERROR"):
                    for line in content_str.split("\n"):
                        if line.startswith("[") and "]" in line:
                            archivo = line[1:line.index("]")]
                            if archivo not in sources:
                                sources.append(archivo)
                
                # Extraer fuentes de web_search
                elif part.tool_name == "web_search" and content_str not in ("NO_RESULTS", "ERROR"):
                    import re
                    # Solo extraer URLs que sigan el patrón "Fuente: https://..."
                    urls = re.findall(r"Fuente:\s*(https?://[^\s\n]+)", content_str)
                    for u in urls:
                        if u not in sources:
                            sources.append(u)

            elif isinstance(part, ThinkingPart):
                steps.append({"type": "thinking", "name": "Razonamiento", "content": part.content})

    return result.output, sources, steps
