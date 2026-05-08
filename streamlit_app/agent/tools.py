"""
Tool definitions for the pydantic-ai RAG agent.
Cada tool encapsula una capacidad concreta: búsqueda en corpus, web, fecha y calculadora.
"""

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from datetime import datetime
from typing import Literal, List, Optional


# ── Input/Output schemas ───────────────────────────────────────────────────────

class SearchDocumentsInput(BaseModel):
    query: str = Field(
        description=(
            "Consulta breve optimizada para búsqueda semántica. "
            "Usa palabras clave (no preguntas completas) e incluye conceptos relevantes "
            "y, si aplica, la empresa (Google, Spotify). "
            "Ejemplo: 'Spotify privacy policy personal data collection'."
        )
    )

    categoria: Optional[List[Literal['guides', 'privacy_policies', 'terms']]] = Field(
        default=None,
        description="Filtros de categoría de documento. Solo selecciona de la lista permitida."
    )

    empresa: Optional[List[Literal['Google', 'Spotify']]] = Field(
        default=None,
        description="Filtros de empresas. El corpus SOLO contiene información de Google y Spotify. No uses otros nombres aquí."
    )


class WebSearchInput(BaseModel):
    query: str = Field(
        description="Consulta breve para buscar información en la web"
    )


class GetCurrentDateOutput(BaseModel):
    current_date: str = Field(
        description="Fecha actual en formato YYYY-MM-DD"
    )


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="Expresión matemática simple (ej: '1000 * 0.04')"
    )


# ── Registro de tools en el agente ───────────────────────────────────────────

def register_tools(
    agent,
    cliente_qdrant,
    modelo_emb,
    sparse_model,
    reranker,
    detectar_categorias,
    tavily_client,
) -> None:
    """
    Registra las 4 tools (search_documents, web_search, get_current_date, calculator)
    en el agente pydantic-ai proporcionado.
    Los objetos de modelo/cliente se capturan por closure.
    """
    from pydantic_ai import RunContext
    from datetime import datetime
    
    from streamlit_app.agent.retrieval import buscar_qdrant, detectar_empresas

    # ── Tool 1: search_documents ──────────────────────────────────────────────
    @agent.tool
    def search_documents(ctx: RunContext, input: SearchDocumentsInput) -> str:
        """
        Busca información en un corpus de documentos oficiales y verificados sobre privacidad y protección de datos.

        El corpus incluye tres tipos de contenido:
        (1) guías explicativas sobre conceptos como GDPR, cookies y derechos de los usuarios.
        (2) políticas de privacidad que describen cómo empresas como Google y Spotify recopilan, usan y comparten datos personales.
        (3) términos de servicio con condiciones de uso de plataformas como Google y Spotify.

        Usa esta herramienta cuando la pregunta requiera información específica, detallada o normativa de estos documentos.
        Formula la query como una búsqueda semántica breve incluyendo conceptos clave.
        Puedes especificar filtros opcionales para mejorar la precisión de la búsqueda según la pregunta del usuario:
            - categoria: filtros disponibles (guides, privacy_policies, terms), puedes seleccionar varias categorías si es necesario.
            - empresa: filtros disponibles (Google, Spotify), solo aplicable para categorías "terms" y "privacy_policies"
        """
        try:
            resultados = buscar_qdrant(
                input.query,
                cliente_qdrant,
                modelo_emb,
                sparse_model,
                reranker,
                # Si el agente no especifica filtros, se intenta detectar automáticamente 
                # a partir de la query para mejorar la búsqueda
                filtro_empresa=input.empresa or detectar_empresas(input.query),
                filtro_categoria=input.categoria or detectar_categorias(input.query),
            )
            
            if not resultados:
                return "NO_RESULTS"

            # Añadir nombre de documento al principio de cada chunk para mayor trazabilidad
            return "\n\n".join(
                f"[{r['archivo']}] {r['chunk']}"
                for r in resultados
            )

        except Exception:
            return "ERROR"

    # ── Tool 2: web_search ────────────────────────────────────────────────────
    @agent.tool
    def web_search(ctx: RunContext, input: WebSearchInput) -> str:
        """
        Busca información en internet.

        Usa esta herramienta SOLO cuando la herramienta search_documents no devuelva resultados suficientes.
        """
        try:
            response = tavily_client.search(
                query=input.query,
                max_results=5
            )

            results = response.get("results", [])

            if not results:
                return "NO_RESULTS"

            # Construir contexto bien estructurado con HTML para asegurar el renderizado dentro del div
            formatted_results = []
            for r in results:
                url = r['url']
                title = r.get('title', 'Sin título')
                content = r.get('content', '')
                
                block = (
                    f"<b>Fuente:</b> <a href='{url}' target='_blank' style='color: #4facfe;'>{url}</a><br>"
                    f"<b>Título:</b> {title}<br>"
                    f"<p>{content}</p>"
                )
                formatted_results.append(block)

            return "".join(formatted_results)

        except Exception:
            return "ERROR"

    # ── Tool 3: get_current_date ──────────────────────────────────────────────
    @agent.tool
    def get_current_date(ctx: RunContext) -> GetCurrentDateOutput:
        """
        Devuelve la fecha actual.

        Usa esta herramienta cuando la pregunta requiera información temporal o dependa de la fecha actual.
        """
        try:
            return GetCurrentDateOutput(
                current_date=datetime.now().strftime("%Y-%m-%d")
            )
        except Exception:
            return GetCurrentDateOutput(current_date="ERROR")

    # ── Tool 4: calculator ────────────────────────────────────────────────────
    @agent.tool
    def calculator(ctx: RunContext, input: CalculatorInput) -> str:
        """
        Realiza cálculos matemáticos.

        Usa esta herramienta para operaciones numéricas precisas,
        como porcentajes o cálculos relacionados con multas o límites.
        """
        try:
            # Evaluación controlada (simple)
            expression = input.expression.replace(" ", "")
            
            # Solo permitir caracteres válidos
            allowed_chars = "0123456789+-*/().%"
            if any(c not in allowed_chars for c in expression):
                return "ERROR"

            result = eval(expression)
            return str(result)

        except Exception:
            return "ERROR"
