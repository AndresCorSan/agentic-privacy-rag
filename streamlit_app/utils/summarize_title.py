import os
from groq import Groq

def generate_chat_title(text: str) -> str:
    """Genera un título corto y descriptivo para el chat usando GPT-OSS 20B."""
    try:
        client = Groq(api_key=os.getenv("GROQ_KEY"))
        # Nota: gpt-oss es un modelo de reasoning; el razonamiento comparte
        # presupuesto con la respuesta, así que hay que dejar margen de tokens
        # (con max_tokens=25 devolvía contenido vacío) y esfuerzo bajo.
        # Además Groq recomienda evitar system prompts en estos modelos.
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Eres un experto en micro-resúmenes. Resume la pregunta del usuario en un título de MÁXIMO 5 palabras. "
                        "Prohibido usar más de una línea. Prohibido usar saltos de línea. Responde únicamente con el texto del título.\n\n"
                        f"Pregunta: {text}"
                    ),
                },
            ],
            max_completion_tokens=128,
            temperature=0.5,
            reasoning_effort="low",
            include_reasoning=False,
        )
        content = completion.choices[0].message.content or ""
        title = content.strip()
        # Limpieza de saltos de línea y formateo
        title = title.replace("\n", " ").replace("\r", " ")
        title = title.replace('"', '').replace("'", "").replace(".", "")
        title = " ".join(title.split()) # Elimina espacios múltiples
        if not title:
            # Respuesta vacía (p. ej. truncada por reasoning): fallback
            return text[:42] + ("..." if len(text) > 42 else "")
        return title
    except Exception:
        # Fallback seguro: recorte tradicional
        return text[:42] + ("..." if len(text) > 42 else "")
