import os
from groq import Groq

def generate_chat_title(text: str) -> str:
    """Genera un título corto y descriptivo para el chat usando GPT-OSS 20B."""
    try:
        client = Groq(api_key=os.getenv("GROQ_KEY"))
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "Eres un experto en micro-resúmenes. Resume la pregunta del usuario en un título de MÁXIMO 5 palabras. Prohibido usar más de una línea. Prohibido usar saltos de línea. Responde únicamente con el texto del título."},
                {"role": "user", "content": text}
            ],
            max_tokens=25,
            temperature=0.3
        )
        title = completion.choices[0].message.content.strip()
        # Limpieza de saltos de línea y formateo
        title = title.replace("\n", " ").replace("\r", " ")
        title = title.replace('"', '').replace("'", "").replace(".", "")
        return " ".join(title.split()) # Elimina espacios múltiples
    except Exception:
        # Fallback seguro: recorte tradicional
        return text[:42] + ("..." if len(text) > 42 else "")
