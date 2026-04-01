#######################################
# Describir imágenes con VLM (API)
#######################################

import base64
import mimetypes
import os

from openai import OpenAI

HF_TOKEN = os.getenv("HF_TOKEN", "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")
HF_MODEL  = os.getenv("HF_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
PROMPTO   = os.getenv(
    "CAPTION_PROMPT",
    (
        "Describe this image following this exact format:\n"
        "- One opening sentence summarizing what the image shows.\n"
        "- A short bullet list (3-5 items) with the key elements, steps or concepts visible.\n"
        "- One closing sentence explaining what the image illustrates or communicates.\n"
        "Be concise. Do not add extra sections or explanations beyond this format."
    ),
)

IMAGES_FOLDER = "corpus/images"
OUTPUT_FOLDER = "corpus/image_descriptions"

CLIENT = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)


# =========================
# GENERAR DESCRIPCIÓN
# =========================

def generar_descripcion(imagen_path):
    mime, _ = mimetypes.guess_type(imagen_path)
    if not mime:
        mime = "image/jpeg"

    with open(imagen_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    data_url = f"data:{mime};base64,{b64}"

    try:
        resp = CLIENT.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPTO},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"❌ Error en la llamada a la API: {exc}")
        return None


# =========================
# GUARDAR DESCRIPCIÓN
# =========================

def guardar_descripcion(nombre_imagen, descripcion):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    nombre_base = os.path.splitext(nombre_imagen)[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{nombre_base}.txt")

    contenido = f"""[IMAGE: {nombre_imagen}]

Descripción:
{descripcion}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(contenido)

    print(f"💾 Guardado: {output_path}")


# =========================
# PROCESAR TODAS LAS IMÁGENES
# =========================

def procesar_imagenes():
    if not os.path.isdir(IMAGES_FOLDER):
        print(f"❌ La carpeta de imágenes no existe: {IMAGES_FOLDER}")
        return

    for filename in os.listdir(IMAGES_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):

            path = os.path.join(IMAGES_FOLDER, filename)

            print(f"⬇️ Procesando {filename}...")

            descripcion = generar_descripcion(path)

            if descripcion:
                guardar_descripcion(filename, descripcion)

            print("-" * 50)


# =========================
# MAIN
# =========================

# python describir_imagen.py 
if __name__ == "__main__":
    procesar_imagenes()