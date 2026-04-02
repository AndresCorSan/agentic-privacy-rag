import os
import re

# =========================
# CONFIGURACIÓN
# =========================

MIN_WORDS = 3000   # mínimo recomendado por documento
MAX_WORDS = 8000   # máximo recomendado por documento


# =========================
# FUNCIONES
# =========================

def count_words(text):
    # eliminar markdown básico
    text = re.sub(r'#', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # links
    text = re.sub(r'[`*_>-]', '', text)

    words = text.split()
    return len(words)


def evaluate_size(word_count):
    if word_count < MIN_WORDS:
        return "🔻 DEMASIADO PEQUEÑO"
    elif word_count > MAX_WORDS:
        return "🔺 DEMASIADO GRANDE"
    else:
        return "✅ ADECUADO"


def analyze_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    word_count = count_words(content)
    status = evaluate_size(word_count)

    print(f"\n📄 {filepath}")
    print(f"   → Palabras: {word_count}")
    print(f"   → Estado: {status}")


def analyze_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                analyze_file(filepath)


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    path = "corpus\\"  # ruta al archivo o carpeta a analizar

    if os.path.isfile(path):
        analyze_file(path)
    elif os.path.isdir(path):
        analyze_folder(path)
    else:
        print("❌ Ruta no válida")