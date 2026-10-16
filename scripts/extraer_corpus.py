import os
import requests
from bs4 import BeautifulSoup

# =========================
# CONFIGURACIÓN
# =========================

SOURCES = {
    "privacy_policies": [
        ("google_privacy", "https://policies.google.com/privacy"),
        ("spotify_privacy", "https://www.spotify.com/legal/privacy-policy/"),
    ],
    "terms": [
        ("google_terms", "https://policies.google.com/terms"),
        ("spotify_terms", "https://www.spotify.com/legal/end-user-agreement/"),
    ],
    "guides": [
        ("gdpr_explained", "https://gdpr.eu/what-is-gdpr/"),
        ("cookies_guide", "https://www.kaspersky.com/resource-center/definitions/cookies"),
    ],
}

OUTPUT_DIR = "corpus"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# =========================
# FUNCIONES
# =========================

def fetch_html(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"❌ Error descargando {url}: {e}")
        return None


def clean_soup(soup):
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup


# =========================
# TABLAS
# =========================

def table_to_markdown(table):
    rows = table.find_all("tr")
    md_table = []

    for i, row in enumerate(rows):
        cols = row.find_all(["td", "th"])
        cols_text = [col.get_text(" ", strip=True) for col in cols]

        if not cols_text:
            continue

        md_table.append("| " + " | ".join(cols_text) + " |")

        if i == 0:
            md_table.append("| " + " | ".join(["---"] * len(cols_text)) + " |")

    return "\n".join(md_table)


def is_inside_table(element):
    return element.find_parent("table") is not None


# =========================
# LISTAS
# =========================

def is_nested_li(element):
    return element.find_parent("li") is not None


def process_list(li):
    lines = []

    # texto principal (sin hijos)
    main_text = li.find(string=True, recursive=False)
    if main_text:
        main_text = main_text.strip()
        if main_text:
            lines.append(f"- {main_text}")

    # sublista directa
    sublist = li.find("ul")
    if sublist:
        for sub_li in sublist.find_all("li", recursive=False):
            sub_text = sub_li.get_text(" ", strip=True)
            if sub_text:
                lines.append(f"  - {sub_text}")

    return "\n".join(lines)


# =========================
# HTML → MARKDOWN
# =========================

def html_to_markdown(soup):
    markdown_lines = []

    for element in soup.find_all(["h1", "h2", "h3", "p", "li", "table"]):

        # evitar contenido dentro de tablas
        if is_inside_table(element):
            continue

        # procesar tablas
        if element.name == "table":
            table_md = table_to_markdown(element)
            markdown_lines.append("\n" + table_md + "\n")
            continue

        # evitar duplicados de sublistas
        if element.name == "li" and is_nested_li(element):
            continue

        text = element.get_text(" ", strip=True)

        if not text:
            continue

        if element.name == "h1":
            markdown_lines.append(f"# {text}\n")
        elif element.name == "h2":
            markdown_lines.append(f"## {text}\n")
        elif element.name == "h3":
            markdown_lines.append(f"### {text}\n")
        elif element.name == "li":
            markdown_lines.append(process_list(element))
        else:
            markdown_lines.append(text + "\n")

    return "\n".join(markdown_lines)


# =========================
# GUARDADO
# =========================

def save_markdown(content, category, name):
    folder = os.path.join(OUTPUT_DIR, category)
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, f"{name}.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Guardado: {filepath}")


# =========================
# MAIN
# =========================

def main():
    for category, sources in SOURCES.items():
        for name, url in sources:
            print(f"⬇️ Descargando: {url}")

            html = fetch_html(url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            soup = clean_soup(soup)

            markdown = html_to_markdown(soup)

            header = f"# {name}\n\nFuente: {url}\n\n---\n\n"
            markdown = header + markdown

            save_markdown(markdown, category, name)


if __name__ == "__main__":
    main()