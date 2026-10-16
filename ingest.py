"""Ingesta del corpus en Qdrant.

Logica reutilizada de notebooks/fase2.ipynb (celdas 25, 26, 29 y 30):
chunking recursivo + embeddings E5 + vectores sparse BM25 + upsert.
Ejecutar desde la raiz del repo con Qdrant levantado:

    docker compose up -d
    python ingest.py
"""
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunking_documento(doc_path):
    """
    Realiza chunking recursivo con solapamiento para un documento,
    devolviendo chunks junto con metadatos de archivo, categoria y empresa.
    """
    # Extraer nombre del archivo y categoria para metadatos
    ruta = Path(doc_path)
    archivo = ruta.name
    categoria = ruta.parent.name

    # Guardar empresa como metadato adicional
    nombre_archivo = archivo.lower()
    if "google" in nombre_archivo:
        empresa = "Google"
    elif "spotify" in nombre_archivo:
        empresa = "Spotify"
    else:
        empresa = None

    # Leer el contenido del documento
    with open(doc_path, "r", encoding="utf-8") as f:
        texto = f.read()

    # Chunking recursivo
    splitter_recursivo = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # maximo de caracteres por chunk
        chunk_overlap=150,    # 15% de solapamiento
        # Prueba separadores de mayor a menor granularidad
        separators=["\n# ", "\n## ", "\n### ", "\n---", "\n\n", "\n", ". ", " "],
        length_function=len,
    )
    chunks = splitter_recursivo.split_text(texto)

    # En algunos casos, los titulos quedan aislados como chunks muy cortos.
    # Se unen al siguiente chunk para mejorar contexto.
    chunks_unidos = []
    buffer = ""
    for chunk in chunks:
        # si es titulo (empieza por #) y chunk corto
        if chunk.startswith("#") and len(chunk) < 100:
            buffer = chunk
        else:
            if buffer:
                chunks_unidos.append(buffer + "\n" + chunk)
                buffer = ""
            else:
                chunks_unidos.append(chunk)
    if buffer:
        chunks_unidos.append(buffer)

    # Devolver chunks + metadatos
    return {
        "archivo": archivo,
        "categoria": categoria,
        "empresa": empresa,
        "chunks": chunks_unidos,
    }


def chunking_corpus(corpus_dir):
    """
    Realiza el proceso de chunking para todos los documentos .md en un directorio dado,
    devolviendo una lista de diccionarios con los chunks y metadatos asociados a cada documento.
    """
    corpus = []
    for doc_path in Path(corpus_dir).rglob("*.md"):
        doc_chunks = chunking_documento(doc_path)
        corpus.append(doc_chunks)
    return corpus


def generar_embeddings_corpus(chunks_corpus, modelo_emb):
    """
    Genera embeddings para todos los chunks del corpus y preserva metadata.
    """
    flat_chunks = []

    # Aplanar manteniendo metadata
    for doc in chunks_corpus:
        for chunk in doc["chunks"]:
            flat_chunks.append({
                "chunk": chunk,
                "archivo": doc["archivo"],
                "categoria": doc["categoria"],
                "empresa": doc["empresa"]
            })

    # Preparar textos para E5
    chunks_e5 = ["passage: " + c["chunk"] for c in flat_chunks]

    # Embeddings en batch
    embeddings = modelo_emb.encode(
        chunks_e5,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Unir embeddings + metadata
    embeddings_metadata = []
    for i, item in enumerate(flat_chunks):
        embeddings_metadata.append({
            "embedding": embeddings[i],
            "chunk": item["chunk"],
            "archivo": item["archivo"],
            "categoria": item["categoria"],
            "empresa": item["empresa"]
        })

    return embeddings_metadata


def main():
    import time
    from sentence_transformers import SentenceTransformer
    from fastembed import SparseTextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct, SparseVector

    t0 = time.time()
    print("=" * 60)
    print("  INGESTA DEL CORPUS EN QDRANT")
    print("=" * 60)

    print("\n[1/4] Cargando modelos (E5 + BM25)...")
    modelo_emb = SentenceTransformer("intfloat/multilingual-e5-base")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    corpus_dir = "corpus/"
    print(f"      Hecho en {time.time() - t0:.1f}s")

    print("\n[2/4] Chunking + embeddings del corpus...")
    t1 = time.time()
    # 1. Chunking + embeddings para todo el corpus
    chunks_corpus = chunking_corpus(corpus_dir)
    embeddings_corpus = generar_embeddings_corpus(chunks_corpus, modelo_emb)

    # Sparse vectors para busqueda BM25
    textos = [item["chunk"] for item in embeddings_corpus]
    sparse_vectors = list(sparse_model.embed(textos))

    print(f"      Documentos procesados:      {len(chunks_corpus)}")
    print(f"      Chunks con embedding:       {len(embeddings_corpus)}")
    print(f"      Vectores sparse generados:  {len(sparse_vectors)}")
    print(f"      Hecho en {time.time() - t1:.1f}s")

    print("\n[3/4] Conectando con Qdrant (localhost:6333)...")
    # 2. Conectar con Qdrant
    cliente = QdrantClient(host="localhost", port=6333)

    collection_name = "rag_privacy_corpus"

    # Recrear coleccion para evitar conflicto de esquema o ids previos
    if cliente.collection_exists(collection_name=collection_name):
        cliente.delete_collection(collection_name=collection_name)

    cliente.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=768, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )

    print(f"      Coleccion lista: {collection_name}")

    print("\n[4/4] Insertando fragmentos en Qdrant...")
    # 3.Ingest: insertar los embeddings con metadata en Qdrant
    puntos = []
    for i, item in enumerate(embeddings_corpus):
        puntos.append(
            PointStruct(
                id=i,
                vector={
                    "dense": item["embedding"].tolist(),
                    "sparse": SparseVector(
                        indices=sparse_vectors[i].indices.tolist(),
                        values=sparse_vectors[i].values.tolist(),
                    ),
                },
                payload={
                    "chunk": item["chunk"],
                    "archivo": item["archivo"],
                    "categoria": item["categoria"],
                    "empresa": item["empresa"],
                },
            )
        )

    # Upsert: inserta o actualiza si el id ya existe
    cliente.upsert(collection_name="rag_privacy_corpus", points=puntos)
    print(f"      Fragmentos insertados/actualizados: {len(puntos)}")
    print(f"\n  Ingesta completa en {time.time() - t0:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
