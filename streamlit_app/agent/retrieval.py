"""
RAG retrieval pipeline: hybrid search (dense + sparse) + reranking + filter detection.
Extrae toda la lógica de Qdrant del agente para mantenerla separada.
"""

import os
import logging
import warnings
import numpy as np

# ── Silence redundant logs ────────────────────────────────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("fastembed").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ── Model loaders (lazy, para no importar en el módulo global) ────────────────

def load_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(host="localhost", port=6333)


def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("intfloat/multilingual-e5-base")


def load_sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding("Qdrant/bm25")


def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-v2-m3")


# ── Búsqueda híbrida en Qdrant ────────────────────────────────────────────────

def buscar_qdrant(
    query: str,
    cliente_qdrant,
    modelo_emb,
    sparse_model,
    reranker,
    top_k_inicial: int = 10,
    top_k_final: int = 3,
    filtro_categoria=None,
    filtro_empresa=None,
) -> list[dict]:
    """
    Realiza una búsqueda en Qdrant utilizando un pipeline completo de RAG:

    1. Búsqueda híbrida (dense + sparse/BM25):
       - Combina embeddings densos (E5) con representaciones sparse (BM25)
       - Los resultados de ambas búsquedas se fusionan mediante Reciprocal Rank Fusion (RRF)
       - Mejora la recuperación al capturar similitud semántica y coincidencias léxicas

    2. Filtrado por metadatos (opcional):
       - Permite restringir resultados por categoría o empresa

    3. Reranking con cross-encoder:
       - Reordena los resultados iniciales evaluando directamente la relación (query, chunk)
       - Mejora la precisión final del contexto

    Parameters:
        query (str): consulta del usuario
        top_k_inicial (int): número de candidatos recuperados inicialmente
        top_k_final (int): número de resultados tras reranking
        filtro_categoria (str, opcional): filtrar por categoría
        filtro_empresa (str, opcional): filtrar por empresa

    Returns:
        list[dict]: lista de chunks relevantes con metadata y scores
    """
    from qdrant_client import models
    from qdrant_client.models import Filter, FieldCondition, MatchValue, SparseVector

    # --- 1. RETRIEVAL HÍBRIDO (dense + sparse)
    # Embedding denso (semántico)
    query_emb = modelo_emb.encode(["query: " + query], normalize_embeddings=True)[0]

    # Embedding disperso (coincidencia léxica)
    query_sparse_raw = list(sparse_model.embed([query]))[0]
    query_sparse = SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist(),
    )

    # --- 2. FILTRADO POR METADATA (opcional)
    filtro = None

    if filtro_categoria or filtro_empresa:
        must_conditions = []
        
        # Categorías (OR entre varias categorías)
        if filtro_categoria:
            if isinstance(filtro_categoria, list):
                # Si es una lista de categorías, se crea un bloque OR (should) para ellas
                categorias_cond = Filter(
                    should=[
                        FieldCondition(
                            key="categoria",
                            match=MatchValue(value=cat)
                        )
                        for cat in filtro_categoria
                    ]
                )
                must_conditions.append(categorias_cond)
            else:
                # Si es una única categoría, se añade directamente como condición
                must_conditions.append(
                    FieldCondition(
                        key="categoria",
                        match=MatchValue(value=filtro_categoria)
                    )
                )

        # Empresas (OR entre varias empresas)
        # Solo se aplica este filtro si la categoría es "terms" o "privacy_policies"
        if filtro_empresa:
            aplicar_filtro_empresa = False

            if isinstance(filtro_categoria, list):
                if "guides" in filtro_categoria:
                    aplicar_filtro_empresa = False
                else:
                    aplicar_filtro_empresa = any(
                        cat in ["terms", "privacy_policies"] for cat in filtro_categoria
                    )
            else:
                aplicar_filtro_empresa = filtro_categoria in ["terms", "privacy_policies"]

            if aplicar_filtro_empresa:
                if isinstance(filtro_empresa, list):
                    empresas_cond = Filter(
                        should=[
                            FieldCondition(
                                key="empresa",
                                match=MatchValue(value=emp)
                            )
                            for emp in filtro_empresa
                        ]
                    )
                    must_conditions.append(empresas_cond)
                else:
                    must_conditions.append(
                        FieldCondition(
                            key="empresa",
                            match=MatchValue(value=filtro_empresa)
                        )
                    )

        # Construcción final (AND entre bloques)
        filtro = Filter(must=must_conditions)

    # --- 3. BÚSQUEDA EN QDRANT (HÍBRIDA)
    # Qdrant combina automáticamente dense + sparse en un único score híbrido
    resultados = cliente_qdrant.query_points(
        collection_name="rag_privacy_corpus",
        prefetch=[
            # Recupera candidatos de forma independiente para cada tipo de representación
            models.Prefetch(
                query=query_emb.tolist(),
                using="dense",
                limit=top_k_inicial * 2, # Doble de candidatos para mejorar recall antes del reranking
            ),
            models.Prefetch(
                query=query_sparse,
                using="sparse",
                limit=top_k_inicial * 2,
            ),
        ],
        # Combina los rankings de dense y sparse usando RRF
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        # Se reduce el número de candidatos antes del reranking
        limit=top_k_inicial, 
        query_filter=filtro,
        with_payload=True,
    )

    # --- 4. RERANKING CON CROSS-ENCODER: evalúa cada par (query, chunk) conjuntamente
    # Crear pares (query, chunk)
    pairs = [
        (query, punto.payload["chunk"])
        for punto in resultados.points
    ]

    # Obtener scores del reranker
    scores = reranker.predict(pairs)

    # Combinar resultados
    reranked = []
    for i, punto in enumerate(resultados.points):
        reranked.append({
            "chunk": punto.payload["chunk"],
            "categoria": punto.payload["categoria"],
            "archivo": punto.payload["archivo"],
            "empresa": punto.payload.get("empresa", None),
            # Score híbrido (dense + sparse combinado por Qdrant)
            "score_hybrid": punto.score,
            # Score del reranker (decisión final)
            "score_rerank": float(scores[i])
        })

    # Ordenar por score del reranker
    reranked = sorted(reranked, key=lambda x: x["score_rerank"], reverse=True)

    # Seleccionar top final
    return reranked[:top_k_final]


# ── Detección automática de filtros ──────────────────────────────────────────

def detectar_empresas(query):
    empresas = []
    
    query_lower = query.lower()
    if "google" in query_lower:
        empresas.append("Google")
    if "spotify" in query_lower:
        empresas.append("Spotify")

    return empresas if empresas else None


def make_detectar_categorias(modelo_emb):
    """
    Factoría que devuelve una función detectar_categorias(query) usando similitud
    semántica contra descripciones de cada categoría del corpus.
    """
    categoria_descripciones = {
        "guides": "explanations of privacy concepts like GDPR and cookies",
        "privacy_policies": "how companies collect and use personal data",
        "terms": "terms of service and usage conditions",
    }
    categoria_embeddings = {
        cat: modelo_emb.encode(["passage: " + desc], normalize_embeddings=True)[0]
        for cat, desc in categoria_descripciones.items()
    }

    def detectar_categorias(query, margin=0.02, min_score=0.20):
        query_emb = modelo_emb.encode(["query: " + query], normalize_embeddings=True)[0]

        # Calcular similitud coseno entre la query y cada descripción de categoría
        scores = {}
        for cat, desc_emb in categoria_embeddings.items():
            scores[cat] = np.dot(query_emb, desc_emb)

        # Umbral mínimo real: si no hay suficiente similitud, no aplicar filtro de categoría
        max_score = max(scores.values())
        if max_score < min_score:
            return None

        # Identificar categorías con score cercano al máximo (dentro de un margen)
        categorias = [
            cat for cat, score in scores.items()
            if (max_score - score) <= margin
        ]

        return categorias if categorias else None

    return detectar_categorias
