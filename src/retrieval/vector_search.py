from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest
from llama_index.embeddings.ollama import OllamaEmbedding
from dataclasses import dataclass


COLLECTION_NAME = "documents"


@dataclass
class SearchResult:
    content: str
    metadata: dict
    score: float
    source: str   # "vector" ou "bm25" — útil para debug


def vector_search(
    query: str,
    client: QdrantClient,
    top_k: int = 10,
) -> list[SearchResult]:
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    query_vector = embed_model.get_text_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        SearchResult(
            content=r.payload["content"],
            metadata={k: v for k, v in r.payload.items() if k != "content"},
            score=r.score,
            source="vector",
        )
        for r in results.points
    ]


def vector_search_multi(
    queries: list[str],
    client: QdrantClient,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Busca vetorial para múltiplas queries (query expansion).
    Retorna resultados únicos por conteúdo.
    """
    seen = set()
    all_results = []

    for query in queries:
        results = vector_search(query, client, top_k)
        for r in results:
            key = r.content[:100]  # fingerprint pelo início do conteúdo
            if key not in seen:
                seen.add(key)
                all_results.append(r)

    return all_results