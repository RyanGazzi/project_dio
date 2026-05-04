from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from .vector_search import SearchResult

COLLECTION_NAME = "documents"


def _fetch_all_documents(client: QdrantClient) -> list[dict]:
    """
    Busca todos os documentos do Qdrant para montar o índice BM25.
    Em produção isso seria cacheado — não se busca tudo a cada query.
    """
    results, offset = [], None

    while True:
        batch, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        results.extend(batch)
        if next_offset is None:
            break
        offset = next_offset

    return results


def bm25_search(
    query: str,
    client: QdrantClient,
    top_k: int = 10,
) -> list[SearchResult]:
    docs = _fetch_all_documents(client)

    # Tokeniza cada documento (simples, por espaço)
    tokenized_docs = [
        doc.payload["content"].lower().split()
        for doc in docs
    ]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)

    # Pega os top_k índices com maior score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        SearchResult(
            content=docs[i].payload["content"],
            metadata={k: v for k, v in docs[i].payload.items() if k != "content"},
            score=float(scores[i]),
            source="bm25",
        )
        for i in top_indices
        if scores[i] > 0  # ignora documentos sem nenhuma palavra em comum
    ]