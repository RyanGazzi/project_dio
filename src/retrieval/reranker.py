from sentence_transformers import CrossEncoder
from .vector_search import SearchResult


# Carrega uma vez — é um modelo local, não precisa de API
_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Reordena os candidatos usando um cross-encoder.
    Diferente do bi-encoder (embedding), o cross-encoder vê
    query + documento juntos — muito mais preciso, mas mais lento.
    Por isso só roda nos top-20 do RRF, não em tudo.
    """
    if not results:
        return []

    pairs = [[query, r.content] for r in results]
    scores = _model.predict(pairs)

    # Atualiza scores e reordena
    for result, score in zip(results, scores):
        result.score = float(score)

    reranked = sorted(results, key=lambda r: r.score, reverse=True)
    return reranked[:top_k]