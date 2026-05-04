from collections import defaultdict
from .vector_search import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,           # constante de suavização, 60 é o padrão da literatura
    top_k: int = 20,
) -> list[SearchResult]:
    """
    Combina múltiplas listas de resultados rankeados.
    O score RRF de um documento = soma de 1/(k + posição) em cada lista.
    Documentos que aparecem em várias listas ganham score mais alto.
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, SearchResult] = {}

    for result_list in result_lists:
        for position, result in enumerate(result_list):
            key = result.content[:100]
            scores[key] += 1.0 / (k + position + 1)
            doc_map[key] = result

    # Ordena por score RRF decrescente
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused = []
    for key, rrf_score in ranked[:top_k]:
        result = doc_map[key]
        result.score = rrf_score   # substitui score original pelo RRF
        fused.append(result)

    return fused