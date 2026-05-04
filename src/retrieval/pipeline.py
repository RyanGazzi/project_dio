from qdrant_client import QdrantClient
from loguru import logger

from .query_expander import expand_query
from .vector_search import vector_search_multi
from .bm25_search import bm25_search
from .fusion import reciprocal_rank_fusion
from .reranker import rerank
from .compressor import compress_context


def retrieve(
    query: str,
    client: QdrantClient,
    expand: bool = True,
    compress: bool = True,
    top_k_retrieval: int = 20,
    top_k_final: int = 5,
) -> list:
    """
    Pipeline completo de retrieval:
    query → expansão → busca híbrida → RRF → reranking → compressão
    """

    # 1. Expansão da query
    queries = expand_query(query) if expand else [query]
    logger.info(f"Queries após expansão: {queries}")

    # 2. Busca híbrida
    vector_results = vector_search_multi(queries, client, top_k=top_k_retrieval)
    bm25_results   = bm25_search(query, client, top_k=top_k_retrieval)
    logger.info(f"Vetorial: {len(vector_results)} | BM25: {len(bm25_results)}")

    # 3. Fusão RRF
    fused = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        top_k=top_k_retrieval,
    )
    logger.info(f"Após RRF: {len(fused)} candidatos")

    # 4. Reranking
    reranked = rerank(query, fused, top_k=top_k_final)
    logger.info(f"Após reranking: {len(reranked)} chunks finais")

    # 5. Compressão de contexto (opcional, consome tokens)
    if compress:
        reranked = compress_context(query, reranked)
        logger.info(f"Após compressão: {len(reranked)} chunks")

    return reranked

# --- NOVO ---

def retrieve_debug(
    query: str,
    client: QdrantClient,
    expand: bool = True,
    compress: bool = True,
    top_k_retrieval: int = 20,
    top_k_final: int = 5,
) -> dict:
    """
    Mesma lógica do retrieve(), mas retorna todas as etapas
    para fins de debug e visualização.
    """

    # 1. Query expansion
    queries = expand_query(query) if expand else [query]

    # 2. Busca híbrida
    vector_results = vector_search_multi(queries, client, top_k=top_k_retrieval)
    bm25_results = bm25_search(query, client, top_k=top_k_retrieval)

    # 3. RRF
    fused = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        top_k=top_k_retrieval,
    )

    # 4. Reranking — guarda o que foi descartado
    reranked = rerank(query, fused, top_k=top_k_final)
    reranked_ids = {r.content[:50] for r in reranked}
    descartados_reranking = [
        r for r in fused
        if r.content[:50] not in reranked_ids
    ]

    # 5. Compressão — guarda o que foi descartado
    if compress:
        comprimidos = compress_context(query, reranked)
        comprimidos_ids = {r.content[:50] for r in comprimidos}
        descartados_compressao = [
            r for r in reranked
            if r.content[:50] not in comprimidos_ids
        ]
    else:
        comprimidos = reranked
        descartados_compressao = []

    return {
        "queries": queries,
        "vector_results": vector_results,
        "bm25_results": bm25_results,
        "apos_rrf": fused,
        "apos_reranking": reranked,
        "descartados_reranking": descartados_reranking,
        "apos_compressao": comprimidos,
        "descartados_compressao": descartados_compressao,
    }