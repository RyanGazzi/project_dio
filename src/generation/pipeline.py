from collections.abc import Generator
from qdrant_client import QdrantClient
from loguru import logger

from src.retrieval.pipeline import retrieve
from .prompt_builder import build_prompt
from .generator import generate_streaming


def rag_pipeline(
    query: str,
    qdrant_client: QdrantClient,
    expand: bool = True,
    compress: bool = True,
) -> Generator[str, None, None]:
    """
    Pipeline RAG completo com streaming:
    query → retrieval → prompt → geração
    """
    # 1. Retrieval
    logger.info(f"Query recebida: '{query}'")
    chunks = retrieve(
        query=query,
        client=qdrant_client,
        expand=expand,
        compress=compress,
    )

    if not chunks:
        yield "Não encontrei informações relevantes nos documentos para responder essa pergunta."
        return

    # 2. Monta prompt com os chunks
    messages = build_prompt(query, chunks)

    # 3. Geração com streaming
    yield from generate_streaming(messages)