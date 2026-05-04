from dataclasses import dataclass, field
from pathlib import Path
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from configs.settings import settings


@dataclass
class Chunk:
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)


def semantic_chunk(
    pages: list,           # lista de ParsedPage
    source_file: str,
    buffer_size: int = 1,           # janela de sentenças para comparar
    breakpoint_threshold: int = 95, # percentil para quebra (mais alto = chunks maiores)
) -> list[Chunk]:
    """
    Chunking semântico: quebra o texto onde o SIGNIFICADO muda,
    não onde termina um número fixo de tokens.
    """
    embed_model = OllamaEmbedding(
        model_name=settings.ollama_embed_model,  # modelo configurado no .env
        base_url=settings.ollama_host,           # host do Ollama (local ou Docker)
    )

    splitter = SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_threshold,
        embed_model=embed_model,
    )

    # Junta todas as páginas em documentos LlamaIndex
    documents = [
        Document(
            text=page.content,
            metadata={
                "source": source_file,
                "page": page.page_number,
                "headings": getattr(page, "headings", []),
            }
        )
        for page in pages
    ]

    nodes = splitter.get_nodes_from_documents(documents)

    return [
        Chunk(
            chunk_id=f"{source_file}::semantic::{i}",
            content=node.get_content(),
            metadata={
                **node.metadata,
                "chunk_index": i,
                "chunk_strategy": "semantic",
                "content_type": "text",
            }
        )
        for i, node in enumerate(nodes)
    ]