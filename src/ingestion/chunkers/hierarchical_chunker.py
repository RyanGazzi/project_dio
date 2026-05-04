from dataclasses import dataclass, field

from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import Document


@dataclass
class HierarchicalChunk:
    chunk_id: str
    content: str
    parent_content: str   # contexto maior enviado ao LLM
    metadata: dict = field(default_factory=dict)


def hierarchical_chunk(
    pages: list,
    source_file: str,
    chunk_sizes: list[int] = [2048, 512, 128],
) -> list[HierarchicalChunk]:
    """
    Cria chunks em múltiplos tamanhos hierárquicos.
    - chunk pequeno (128 tokens): usado no retrieval (precisão)
    - chunk pai (512/2048 tokens): enviado ao LLM (contexto rico)
    """
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)

    documents = [
        Document(
            text=page.content,
            metadata={"source": source_file, "page": page.page_number}
        )
        for page in pages
    ]

    all_nodes = parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)

    # Monta mapa de parent_id → conteúdo do pai
    node_map = {n.node_id: n for n in all_nodes}

    chunks = []
    for i, leaf in enumerate(leaf_nodes):
        parent = node_map.get(leaf.parent_node.node_id) if leaf.parent_node else None
        chunks.append(HierarchicalChunk(
            chunk_id=f"{source_file}::hier::{i}",
            content=leaf.get_content(),
            parent_content=parent.get_content() if parent else leaf.get_content(),
            metadata={
                **leaf.metadata,
                "chunk_index": i,
                "chunk_strategy": "hierarchical",
            }
        ))

    return chunks