from dataclasses import dataclass
from loguru import logger
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from configs.settings import settings
import uuid

COLLECTION_NAME = "documents"
VECTOR_SIZE = 768  # nomic-embed-text


def get_embed_model() -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=settings.ollama_embed_model,
        base_url=settings.ollama_host,
    )


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Coleção '{COLLECTION_NAME}' criada.")


def embed_and_store(chunks: list, client: QdrantClient) -> int:
    embed_model = get_embed_model()
    ensure_collection(client)

    points = []
    for chunk in chunks:
        content = chunk.content.strip()  # ← define content aqui, uma vez só

        if len(content) < 80:
            logger.warning(f"Chunk ignorado por tamanho: '{content[:30]}'")
            continue

        words = [w for w in content.split() if len(w) > 2]
        if len(words) < 8:
            logger.warning(f"Chunk ignorado por poucas palavras: '{content[:30]}'")
            continue

        vector = embed_model.get_text_embedding(content)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "content": content,
                "chunk_id": chunk.chunk_id,
                **chunk.metadata,
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"{len(points)} chunks inseridos no Qdrant.")
    return len(points)