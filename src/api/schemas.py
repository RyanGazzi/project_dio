from __future__ import annotations
from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    expand: bool = True
    compress: bool = True


class IngestRequest(BaseModel):
    file_path: str


class IngestResponse(BaseModel):
    file: str
    doc_type: str
    pages: int
    chunks_generated: int


class ChunkDebug(BaseModel):
    chunk_id: str
    content: str
    source: str
    page: int | str
    score: float
    content_type: str | None = None


class RetrievalDebug(BaseModel):
    query_original: str
    queries_expandidas: list[str]
    total_vetorial: int
    total_bm25: int
    apos_rrf: list[ChunkDebug]
    descartados_reranking: list[ChunkDebug]
    apos_reranking: list[ChunkDebug]
    descartados_compressao: list[ChunkDebug]
    apos_compressao: list[ChunkDebug]
    prompt_final: str