from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient
from pathlib import Path
from loguru import logger

from configs.settings import settings
from src.generation.pipeline import rag_pipeline
from src.ingestion.pipeline import ingest_document
from .schemas import ChatRequest, IngestRequest, IngestResponse

app = FastAPI(title="RAG Chatbot API", version="0.1.0")

# Cliente Qdrant compartilhado
qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """
    Endpoint principal — recebe a query e retorna a resposta em streaming.
    """
    def stream():
        try:
            yield from rag_pipeline(
                query=request.query,
                qdrant_client=qdrant,
                expand=request.expand,
                compress=request.compress,
            )
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            yield f"Erro interno: {str(e)}"

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """
    Ingere um PDF e indexa no Qdrant.
    """
    pdf_path = Path(request.file_path)

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são suportados")

    try:
        result = ingest_document(pdf_path, qdrant)
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Erro na ingestão: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """
    Remove todos os chunks de um documento do Qdrant.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    qdrant.delete(
        collection_name="documents",
        points_selector=Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=filename)
            )]
        )
    )
    return {"deleted": filename}

@app.post("/debug")
def debug(request: ChatRequest):
    """
    Retorna todas as etapas do retrieval para visualização e debug.
    Não gera resposta do LLM — só mostra o que foi recuperado.
    """
    from src.retrieval.pipeline import retrieve_debug
    from src.generation.prompt_builder import build_prompt

    def _format(results: list) -> list:
        return [
            {
                "chunk_id": r.metadata.get("chunk_id", "?"),
                "content": r.content,
                "source": r.metadata.get("source", "?"),
                "page": r.metadata.get("page", "?"),
                "score": round(r.score, 4),
                "content_type": r.metadata.get("content_type", "text"),
            }
            for r in results
        ]

    data = retrieve_debug(
        query=request.query,
        client=qdrant,
        expand=request.expand,
        compress=request.compress,
    )

    # Monta o prompt final para visualização
    prompt_final = build_prompt(request.query, data["apos_compressao"])
    prompt_str = "\n\n".join(
        f"[{m['role'].upper()}]\n{m['content']}"
        for m in prompt_final
    )

    return {
        "query_original": request.query,
        "queries_expandidas": data["queries"][1:],  # sem a original

        "busca": {
            "total_vetorial": len(data["vector_results"]),
            "total_bm25": len(data["bm25_results"]),
        },

        "apos_rrf": {
            "total": len(data["apos_rrf"]),
            "chunks": _format(data["apos_rrf"]),
        },

        "apos_reranking": {
            "total": len(data["apos_reranking"]),
            "chunks": _format(data["apos_reranking"]),
            "descartados": _format(data["descartados_reranking"]),
        },

        "apos_compressao": {
            "total": len(data["apos_compressao"]),
            "chunks": _format(data["apos_compressao"]),
            "descartados": _format(data["descartados_compressao"]),
        },

        "prompt_final": prompt_str,
    }