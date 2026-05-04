from pathlib import Path
from loguru import logger

from qdrant_client import QdrantClient

from .classifier import classify_document, DocType
from .parsers.text_parser import parse_text_pdf
from .parsers.table_parser import parse_tables
from .parsers.ocr_parser import parse_scanned_pdf
from .chunkers.semantic_chunker import semantic_chunk
from .chunkers.hierarchical_chunker import hierarchical_chunk
from .chunkers.table_chunker import table_chunk
from .embedder import embed_and_store


def ingest_document(pdf_path: Path, qdrant_client: QdrantClient) -> dict:
    """
    Ponto de entrada principal da ingestão.
    Recebe um PDF e persiste tudo no vector store.
    """
    logger.info(f"Iniciando ingestão: {pdf_path.name}")

    # 1. Classificar
    classification = classify_document(pdf_path)
    logger.info(f"Tipo detectado: {classification.doc_type}")

    all_chunks = []

    # 2. Parser + chunker correto para cada tipo
    match classification.doc_type:
        case DocType.SCANNED:
            pages = parse_scanned_pdf(pdf_path)
            chunks = semantic_chunk(pages, pdf_path.name)
            all_chunks.extend(chunks)

        case DocType.TABULAR:
            tables = parse_tables(pdf_path)
            chunks = table_chunk(tables, pdf_path.name)
            all_chunks.extend(chunks)

        case DocType.TECHNICAL:
            pages = parse_text_pdf(pdf_path)
            chunks = hierarchical_chunk(pages, pdf_path.name)
            all_chunks.extend(chunks)

        case DocType.MIXED | DocType.TEXT:
            # Texto: chunking semântico
            pages = parse_text_pdf(pdf_path)
            text_chunks = semantic_chunk(pages, pdf_path.name)
            all_chunks.extend(text_chunks)

            # Tabelas presentes: processa separado
            if classification.has_tables:
                tables = parse_tables(pdf_path)
                tbl_chunks = table_chunk(tables, pdf_path.name)
                all_chunks.extend(tbl_chunks)

    # 3. Embed + persistir
    total = embed_and_store(all_chunks, qdrant_client)

    return {
        "file": pdf_path.name,
        "doc_type": classification.doc_type,
        "pages": classification.page_count,
        "chunks_generated": total,
    }