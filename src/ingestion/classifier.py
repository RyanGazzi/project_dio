#imports
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pdfplumber
import pymupdf4llm


#possiveis estruturas de documentos
class DocType(str, Enum):
    TEXT = "text"          # texto corrido, artigos, manuais
    TECHNICAL = "technical" # seções H1/H2 claras, estruturado
    TABULAR = "tabular"    # maioria do conteúdo em tabelas
    SCANNED = "scanned"    # PDF sem texto selecionável (imagem)
    MIXED = "mixed"        # combinação dos tipos acima


#classe que vai armazenar as características do PDF
@dataclass
class ClassificationResult:
    doc_type: DocType
    has_tables: bool
    has_text: bool
    is_scanned: bool
    page_count: int
    avg_chars_per_page: float
    table_density: float   # proporção de páginas com tabelas


def classify_document(pdf_path: Path) -> ClassificationResult:
    
    #lendo o documento pdf
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        pages_with_tables = 0
        total_chars = 0

        for page in pdf.pages:
            text = page.extract_text() or ""
            total_chars += len(text)
            if page.extract_tables():
                pages_with_tables += 1

    avg_chars = total_chars / page_count if page_count else 0
    table_density = pages_with_tables / page_count if page_count else 0
    is_scanned = avg_chars < 100  # muito pouco texto = provavelmente imagem

    #detecta se tem estrutura de seções via markdown do pymupdf4llm
    md_text = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=False)
    heading_count = md_text.count("\n#")
    has_sections = heading_count >= 3

    if is_scanned:
        doc_type = DocType.SCANNED
    elif table_density > 0.5:
        doc_type = DocType.TABULAR
    elif table_density > 0.2 and avg_chars > 500:
        doc_type = DocType.MIXED
    elif has_sections:
        doc_type = DocType.TECHNICAL
    else:
        doc_type = DocType.TEXT

    return ClassificationResult(
        doc_type=doc_type,
        has_tables=pages_with_tables > 0,
        has_text=avg_chars > 100,
        is_scanned=is_scanned,
        page_count=page_count,
        avg_chars_per_page=avg_chars,
        table_density=table_density,
    )