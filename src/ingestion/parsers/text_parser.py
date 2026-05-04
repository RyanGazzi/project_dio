from pathlib import Path
from dataclasses import dataclass

import pymupdf4llm


@dataclass
class ParsedPage:
    page_number: int
    content: str
    headings: list[str]   # H1/H2 encontrados na página


def parse_text_pdf(pdf_path: Path) -> list[ParsedPage]:
    """
    Converte PDF para Markdown preservando a estrutura (títulos, listas).
    Retorna uma lista de páginas com metadados de heading.
    """
    # page_chunks=True retorna uma lista, um item por página
    page_chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
    )

    parsed = []
    for chunk in page_chunks:
        text = chunk["text"]
        page_num = chunk["metadata"]["page"]

        # Extrai headings presentes na página para enriquecer metadados
        headings = [
            line.lstrip("#").strip()
            for line in text.splitlines()
            if line.startswith("#")
        ]

        parsed.append(ParsedPage(
            page_number=page_num,
            content=text,
            headings=headings,
        ))

    return parsed