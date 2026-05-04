import pdfplumber
from pathlib import Path
from dataclasses import dataclass
from loguru import logger


@dataclass
class ParsedTable:
    page_number: int
    table_index: int
    markdown: str
    raw: list[list]


TABLE_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "lines",
    "snap_tolerance": 5,
    "join_tolerance": 3,
    "edge_min_length": 10,
    "min_words_vertical": 1,
}


def parse_tables(pdf_path: Path) -> list[ParsedTable]:
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_tables = page.extract_tables()

            if raw_tables and _is_fragmented(raw_tables):
                raw_tables = page.extract_tables(table_settings=TABLE_SETTINGS)

            for idx, raw in enumerate(raw_tables):
                if not raw or len(raw) < 2:
                    continue

                raw = [row for row in raw if any(cell for cell in row)]
                raw = _remove_empty_columns(raw)

                if _is_low_quality(raw):
                    logger.warning(
                        f"Tabela descartada por baixa qualidade: "
                        f"página {page.page_number}, tabela {idx}"
                    )
                    continue

                markdown = _table_to_markdown(raw)
                if not markdown:
                    continue

                tables.append(ParsedTable(
                    page_number=page.page_number,
                    table_index=idx,
                    markdown=markdown,
                    raw=raw,
                ))

    return tables


def _is_fragmented(tables: list[list[list]]) -> bool:
    small_tables = [t for t in tables if len(t) <= 2]
    return len(small_tables) > len(tables) * 0.5


def _is_low_quality(table: list[list]) -> bool:
    all_cells = [
        str(cell).strip()
        for row in table
        for cell in row
        if cell
    ]
    if not all_cells:
        return True

    # Filtro 1: muitas células curtas
    short_cells = [c for c in all_cells if len(c) <= 2]
    ratio = len(short_cells) / len(all_cells)
    if ratio > 0.4:
        return True

    # Filtro 2: muitas células com quebras de linha — texto fragmentado
    multiline_cells = [c for c in all_cells if "\n" in c]
    multiline_ratio = len(multiline_cells) / len(all_cells)
    if multiline_ratio > 0.3:
        return True

    return False


def _remove_empty_columns(table: list[list]) -> list[list]:
    if not table:
        return table

    num_cols = max(len(row) for row in table)
    cols_to_keep = []

    for col_idx in range(num_cols):
        col_values = [
            str(row[col_idx]).strip() if col_idx < len(row) else ""
            for row in table
        ]
        # corrigido: checa strip() explicitamente para ignorar células com só espaços
        if any(v.strip() for v in col_values):
            cols_to_keep.append(col_idx)

    return [
        [row[i] if i < len(row) else "" for i in cols_to_keep]
        for row in table
    ]


def _table_to_markdown(table: list[list]) -> str:
    if not table:
        return ""

    header = [str(cell or "").strip() for cell in table[0]]
    separator = ["---"] * len(header)
    rows = [
        [str(cell or "").strip() for cell in row]
        for row in table[1:]
    ]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)