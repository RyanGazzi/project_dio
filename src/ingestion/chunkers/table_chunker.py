from dataclasses import dataclass, field

from src.ingestion.parsers.table_parser import ParsedTable


@dataclass
class Chunk:
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)


def _remove_empty_columns(table: list[list]) -> list[list]:
    """Remove colunas onde todas as células são vazias."""
    if not table:
        return table

    num_cols = max(len(row) for row in table)
    cols_to_keep = []

    for col_idx in range(num_cols):
        col_values = [
            str(row[col_idx]).strip() if col_idx < len(row) else ""
            for row in table
        ]
        if any(v for v in col_values):  # mantém se tiver qualquer valor
            cols_to_keep.append(col_idx)

    return [
        [row[i] if i < len(row) else "" for i in cols_to_keep]
        for row in table
    ]

def table_chunk(tables: list[ParsedTable], source_file: str) -> list[Chunk]:
    """
    Cada tabela vira um chunk independente.
    O conteúdo é o markdown da tabela com um cabeçalho de contexto
    para o LLM entender o que está lendo.
    """
    chunks = []

    for table in tables:
        # Adiciona contexto antes da tabela para o LLM entender
        content = (
            f"Tabela extraída da página {table.page_number} "
            f"(tabela {table.table_index + 1}):\n\n"
            f"{table.markdown}"
        )

        chunks.append(Chunk(
            chunk_id=f"{source_file}::table::p{table.page_number}::t{table.table_index}",
            content=content,
            metadata={
                "source": source_file,
                "page": table.page_number,
                "table_index": table.table_index,
                "chunk_strategy": "table",
                "content_type": "table",
            }
        ))

    return chunks