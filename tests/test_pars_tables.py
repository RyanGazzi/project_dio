import pdfplumber
from pathlib import Path

pdf_path = Path("src/ingestion/pdfs/SCR_InstrucoesDePreenchimento_Doc3040.pdf")

pages_to_check = [70, 93, 45]

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        if page.page_number not in pages_to_check:
            continue
        
        raw_tables = page.extract_tables()
        print(f"\n=== Página {page.page_number} — {len(raw_tables)} tabela(s) ===")
        
        for idx, raw in enumerate(raw_tables):
            all_cells = [
                str(cell).strip()
                for row in raw
                for cell in row
                if cell
            ]
            short_cells = [c for c in all_cells if len(c) <= 2]
            ratio = len(short_cells) / len(all_cells) if all_cells else 1
            
            print(f"  Tabela {idx}: {len(all_cells)} células | {len(short_cells)} curtas | ratio: {ratio:.2f} | baixa qualidade: {ratio > 0.4}")

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        if page.page_number not in pages_to_check:
            continue
        
        raw_tables = page.extract_tables(table_settings={
            "vertical_strategy": "text",
            "horizontal_strategy": "lines",
            "snap_tolerance": 5,
            "join_tolerance": 3,
            "edge_min_length": 10,
            "min_words_vertical": 1,
        })
        print(f"\n=== Página {page.page_number} — {len(raw_tables)} tabela(s) com TABLE_SETTINGS ===")
        
        for idx, raw in enumerate(raw_tables):
            all_cells = [
                str(cell).strip()
                for row in raw
                for cell in row
                if cell
            ]
            short_cells = [c for c in all_cells if len(c) <= 2]
            ratio = len(short_cells) / len(all_cells) if all_cells else 1
            print(f"  Tabela {idx}: {len(all_cells)} células | {len(short_cells)} curtas | ratio: {ratio:.2f} | baixa qualidade: {ratio > 0.4}")