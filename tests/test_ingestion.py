from pathlib import Path
from qdrant_client import QdrantClient
from src.ingestion.pipeline import ingest_document

client = QdrantClient(host='localhost', port=6333)
result = ingest_document(Path('src/ingestion/pdfs/SCR_InstrucoesDePreenchimento_Doc3040.pdf'), client)
print(result)
