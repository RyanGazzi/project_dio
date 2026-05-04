from qdrant_client import QdrantClient
from src.retrieval.pipeline import retrieve

client = QdrantClient(host="localhost", port=6333)

results = retrieve(
    query="Qual a código da modalidade de Home Equity?",
    client=client,
)

for r in results:
    print(f"[score: {r.score:.4f}] [{r.metadata.get('source')}]")
    print(r.content[:300])
    print("---")