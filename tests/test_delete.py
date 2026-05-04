from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)

client.delete(
    collection_name="documents",
    points_selector=Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value="SCR_InstrucoesDePreenchimento_Doc3040.pdf")
            )
        ]
    )
)
print("Deletado!")