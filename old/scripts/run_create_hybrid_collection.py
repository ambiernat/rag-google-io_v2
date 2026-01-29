# scripts/run_create_hybrid_collection.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
)

COLLECTION_NAME = "hybrid_collection"
DENSE_SIZE = 384 # match your embedding model

client = QdrantClient(url="http://localhost:6333")

# Check if collection exists and delete if it does
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")

# Create the collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": VectorParams(
            size=DENSE_SIZE,
            distance=Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "text": SparseVectorParams()
    },
)

print(f"Hybrid collection created: {COLLECTION_NAME}")