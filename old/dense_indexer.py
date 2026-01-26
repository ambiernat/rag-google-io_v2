from typing import List, Dict
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

class DenseRetriever:
    """
    Dense-only retriever (embeddings).
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

    def _string_to_int_id(self, string_id: str) -> int:
        return int(hashlib.md5(string_id.encode()).hexdigest()[:15], 16)

    def create_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                )
            },
        )
        print(f"Created dense collection: {self.collection_name}")

    def index_documents(self, docs: List[Dict], embeddings: List[List[float]]):
        """
        docs[i] corresponds to embeddings[i]
        """
        points = []

        for doc, emb in zip(docs, embeddings):
            points.append(
                PointStruct(
                    id=self._string_to_int_id(doc["id"]),
                    vector={"dense": emb},
                    payload={
                        "text": doc["text"],
                        "video_id": doc["video_id"],
                        "original_id": doc["id"],
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(f"Indexed {len(points)} dense documents")