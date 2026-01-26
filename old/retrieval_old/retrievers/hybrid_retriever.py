# retrieval/retrievers/hybrid_retriever.py
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, Prefetch

class HybridRetriever:
    """
    Sparse + Dense hybrid retriever.
    """
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
    
    def search(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        limit: int = 5,
    ):
        """
        Qdrant hybrid search:
        - sparse recall (prefetch)
        - dense reranking
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=sparse_vector,
                    using="text",
                    limit=50,
                )
            ],
            query=dense_vector,
            using="dense",
            limit=limit,
        )
        return results.points