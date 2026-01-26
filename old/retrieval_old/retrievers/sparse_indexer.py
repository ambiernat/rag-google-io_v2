from qdrant_client.models import SparseVectorParams, PointStruct
import hashlib

class SparseIndexer:
    def __init__(self, client, collection_name):
        self.client = client
        self.collection_name = collection_name
    
    def _string_to_int_id(self, string_id: str) -> int:
        return int(hashlib.md5(string_id.encode()).hexdigest()[:15], 16)
    
    def create_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config={},
            sparse_vectors_config={"text": SparseVectorParams()},
        )
        print(f"Created sparse collection: {self.collection_name}")
    
    def index_documents(self, docs, sparse_vectors):
        """
        docs: list of document dicts
        sparse_vectors: list of SparseVector (already encoded)
        """
        points = []
        for doc, sparse_vector in zip(docs, sparse_vectors):
            points.append(
                PointStruct(
                    id=self._string_to_int_id(doc["id"]),
                    vector={"text": sparse_vector},
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
        print(f"Indexed {len(points)} sparse documents")