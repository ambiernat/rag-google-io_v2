# search.py
#This is Search v0 — the baseline retrieval system.

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config
# ---------------------------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "google-io-transcripts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


# ---------------------------
# Init clients
# ---------------------------
q_client = QdrantClient(url=QDRANT_URL)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------
# Vector search function
# ---------------------------

def vector_search(query: str, top_k: int = 5, with_vectors: bool = False):
    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    hits = q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,      # 👈 IMPORTANT: this explicitly requests payloads 

        #You must explicitly request payloads if you want RAG.
        with_vectors=with_vectors    # 👈 IMPORTANT: this explicitly requests vectors - 
    )

    return hits.points


# ---------------------------
# print helper
# ---------------------------
def print_results(results):
    for i, hit in enumerate(results, start=1):

        # Case 1: ScoredPoint
        if hasattr(hit, "payload"):
            payload = hit.payload
            score = hit.score

        # Case 2: tuple (point, score)
        elif isinstance(hit, tuple):
            point, score = hit
            payload = point.payload if hasattr(point, "payload") else None

        else:
            raise ValueError(f"Unexpected result type: {type(hit)}")

        print(f"\nResult {i}")
        print("-" * 40)
        print(f"Score: {score:.4f}")
        print(payload["text"][:500])

# ---------------------------
# CLI entry point
# ---------------------------
if __name__ == "__main__":
    query = input("\nEnter your query: ")
    results = vector_search(query, with_vectors=True)
    print_results(results)
    print(vector_search("test query"))