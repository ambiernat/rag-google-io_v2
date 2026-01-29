import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from retrieval.retrievers.hybrid_retriever import HybridRetriever

# Load BM25 encoder
with open("bm25_encoder.pkl", "rb") as f:
    bm25_encoder = pickle.load(f)

# TODO: Replace with real embedding model
def get_query_embedding(text):
    return [0.01] * 768

# Query
query = "What are neural networks?"

# Tokenize query
def tokenize(text):
    return text.lower().split()

query_tokens = tokenize(query)

# Get sparse vector
query_sparse = bm25_encoder.encode_query(query_tokens)

# Get dense vector
query_dense = get_query_embedding(query)

# Search
client = QdrantClient(url="http://localhost:6333")
retriever = HybridRetriever(client, "hybrid_collection")

results = retriever.search(
    dense_vector=query_dense,
    sparse_vector=query_sparse,
    limit=5,
)

print(f"\nQuery: {query}\n")
for r in results:
    print(f"Score: {r.score:.4f}")
    print(f"Text: {r.payload['text'][:100]}...")
    print(f"Video ID: {r.payload['video_id']}\n")