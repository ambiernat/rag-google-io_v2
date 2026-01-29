import json
import pickle
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import hashlib

# Load documents
with open("data/canonical/all_documents.json", "r", encoding="utf-8") as f:
    all_docs = json.load(f)

# Load BM25 encoder
with open("data/models/bm25_encoder.pkl", "rb") as f:
    bm25_encoder = pickle.load(f)

# Use actual embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(texts):
    """Generate embeddings using sentence-transformers"""
    return embedding_model.encode(texts, show_progress_bar=True).tolist()

# Tokenize
def tokenize(text):
    return text.lower().split()

tokenized_corpus = [tokenize(doc["text"]) for doc in all_docs]

# Get sparse vectors
print("Encoding sparse vectors...")
sparse_vectors = bm25_encoder.encode_documents(tokenized_corpus)

# Get dense embeddings
print("Generating dense embeddings...")
texts = [doc["text"] for doc in all_docs]
dense_embeddings = get_embeddings(texts)

# Index into hybrid collection
client = QdrantClient(url="http://localhost:6333")
COLLECTION = "hybrid_collection"

def string_to_int_id(string_id: str) -> int:
    return int(hashlib.md5(string_id.encode()).hexdigest()[:15], 16)

points = []
for doc, sparse_vec, dense_vec in zip(all_docs, sparse_vectors, dense_embeddings):
    # Use all fields from the original document
    payload = {
        "doc_id": doc["id"],  # Changed from original_id to doc_id
        "text": doc["text"],
        "video_id": doc["video_id"],
    }
    
    # Add optional fields if they exist
    if "title" in doc:
        payload["title"] = doc["title"]
    if "timestamp_start" in doc:
        payload["timestamp_start"] = doc["timestamp_start"]
    if "timestamp_end" in doc:
        payload["timestamp_end"] = doc["timestamp_end"]
    if "source" in doc:
        payload["source"] = doc["source"]
    if "speaker" in doc:
        payload["speaker"] = doc["speaker"]
    
    points.append(
        PointStruct(
            id=string_to_int_id(doc["id"]),
            vector={
                "dense": dense_vec,
                "text": sparse_vec,
            },
            payload=payload,
        )
    )

client.upsert(
    collection_name=COLLECTION,
    points=points,
)

print(f"Hybrid indexing complete! Indexed {len(points)} documents")