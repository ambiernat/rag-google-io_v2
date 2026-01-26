import json
import pickle
from qdrant_client import QdrantClient
from retrieval.retrievers.sparse_indexer import SparseIndexer

# Load documents
with open("data/canonical/all_documents.json", "r", encoding="utf-8") as f:
    all_docs = json.load(f)

# Load BM25 encoder
with open("bm25_encoder.pkl", "rb") as f:
    bm25_encoder = pickle.load(f)

# Tokenize documents (same as when building BM25)
def tokenize(text):
    return text.lower().split()

tokenized_corpus = [tokenize(doc["text"]) for doc in all_docs]

# Encode documents as sparse vectors
sparse_vectors = bm25_encoder.encode_documents(tokenized_corpus)

# Create collection and index
client = QdrantClient(url="http://localhost:6333")
indexer = SparseIndexer(client, collection_name="sparse_collection")
indexer.create_collection()
indexer.index_documents(all_docs, sparse_vectors)

print("Sparse indexing complete!")