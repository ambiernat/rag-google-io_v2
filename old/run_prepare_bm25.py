# scripts/run_prepare_bm25.py
import json
import pickle
import os
from retrieval.retrievers.BM25 import BM25Encoder

# Load documents
with open("data/canonical/all_documents.json", "r", encoding="utf-8") as f:
    all_docs = json.load(f)

# Simple tokenization
def tokenize(text):
    return text.lower().split()

# Tokenize all documents
tokenized_corpus = [tokenize(doc["text"]) for doc in all_docs]

# Build BM25 encoder
print(f"Building BM25 encoder with {len(tokenized_corpus)} documents...")
bm25_encoder = BM25Encoder(tokenized_corpus)

# Create directory if it doesn't exist
os.makedirs("data/models", exist_ok=True)

# Save to data/models/
output_path = "data/models/bm25_encoder.pkl"
with open(output_path, "wb") as f:
    pickle.dump(bm25_encoder, f)

print(f"✓ BM25 encoder saved to {output_path}")
print(f"  - Documents: {len(tokenized_corpus)}")
print(f"  - Vocabulary size: {len(bm25_encoder.vocab)}")