#!/usr/bin/env python
# coding: utf-8
"""
retrieve_dense.py
Function-based dense retriever using SentenceTransformers + Qdrant.
No interactive input. Designed for evaluation pipelines.
"""
from pathlib import Path
import yaml
from qdrant_client import QdrantClient
from api.models import get_embedding_model
import os

# -------------------------------------------------
# Load config
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rag-google-io/
CONFIG_PATH = PROJECT_ROOT / "retrieval" / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Allow environment variable to override
QDRANT_URL = os.getenv("QDRANT_URL", config["qdrant"]["url"])
DENSE_CFG = config["dense"]
COLLECTION_NAME = DENSE_CFG["collection_name"]
EMBEDDING_MODEL_NAME = DENSE_CFG["embedding_model_name"]
DEFAULT_TOP_K = DENSE_CFG.get("top_k", 5)

# -------------------------------------------------
# Initialize shared state (once per process)
# -------------------------------------------------
q_client = QdrantClient(url=QDRANT_URL)
#embedding_model = get_embedding_model() --> Don't load the model at import!!
#EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

print(f"[INFO] Dense retriever initialized")
print(f"[INFO] Collection: {COLLECTION_NAME}")
print(f"[INFO] Embedding model: {EMBEDDING_MODEL_NAME}")

# -------------------------------------------------
# Helpers
# -------------------------------------------------
_embedding_model = None  # Global variable to hold the loaded model

def embed_query(query: str) -> list:
    """
    Embed a query into a dense vector.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    return _embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

# -------------------------------------------------
# Public API
# -------------------------------------------------
def retrieve_dense(query: str, top_k: int = DEFAULT_TOP_K, client: QdrantClient | None = None):
    """
    Retrieve top-K documents from dense Qdrant collection.
    
    Args:
        query (str): input query
        top_k (int): number of results
    
    Returns:
        List of ScoredPoint objects with .id, .score, .payload attributes
        (compatible with evaluate_dense.py expectations)
    """

    from qdrant_client.models import Prefetch, Document
    
    # Use injected client or fall back to global q_client
    q_client_to_use = client or q_client

    vector = embed_query(query)
    
    response = q_client_to_use.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    
    # Return the points directly (list of ScoredPoint objects)
    # This matches what evaluate_dense.py expects
    return response.points

# -------------------------------------------------
# Example usage (non-interactive)
# -------------------------------------------------
if __name__ == "__main__":
    #model = get_embedding_model()
    test_queries = [
        "What is Gemma?",
        "Who is speaking in the keynote?",
    ]
    #print(f"[INFO] Embedding dim: {model.get_sentence_embedding_dimension()}")
    
    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        results = retrieve_dense(q)
        
        for i, point in enumerate(results, 1):
            print(f"\nResult {i}")
            print("-" * 40)
            print(f"Score: {point.score:.4f}")
            print(f"Doc ID: {point.payload.get('doc_id', point.id)}")
            text = point.payload.get("text", "")
            print(f"Text: {text[:300]}..." if len(text) > 300 else f"Text: {text}")