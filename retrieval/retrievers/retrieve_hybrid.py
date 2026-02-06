#!/usr/bin/env python
# coding: utf-8
"""
retrieve_hybrid.py
Function-based hybrid retriever using dense + BM25 vectors in Qdrant.
No interactive input. Designed for evaluation pipelines.
Mirrors retrieve_dense.py structure and behavior.
"""
from pathlib import Path
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Document
from api.models import get_embedding_model

# -------------------------------------------------
# Load config
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rag-google-io/
CONFIG_PATH = PROJECT_ROOT / "retrieval" / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

QDRANT_URL = config["qdrant"]["url"]
HYBRID_CFG = config["hybrid"]
COLLECTION_NAME = HYBRID_CFG["collection_name"]
EMBEDDING_MODEL_NAME = HYBRID_CFG["embedding_model_name"]
DEFAULT_TOP_K = HYBRID_CFG.get("top_k", 5)
SPARSE_PREFETCH_K = HYBRID_CFG.get("sparse_prefetch_k", 50)

# -------------------------------------------------
# Initialize shared state (once per process)
# -------------------------------------------------
q_client = QdrantClient(url=QDRANT_URL)
# embedding_model = get_embedding_model()
# EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

print(f"[INFO] Hybrid retriever initialized")
print(f"[INFO] Collection: {COLLECTION_NAME}")
print(f"[INFO] Embedding model: {EMBEDDING_MODEL_NAME}")
# print(f"[INFO] Embedding dim: {EMBEDDING_DIM}")

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
def retrieve_hybrid(query: str, top_k: int = DEFAULT_TOP_K):
    """
    Retrieve top-K documents from hybrid Qdrant collection using native fusion.
    
    Args:
        query (str): input query
        top_k (int): number of results
    
    Returns:
        List of ScoredPoint objects with .id, .score, .payload attributes
        (compatible with evaluate_hybrid.py expectations)
    """
    from qdrant_client.models import Prefetch, Document
    
    vector = embed_query(query)
    
    # Get fusion type from config
    fusion_type = HYBRID_CFG.get("fusion", {}).get("type", "native")
    
    if fusion_type == "native":
        # Native fusion: use dense as primary query, BM25 as prefetch
        response = q_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,           # Dense vector as primary
            using="dense",
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            prefetch=[
                Prefetch(
                    query=Document(
                        text=query,
                        model="Qdrant/bm25"
                    ),
                    using="bm25",
                    limit=SPARSE_PREFETCH_K,
                )
            ],
        )
    
    elif fusion_type == "rrf":
        # RRF fusion: both methods as prefetch, then fuse
        from qdrant_client.models import FusionQuery, Fusion
        
        response = q_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=vector,
                    using="dense",
                    limit=SPARSE_PREFETCH_K,
                ),
                Prefetch(
                    query=Document(
                        text=query,
                        model="Qdrant/bm25"
                    ),
                    using="bm25",
                    limit=SPARSE_PREFETCH_K,
                )
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
    
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return response.points

# -------------------------------------------------
# Example usage (non-interactive)
# -------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "What is Gemma?",
        "Who is speaking in the keynote?",
    ]
    
    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        results = retrieve_hybrid(q)
        
        for i, point in enumerate(results, 1):
            print(f"\nResult {i}")
            print("-" * 40)
            print(f"Score: {point.score:.4f}")
            print(f"Doc ID: {point.payload.get('doc_id', point.id)}")
            text = point.payload.get("text", "")
            print(f"Text: {text[:300]}..." if len(text) > 300 else f"Text: {text}")