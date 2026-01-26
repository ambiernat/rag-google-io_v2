#!/usr/bin/env python
# coding: utf-8

# # Imports & Setup

# In[1]:


from typing import List, Dict
from collections import defaultdict
import json
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from sentence_transformers import SentenceTransformer


# In[2]:


import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# # Connect to Qdrant

# In[3]:


q_client = QdrantClient(url="http://localhost:6333")
DENSE_COLLECTION = "google-io-transcripts"  # If you have dense-only collection
SPARSE_COLLECTION = "sparse_collection"      # If you have sparse-only collection
HYBRID_COLLECTION = "hybrid_collection"      # Recommended for hybrid search


# # Load Models (Dense + Sparse)

# #### Dense Embedding Model

# In[4]:


embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# In[5]:


def embed_query(text: str):
    return embedding_model.encode(text).tolist()


# # Sparse Query Builder

# In[6]:


import pickle
from retrieval.retrievers.BM25 import BM25Encoder


# In[7]:


# In run_index_sparse.py, run_index_hybrid.py, run_query_hybrid.py, and notebook:

with open("../data/models/bm25_encoder.pkl", "rb") as f:
    bm25_encoder = pickle.load(f)


# #### Query Encoding Functions

# In[8]:


def tokenize(text: str):
    """Must match tokenization used during indexing"""
    return text.lower().split()

def bm25_query(query: str):
    """Encode query as sparse vector using BM25"""
    tokens = tokenize(query)
    return bm25_encoder.encode_query(tokens)


# # Dense & Sparse Retrieval Functions

# In[9]:


def dense_retrieve(query: str, k: int = 10):
    """Retrieve using dense embeddings only"""
    vector = embed_query(query)
    hits = q_client.query_points(
        collection_name=DENSE_COLLECTION,
        query=vector,
        limit=k,
    )
    return hits.points


# In[10]:


def sparse_retrieve(query: str, k: int = 10):
    """Retrieve using sparse BM25 vectors only"""
    sparse_vec = bm25_query(query)
    hits = q_client.query_points(
        collection_name=SPARSE_COLLECTION,
        query=sparse_vec,
        using="text",
        limit=k,
    )
    return hits.points


# In[83]:


def hybrid_retrieve_rrf(query: str, k_dense: int = 50, k_sparse: int = 50, k: int = 10):
    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF)
    Retrieves from separate dense and sparse collections, then fuses
    """
    # Get results from both retrievers
    dense_results = dense_retrieve(query, k=k_dense)
    sparse_results = sparse_retrieve(query, k=k_sparse)

    # Apply RRF fusion
    return rrf([dense_results, sparse_results], k_final=k)


# In[86]:


from qdrant_client.models import Prefetch


# In[87]:


def hybrid_retrieve(query: str, k: int = 10):
    """Native hybrid retrieval using Qdrant's built-in hybrid search"""
    dense_vector = embed_query(query)
    sparse_vector = bm25_query(query)

    response = q_client.query_points(
        collection_name=HYBRID_COLLECTION,
        prefetch=[
            Prefetch(
                query=sparse_vector,
                using="text",
                limit=50,
            )
        ],
        query=dense_vector,
        using="dense",
        limit=k,
    )

    # Convert to dictionary format
    results = []
    for point in response.points:
        results.append({
            "id": point.id,
            "score": point.score,
            "payload": point.payload
        })

    return results

# Update the collection name
HYBRID_COLLECTION = "hybrid_collection"


# # Reciprocal Rank Fusion

# In[13]:


def rrf(rankings: List, k_final: int = 10, k_rrf: int = 60):
    """
    Reciprocal Rank Fusion algorithm

    Args:
        rankings: List of result lists from different retrievers
        k_final: Number of final results to return
        k_rrf: RRF constant (typically 60)
    """
    scores = defaultdict(float)
    payloads = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            # Use hit.id for Qdrant point ID
            doc_id = hit.id
            scores[doc_id] += 1.0 / (k_rrf + rank + 1)
            payloads[doc_id] = hit.payload

    # Sort by score and return top k_final
    fused_results = sorted(
        [
            {"id": doc_id, "score": score, "payload": payloads[doc_id]}
            for doc_id, score in scores.items()
        ],
        key=lambda x: x["score"],
        reverse=True
    )

    return fused_results[:k_final]


# # Hybrid Retrieval

# In[84]:


# # Choose Your Hybrid Retrieval Method

# Use this for RRF-based hybrid (works with separate collections)
# hybrid_retrieve = hybrid_retrieve_rrf

# OR use this for native Qdrant hybrid (requires hybrid_collection)
hybrid_retrieve = hybrid_retrieve_native


# ### Manual Sanity Check

# In[20]:


# Check collections
collections = q_client.get_collections()

print("Available collections:")
for collection in collections.collections:
    print(f"  - {collection.name}")


# In[89]:


query = "What is Gemini?"
print(f"Query: {query}\n")

results = hybrid_retrieve(query, k=5)

for i, r in enumerate(results, 1):
    print(f"{i}. Score: {r['score']:.4f}")
    print(f"   Text: {r['payload']['text'][:500]}...")
    print()


# # Evaluation

# In[90]:


# Load ground truth
with open("../data/eval/ground_truth_gpt5nano.json") as f:
    ground_truth = json.load(f)


# In[91]:


print(f"Loaded {len(ground_truth)} evaluation queries")


# #### Evaluation metrics

# In[93]:


# Debug: see what's actually in the payload
query = "What is Gemini?"
results = hybrid_retrieve(query, k=5)

print("First result payload:")
print(results[0]["payload"])
print("\nAvailable keys:")
print(results[0]["payload"].keys())


# In[61]:


# Debug the actual structure
query = "What is Gemini?"
results = hybrid_retrieve(query, k_final=5)

print(f"Type of results: {type(results)}")
print(f"Number of results: {len(results)}")
print(f"\nFirst result type: {type(results[0])}")
print(f"First result: {results[0]}")
print(f"\nFirst result keys: {results[0].keys() if isinstance(results[0], dict) else 'Not a dict'}")


# In[62]:


# Check the payload structure
print(f"Payload keys: {results[0]['payload'].keys()}")
print(f"Full payload: {results[0]['payload']}")


# In[63]:


def get_original_id(result):
    """Helper to get document ID"""
    return result["payload"]["doc_id"]


# In[64]:


def recall_at_k(results: List[Dict], relevant_ids: List[str], k: int) -> int:
    """Binary recall: 1 if any relevant doc in top-k, else 0"""
    retrieved = [r["payload"]["doc_id"] for r in results[:k]]
    return int(any(rid in retrieved for rid in relevant_ids))


# In[65]:


def mrr(results: List[Dict], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant doc"""
    for i, r in enumerate(results, start=1):
        if r["payload"]["doc_id"] in relevant_ids:
            return 1.0 / i
    return 0.0


# In[66]:


def precision_at_k(results: List[Dict], relevant_ids: List[str], k: int) -> float:
    """Proportion of retrieved docs that are relevant"""
    retrieved = [r["payload"]["doc_id"] for r in results[:k]]
    relevant_retrieved = sum(1 for rid in retrieved if rid in relevant_ids)
    return relevant_retrieved / k if k > 0 else 0.0


# # Run evaluation

# In[75]:


# # Debug: Check what's actually being returned
# query = ground_truth[0]["query"]
# print(f"Test query: {query}\n")

# results = hybrid_retrieve(query, k_final=5)

# print(f"Number of results: {len(results)}")
# print(f"\nFirst result:")
# print(f"  Type: {type(results[0])}")
# print(f"  Keys: {results[0].keys() if isinstance(results[0], dict) else 'Not a dict'}")
# print(f"  Full result: {results[0]}")

# if isinstance(results[0], dict) and 'payload' in results[0]:
#     print(f"\nPayload keys: {results[0]['payload'].keys()}")
#     print(f"Full payload: {results[0]['payload']}")


# In[94]:


# In your notebook, run the evaluation
recalls_5 = []
recalls_10 = []
mrrs = []
precisions_5 = []

for idx, item in enumerate(ground_truth):
    query = item["query"]
    relevant_ids = item["relevant_doc_ids"]

    # Retrieve
    results = hybrid_retrieve(query, k=5)

    # Skip if no results
    if not results:
        continue

    # Calculate metrics
    try:
        retrieved_5 = [r["payload"]["doc_id"] for r in results[:5]]
        retrieved_10 = [r["payload"]["doc_id"] for r in results[:10]]

        recalls_5.append(int(any(rid in retrieved_5 for rid in relevant_ids)))
        recalls_10.append(int(any(rid in retrieved_10 for rid in relevant_ids)))

        # MRR
        mrr_score = 0.0
        for i, r in enumerate(results, start=1):
            if r["payload"]["doc_id"] in relevant_ids:
                mrr_score = 1.0 / i
                break
        mrrs.append(mrr_score)

        # Precision@5
        relevant_count = sum(1 for rid in retrieved_5 if rid in relevant_ids)
        precisions_5.append(relevant_count / 5.0)

    except (KeyError, TypeError) as e:
        print(f"Error on query {idx}: {e}")
        continue

print(f"\n{'='*50}")
print("EVALUATION RESULTS")
print(f"{'='*50}")
print(f"Queries evaluated: {len(recalls_5)}/{len(ground_truth)}")
print(f"Recall@5:      {np.mean(recalls_5):.4f}")
print(f"Recall@10:     {np.mean(recalls_10):.4f}")
print(f"MRR:           {np.mean(mrrs):.4f}")
print(f"Precision@5:   {np.mean(precisions_5):.4f}")
print(f"{'='*50}")


# #### Optional: Compare Dense vs Sparse vs Hybrid

# In[95]:


print("\n" + "="*50)
print("ABLATION STUDY")
print("="*50)


# ##### Dense only

# In[99]:


dense_recalls = []
dense_mrrs = []
for item in ground_truth:
    results_dense = dense_retrieve(item["query"], k=5)
    # Convert to same format as RRF results
    results_formatted = [
        {"payload": hit.payload, "score": hit.score}
        for hit in results_dense
    ]
    dense_recalls.append(recall_at_k(results_formatted, item["relevant_doc_ids"], k=5))
    dense_mrrs.append(mrr(results_formatted, item["relevant_doc_ids"]))

print(f"\nDense Only:")
print(f"  Recall@5: {np.mean(dense_recalls):.4f}")
print(f"  MRR:      {np.mean(dense_mrrs):.4f}")


# ##### Sparse only

# In[100]:


sparse_recalls = []
sparse_mrrs = []
for item in ground_truth:
    results_sparse = sparse_retrieve(item["query"], k=5)
    results_formatted = [
        {"payload": hit.payload, "score": hit.score}
        for hit in results_sparse
    ]
    sparse_recalls.append(recall_at_k(results_formatted, item["relevant_doc_ids"], k=5))
    sparse_mrrs.append(mrr(results_formatted, item["relevant_doc_ids"]))


# In[98]:


print(f"\nSparse Only:")
print(f"  Recall@5: {np.mean(sparse_recalls):.4f}")
print(f"  MRR:      {np.mean(sparse_mrrs):.4f}")

print(f"\nHybrid (RRF):")
print(f"  Recall@5: {np.mean(recalls_5):.4f}")
print(f"  MRR:      {np.mean(mrrs):.4f}")


# # Some checks

# In[101]:


# Compare collection sizes
collections = q_client.get_collections()
for coll in collections.collections:
    info = q_client.get_collection(coll.name)
    print(f"{coll.name}: {info.points_count} documents")


# In[102]:


# Dense-only retrieval using the hybrid_collection
def dense_only_retrieve(query: str, k: int = 10):
    """Retrieve using ONLY dense vectors from hybrid_collection"""
    dense_vector = embed_query(query)

    response = q_client.query_points(
        collection_name="hybrid_collection",  # Same collection as hybrid!
        query=dense_vector,
        using="dense",
        limit=k,
    )

    results = []
    for point in response.points:
        results.append({
            "id": point.id,
            "score": point.score,
            "payload": point.payload
        })

    return results

# Evaluate dense-only on hybrid_collection
dense_recalls = []
dense_mrrs = []

for item in ground_truth:
    results = dense_only_retrieve(item["query"], k=10)

    if not results:
        continue

    retrieved_5 = [r["payload"]["doc_id"] for r in results[:5]]

    dense_recalls.append(int(any(rid in retrieved_5 for rid in item["relevant_doc_ids"])))

    mrr_score = 0.0
    for i, r in enumerate(results, start=1):
        if r["payload"]["doc_id"] in item["relevant_doc_ids"]:
            mrr_score = 1.0 / i
            break
    dense_mrrs.append(mrr_score)

print("\n" + "="*60)
print("FAIR COMPARISON (same collection, same data)")
print("="*60)
print(f"\nDense Only (hybrid_collection):")
print(f"  Recall@5: {np.mean(dense_recalls):.4f}")
print(f"  MRR:      {np.mean(dense_mrrs):.4f}")

print(f"\nHybrid (sparse prefetch → dense rerank):")
print(f"  Recall@5: 0.6000")
print(f"  MRR:      0.3667")

print(f"\nDifference:")
print(f"  Recall@5: {np.mean(dense_recalls) - 0.6000:+.4f}")
print(f"  MRR:      {np.mean(dense_mrrs) - 0.3667:+.4f}")
print("="*60)


# #### Comment:
# 
# Excellent! Now we have a clear answer: The hybrid search is hurting performance on this dataset. Dense-only is significantly better (80% vs 60% recall).
# This is happening because the sparse prefetch is filtering out relevant documents before the dense reranking can find them.
# Let's diagnose and fix this:

# In[106]:


# Analyze where hybrid fails vs dense succeeds
query = ground_truth[0]["query"]
relevant_ids = ground_truth[0]["relevant_doc_ids"]

print(f"Query: {query[:100]}...\n")
print(f"Relevant doc IDs: {relevant_ids}\n")

# Dense-only results
dense_results = dense_only_retrieve(query, k=10)
print("=== DENSE-ONLY TOP 10 ===")
for i, r in enumerate(dense_results[:10], 1):
    is_relevant = "✓" if r["payload"]["doc_id"] in relevant_ids else "✗"
    print(f"{i}. {is_relevant} {r['payload']['doc_id']} (score: {r['score']:.4f})")

# Sparse-only results from hybrid_collection
sparse_vector = bm25_query(query)
sparse_results = q_client.query_points(
    collection_name="hybrid_collection",  # Use hybrid_collection
    query=sparse_vector,
    using="text",
    limit=50,
)
print("\n=== SPARSE TOP 50 ===")
relevant_in_sparse = []
for i, hit in enumerate(sparse_results.points[:50], 1):
    is_relevant = "✓" if hit.payload["doc_id"] in relevant_ids else "✗"
    if is_relevant == "✓":
        relevant_in_sparse.append(i)
        print(f"{i}. {is_relevant} {hit.payload['doc_id']} (score: {hit.score:.4f})")

if not relevant_in_sparse:
    print("❌ NO RELEVANT DOCS IN SPARSE TOP-50!")
else:
    print(f"\n✓ Relevant docs found at positions: {relevant_in_sparse}")

# Hybrid results
hybrid_results = hybrid_retrieve(query, k=10)
print("\n=== HYBRID TOP 10 ===")
for i, r in enumerate(hybrid_results[:10], 1):
    is_relevant = "✓" if r["payload"]["doc_id"] in relevant_ids else "✗"
    print(f"{i}. {is_relevant} {r['payload']['doc_id']} (score: {r['score']:.4f})")


# In[ ]:




