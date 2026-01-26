#!/usr/bin/env python
# coding: utf-8
"""
metrics.py

Common evaluation metrics for retrieval and reranking:
- recall@k
- precision@k
- mean reciprocal rank (MRR)
"""

from typing import List, Optional

# -----------------------------
# Recall @ K
# -----------------------------
def recall_at_k(
    retrieved_ids: List[str], 
    relevant_ids: List[str], 
    k: Optional[int] = None
) -> float:
    """
    Recall at K: whether any relevant document is retrieved in the top-K.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Cutoff rank (if None, use all retrieved)

    Returns:
        1.0 if any relevant doc is in top-k, else 0.0
    """
    if k is None:
        k = len(retrieved_ids)
    top_k = retrieved_ids[:k]
    return float(any(doc_id in relevant_ids for doc_id in top_k))


# -----------------------------
# Precision @ K
# -----------------------------
def precision_at_k(
    retrieved_ids: List[str], 
    relevant_ids: List[str], 
    k: Optional[int] = None
) -> float:
    """
    Precision at K: proportion of top-K retrieved docs that are relevant.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Cutoff rank (if None, use all retrieved)

    Returns:
        Precision@k score
    """
    if k is None:
        k = len(retrieved_ids)
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_count / k


# -----------------------------
# Mean Reciprocal Rank (MRR)
# -----------------------------
def mrr(
    retrieved_ids: List[str], 
    relevant_ids: List[str]
) -> float:
    """
    Mean Reciprocal Rank (for a single query):
    1 / rank of first relevant document in retrieved list.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs

    Returns:
        Reciprocal rank (0.0 if no relevant doc found)
    """
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0
