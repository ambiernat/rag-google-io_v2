"""
retrieval/retrievers/dispatcher.py

Unified retrieval dispatcher. Routes by strategy, converts ScoredPoints to
RetrievalResult at the boundary so callers (agent, evaluation) stay
Qdrant-agnostic.
"""

from __future__ import annotations

from typing import Literal

from retrieval.result import RetrievalResult
from retrieval.retrievers.retrieve_dense import retrieve_dense
from retrieval.retrievers.retrieve_sparse import retrieve_sparse
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid

Strategy = Literal["dense", "sparse", "hybrid"]


def retrieve(
    query: str,
    strategy: Strategy = "hybrid",
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    Retrieve documents using the specified strategy.

    Args:
        query:    Input query string.
        strategy: One of "dense", "sparse", "hybrid". Defaults to "hybrid".
        top_k:    Number of results to return.

    Returns:
        List of RetrievalResult objects, ordered by descending relevance score.

    Raises:
        ValueError: If strategy is not one of the accepted values.
    """
    if strategy == "dense":
        points = retrieve_dense(query, top_k=top_k)
    elif strategy == "sparse":
        points = retrieve_sparse(query, top_k=top_k)
    elif strategy == "hybrid":
        points = retrieve_hybrid(query, top_k=top_k)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Must be one of: dense, sparse, hybrid")

    return [RetrievalResult.from_scored_point(p) for p in points]
