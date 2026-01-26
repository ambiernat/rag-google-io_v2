# retrieval/evaluation/pipelines.py
import time
import numpy as np
from typing import List, Dict

from retrieval.evaluation.metrics import recall_at_k, mrr

def dense_retrieve(q_client, embed_fn, collection, query, k):
    vector = embed_fn(query)

    response = q_client.query_points(
        collection_name=collection,
        query=vector,
        using="dense",
        limit=k,
        with_payload=True,
    )

    return [
        {
            "id": p.payload["doc_id"],
            "score": p.score,
            "payload": p.payload,
        }
        for p in response.points
    ]


def retrieve_and_rerank(
    query: str,
    retrieve_k: int,
    rerank_k: int,
    retriever_fn,
    reranker,
):
    candidates = retriever_fn(query, retrieve_k)

    if not candidates:
        return []

    return reranker.rerank(
        query=query,
        documents=candidates,
        top_k=rerank_k,
        return_scores=True,
    )


def evaluate_reranking(
    ground_truth,
    retrieve_fn,
    reranker,
    retrieve_k,
    rerank_k,
):
    recalls, mrrs, latencies = [], [], []

    for item in ground_truth:
        start = time.time()

        results = retrieve_and_rerank(
            query=item["query"],
            retrieve_k=retrieve_k,
            rerank_k=rerank_k,
            retriever_fn=retrieve_fn,
            reranker=reranker,
        )

        latencies.append(time.time() - start)
        relevant_ids = item["relevant_doc_ids"]

        recalls.append(recall_at_k(results, relevant_ids, 5))
        mrrs.append(mrr(results, relevant_ids))

    return {
        "recall@5": float(np.mean(recalls)),
        "mrr": float(np.mean(mrrs)),
        "latency_avg": float(np.mean(latencies)),
    }
