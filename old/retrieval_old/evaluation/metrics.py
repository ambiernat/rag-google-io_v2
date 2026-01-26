# retrieval/evaluation/metrics.py
import numpy as np
from typing import List, Dict

def recall_at_k(results: List[Dict], relevant_ids: List[str], k: int) -> int:
    retrieved = [r["payload"]["doc_id"] for r in results[:k]]
    return int(any(rid in retrieved for rid in relevant_ids))

def mrr(results: List[Dict], relevant_ids: List[str]) -> float:
    for i, r in enumerate(results, start=1):
        if r["payload"]["doc_id"] in relevant_ids:
            return 1.0 / i
    return 0.0

def precision_at_k(results, relevant_ids, k):
    retrieved = [r["payload"]["doc_id"] for r in results[:k]]
    return sum(rid in relevant_ids for rid in retrieved) / k if k else 0.0

def average_precision(results, relevant_ids):
    precisions = []
    num_rel = 0

    for i, r in enumerate(results, start=1):
        if r["payload"]["doc_id"] in relevant_ids:
            num_rel += 1
            precisions.append(num_rel / i)

    return float(np.mean(precisions)) if precisions else 0.0
