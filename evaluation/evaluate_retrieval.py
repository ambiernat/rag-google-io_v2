#!/usr/bin/env python
# coding: utf-8

import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from retrieval.retrieve_sparse import retrieve_sparse
from retrieval.retrieve_dense import retrieve_dense
from retrieval.retrieve_hybrid import hybrid_retrieve

# -----------------------------
# Load config
# -----------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

TOP_K_DEFAULT = 10
GROUND_TRUTH_PATH = "../data/eval/ground_truth_gpt5nano.json"
OUTPUT_PATH = "../data/eval/retrieval_results.json"

# -----------------------------
# Load ground truth
# -----------------------------
with open(GROUND_TRUTH_PATH) as f:
    ground_truth = json.load(f)

# -----------------------------
# Metrics
# -----------------------------
def recall_at_k(retrieved_ids, relevant_ids):
    return int(any(rid in relevant_ids for rid in retrieved_ids))

def mrr(retrieved_ids, relevant_ids):
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0

def precision_at_k(retrieved_ids, relevant_ids, k):
    relevant_count = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return relevant_count / k if k > 0 else 0.0

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    y_true = [int(rid in relevant_ids) for rid in retrieved_ids[:k]]
    y_scores = np.linspace(1, 0.1, len(y_true)) if y_true else [0]*k
    return np.array([y_true]).astype(int)
    # Use sklearn's ndcg_score if needed later

# -----------------------------
# Evaluation loop
# -----------------------------
METHODS = ["sparse", "dense", "hybrid"]
results_all = []

for method in METHODS:
    print(f"\nEvaluating {method.upper()} retriever...")
    for item in tqdm(ground_truth, desc=f"{method} queries"):
        query = item["query"]
        relevant_ids = item.get("relevant_doc_ids", [])

        if method == "sparse":
            hits = retrieve_sparse(query)
        elif method == "dense":
            hits = retrieve_dense(query)
        elif method == "hybrid":
            hits = hybrid_retrieve(query)
        else:
            raise ValueError(f"Unknown retrieval method {method}")

        retrieved_ids = [h.payload["doc_id"] for h in hits]

        results_all.append({
            "method": method,
            "query": query,
            "relevant_doc_ids": relevant_ids,
            "retrieved_doc_ids": retrieved_ids,
            "recall_at_5": recall_at_k(retrieved_ids, relevant_ids),
            "mrr_at_k": mrr(retrieved_ids, relevant_ids),
            "precision_at_5": precision_at_k(retrieved_ids, relevant_ids, k=5)
        })

# -----------------------------
# Save results
# -----------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(results_all, f, indent=2)

print(f"Evaluation complete. Results saved to {OUTPUT_PATH}")
