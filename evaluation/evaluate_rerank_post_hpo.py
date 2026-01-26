#!/usr/bin/env python
# coding: utf-8

"""
rerank_eval.py

Reusable script to rerank the latest dense retrieval evaluation JSON file
using CrossEncoder. Automatically picks the most recent HPO best params if available.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict
import numpy as np
from sentence_transformers import CrossEncoder

# -----------------------------
# Project root
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # one level up

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Paths
# -----------------------------
EVAL_RESULTS_DIR = PROJECT_ROOT / "data" / "eval" / "results"
HPO_RESULTS_DIR = EVAL_RESULTS_DIR / "hpo"

# -----------------------------
# Utilities
# -----------------------------
def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def extract_text(doc: Dict) -> str:
    if "text" in doc:
        return doc["text"]
    if "payload" in doc and "text" in doc["payload"]:
        return doc["payload"]["text"]
    return str(doc)

def get_latest_file_by_timestamp(directory: Path, pattern: str) -> Path:
    files = list(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {directory} matching {pattern}")
    # Extract timestamp from filename and sort
    files_sorted = sorted(
        files,
        key=lambda f: f.stem.split("_")[-1],  # assumes timestamp is last part after last underscore
        reverse=True
    )
    return files_sorted[0]

# -----------------------------
# Reranking
# -----------------------------
def crossencoder_rerank(
    query: str,
    documents: List[Dict],
    model_name: str,
    top_k: int = 10,
    device: str = "cpu",
    batch_size: int = 16,
) -> List[Dict]:
    if not documents:
        return []

    model = CrossEncoder(model_name, device=device)
    texts = [extract_text(doc) for doc in documents]
    pairs = [[query, t] for t in texts]
    scores = model.predict(pairs, batch_size=batch_size)

    sorted_idx = np.argsort(scores)[::-1][:top_k]
    reranked = []
    for idx in sorted_idx:
        doc = documents[idx].copy()
        doc["rerank_score"] = float(scores[idx])
        reranked.append(doc)

    return reranked

# -----------------------------
# Main
# -----------------------------
def main():
    # -----------------------------
    # Latest dense eval file
    # -----------------------------
    eval_file = get_latest_file_by_timestamp(EVAL_RESULTS_DIR, "dense_eval_*.json")
    eval_data = load_json(eval_file)
    logger.info(f"Loaded {len(eval_data)} queries from {eval_file.name}")

    # -----------------------------
    # Latest HPO best params (optional)
    # -----------------------------
    try:
        best_params_file = get_latest_file_by_timestamp(HPO_RESULTS_DIR, "best_params_*.json")
        best_params = load_json(best_params_file)
        rerank_model = best_params.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        top_k = best_params.get("rerank_k", 10)
        logger.info(f"Using HPO best params from {best_params_file.name}: model={rerank_model}, top_k={top_k}")
    except FileNotFoundError:
        rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        top_k = 10
        logger.info(f"No HPO best params found. Using default: model={rerank_model}, top_k={top_k}")

    # -----------------------------
    # Rerank all queries
    # -----------------------------
    reranked_results = []

    for item in eval_data:
        query = item["query"]
        retrieved_docs = item.get("retrieved_docs", [])
        if not retrieved_docs and "retrieved_doc_ids" in item:
            # fallback: wrap ids as dicts
            retrieved_docs = [{"id": doc_id} for doc_id in item["retrieved_doc_ids"]]

        reranked_docs = crossencoder_rerank(
            query=query,
            documents=retrieved_docs,
            model_name=rerank_model,
            top_k=top_k,
            device="cpu",
            batch_size=16
        )

        reranked_results.append({
            "query": query,
            "reranked_docs": reranked_docs,
            "relevant_doc_ids": item.get("relevant_doc_ids", [])
        })

    # -----------------------------
    # Save reranked results
    # -----------------------------
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_file = EVAL_RESULTS_DIR / f"reranked_dense_{timestamp}.json"
    save_json(reranked_results, output_file)
    logger.info(f"Reranked results saved to {output_file}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
