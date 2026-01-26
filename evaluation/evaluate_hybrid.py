#!/usr/bin/env python
# coding: utf-8
"""
evaluate_hybrid.py
Evaluate hybrid retrieval (dense + BM25 + Qdrant native fusion) using YAML configuration.
"""
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid  # hybrid retriever
from datetime import datetime, timezone

# -----------------------------
# Helpers
# -----------------------------
def get_latest_ground_truth(gt_dir: Path, prefix: str) -> Path:
    """
    Returns the most recently modified ground truth JSON file
    matching the given prefix in gt_dir.
    """
    files = sorted(
        gt_dir.glob(f"{prefix}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not files:
        raise FileNotFoundError(f"No ground truth files found in {gt_dir}")
    return files[0]

# -----------------------------
# Load config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # rag-google-io/
CONFIG_PATH = PROJECT_ROOT / "evaluation" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

GT_DIR = PROJECT_ROOT / config["evaluation"]["ground_truth_dir"]
GT_PREFIX = config["evaluation"]["ground_truth_prefix"]
TOP_K = config["evaluation"].get("top_k", 5)

GROUND_TRUTH_PATH = get_latest_ground_truth(GT_DIR, GT_PREFIX)

# Prepare timestamped output file
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
OUTPUT_PATH = (
    PROJECT_ROOT
    / config["evaluation"]["output_dir"]
    / f"hybrid_eval_{timestamp}.json"
)

print(f"[INFO] Using ground truth file: {GROUND_TRUTH_PATH.name}")
print(f"[INFO] Results will be saved to: {OUTPUT_PATH}")

# -----------------------------
# Load ground truth
# -----------------------------
with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# -----------------------------
# Metrics
# -----------------------------
def recall_at_k(retrieved_ids, relevant_ids, k=TOP_K):
    return int(any(rid in relevant_ids for rid in retrieved_ids[:k]))

def mrr(retrieved_ids, relevant_ids):
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0

def precision_at_k(retrieved_ids, relevant_ids, k=TOP_K):
    relevant_count = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return relevant_count / k if k > 0 else 0.0

# -----------------------------
# Evaluation loop
# -----------------------------
results_all = []
print(f"[INFO] Evaluating HYBRID retriever for {len(ground_truth)} queries...")

for item in tqdm(ground_truth, desc="Queries"):
    query = item["query"]
    relevant_ids = item.get("relevant_doc_ids", [])
    
    # Retrieve top-K from hybrid Qdrant
    hits = retrieve_hybrid(query, top_k=TOP_K)
    
    # Extract retrieved doc IDs
    retrieved_ids = [hit.payload.get("doc_id", str(hit.id)) for hit in hits]
    
    # Record metrics
    results_all.append({
        "query": query,
        "relevant_doc_ids": relevant_ids,
        "retrieved_doc_ids": retrieved_ids,
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k=TOP_K),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k=TOP_K)
    })

# -----------------------------
# Save results
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results_all, f, indent=2)

print(f"[OK] Hybrid evaluation complete. Results saved to {OUTPUT_PATH}")