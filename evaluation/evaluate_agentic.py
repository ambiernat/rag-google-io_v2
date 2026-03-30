#!/usr/bin/env python
# coding: utf-8
"""
evaluate_agentic.py
Evaluate the full agentic retrieval pipeline (retry_loop.run()) against a ground truth file.
Follows the same pattern as evaluate_hybrid.py.

Baseline (hybrid, dev/multi_doc.json):
  Recall@K:    1.0000
  MRR:         0.9084
  Precision@K: 0.4513
"""
import json
import logging
import os
import yaml
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from retrieval.agent.retry_loop import run as agentic_run
from retrieval.evaluation.metrics import recall_at_k, precision_at_k, mrr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------
# Baseline for comparison table
# -----------------------------
HYBRID_BASELINE = {
    "recall_at_k":    1.0000,
    "mrr":            0.9084,
    "precision_at_k": 0.4513,
}

# -----------------------------
# Load config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # rag-google-io/
CONFIG_PATH = PROJECT_ROOT / "evaluation" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

TOP_K = config["evaluation"].get("top_k", 5)
GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / config["evaluation"]["ground_truth_dir"]
    / config["evaluation"]["ground_truth_file"]
)

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
OUTPUT_PATH = (
    PROJECT_ROOT
    / config["evaluation"]["output_dir"]
    / f"agentic_eval_{timestamp}.json"
)

logger.info("[INFO] Ground truth:  %s", GROUND_TRUTH_PATH.name)
logger.info("[INFO] Output:        %s", OUTPUT_PATH)
logger.info("[INFO] top_k:         %d", TOP_K)

# -----------------------------
# Load ground truth
# -----------------------------
with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

logger.info("[INFO] Loaded %d queries from ground truth.", len(ground_truth))

# -----------------------------
# Evaluation loop
# -----------------------------
results_all = []

for item in tqdm(ground_truth, desc="Queries"):
    query = item["query"]
    relevant_ids = item.get("relevant_doc_ids", [])

    agent_result = agentic_run(query)

    # Extract doc IDs in LLM-score order (best first), then truncate to TOP_K
    retrieved_ids = [r.doc_id for r in agent_result.top_results[:TOP_K]]

    results_all.append({
        "query":             query,
        "relevant_doc_ids":  relevant_ids,
        "retrieved_doc_ids": retrieved_ids,
        "strategy_used":     agent_result.strategy_used,
        "attempts":          agent_result.attempts,
        "good_enough":       agent_result.good_enough,
        "variants":          agent_result.variants,
        "recall_at_k":       recall_at_k(retrieved_ids, relevant_ids, k=TOP_K),
        "mrr":               mrr(retrieved_ids, relevant_ids),
        "precision_at_k":    precision_at_k(retrieved_ids, relevant_ids, k=TOP_K),
    })

# -----------------------------
# Save results
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results_all, f, indent=2)

logger.info("[OK] Agentic evaluation complete. Results saved to %s", OUTPUT_PATH)

# -----------------------------
# Summary table
# -----------------------------
n = len(results_all)
avg_recall    = sum(r["recall_at_k"]    for r in results_all) / n
avg_mrr       = sum(r["mrr"]            for r in results_all) / n
avg_precision = sum(r["precision_at_k"] for r in results_all) / n

strategy_counts = {}
for r in results_all:
    strategy_counts[r["strategy_used"]] = strategy_counts.get(r["strategy_used"], 0) + 1
good_enough_pct = sum(1 for r in results_all if r["good_enough"]) / n * 100

def _delta(agentic: float, baseline: float) -> str:
    diff = agentic - baseline
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:+.4f}"

print()
print("=" * 62)
print(f"{'Metric':<20} {'Hybrid baseline':>16} {'Agentic':>12} {'Delta':>10}")
print("-" * 62)
print(f"{'Recall@K':<20} {HYBRID_BASELINE['recall_at_k']:>16.4f} {avg_recall:>12.4f} {_delta(avg_recall, HYBRID_BASELINE['recall_at_k']):>10}")
print(f"{'MRR':<20} {HYBRID_BASELINE['mrr']:>16.4f} {avg_mrr:>12.4f} {_delta(avg_mrr, HYBRID_BASELINE['mrr']):>10}")
print(f"{'Precision@K':<20} {HYBRID_BASELINE['precision_at_k']:>16.4f} {avg_precision:>12.4f} {_delta(avg_precision, HYBRID_BASELINE['precision_at_k']):>10}")
print("=" * 62)
print(f"\nQueries evaluated:  {n}")
print(f"Good enough (%):    {good_enough_pct:.1f}%")
print(f"Strategy breakdown: {strategy_counts}")
