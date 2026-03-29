#!/usr/bin/env python
# coding: utf-8
"""
ground_truth_llm_multi-doc.py

Enriches ground truth with multi-doc relevance labels using two-stage candidate generation:
  Stage 1 (Option 5): All chunks from the same video as the original relevant doc
  Stage 2 (Option 1): Union of dense, sparse, hybrid retrieval results
Judge: GPT labels each candidate as relevant or not.
"""

from openai import OpenAI
import json
import yaml
import getpass
from pathlib import Path
from datetime import datetime, UTC
from tqdm import tqdm
import sys
import os
from collections import defaultdict

# -----------------------------
# Config
# -----------------------------
CONFIG_PATH = Path(__file__).resolve().parents[0] / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_DIR = PROJECT_ROOT / config["enrich_gt"]["ground_truth_dir"]
GT_PREFIX = config["enrich_gt"]["ground_truth_prefix"]
TOP_K = config["enrich_gt"].get("top_k", 5)

MODEL_NAME = "gpt-4o-mini"
timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
OUTPUT_PATH = GT_DIR / f"gt_llm_multi-doc_{timestamp}.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Find latest paraphrased ground truth
# -----------------------------
def get_latest_ground_truth(gt_dir: Path, prefix: str) -> Path:
    files = sorted(
        gt_dir.glob(f"{prefix}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not files:
        raise FileNotFoundError(
            f"No ground truth files found in {gt_dir} with prefix '{prefix}'"
        )
    return files[0]

INPUT_PATH = get_latest_ground_truth(GT_DIR, "gt_gpt-5-nano_paraphrased")
print(f"[INFO] Loading ground truth from {INPUT_PATH.name}")

# -----------------------------
# Load all documents (for Option 5 — video-scoped candidates)
# -----------------------------
ALL_DOCS_PATH = PROJECT_ROOT / "data" / "canonical" / "all_documents.json"
with open(ALL_DOCS_PATH) as f:
    all_documents = json.load(f)

# Index by video_id for fast lookup
docs_by_video = defaultdict(list)
for doc in all_documents:
    docs_by_video[doc["video_id"]].append(doc)

print(f"[INFO] Loaded {len(all_documents)} documents across {len(docs_by_video)} videos")

# -----------------------------
# OpenAI client
# -----------------------------
try:
    API_KEY = getpass.getpass("Enter OpenAI API key: ")
except Exception as e:
    print(f"ERROR: {e}")
    raise

client = OpenAI(api_key=API_KEY)

# -----------------------------
# Retrievers (for Option 1 — union of all strategies)
# -----------------------------
os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL", "http://localhost:6333")
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retrievers.retrieve_dense import retrieve_dense
from retrieval.retrievers.retrieve_sparse import retrieve_sparse
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid

# -----------------------------
# Judge function
# -----------------------------
def judge_relevance(query: str, chunk_text: str) -> bool:
    prompt = f"""You are a relevance judge for a retrieval system.

Given a user query and a text chunk from a Google I/O 2025 talk transcript,
decide if the chunk contains information that is relevant to answering the query.

Respond with only "YES" or "NO".

Query: {query}

Chunk:
{chunk_text[:1000]}

Is this chunk relevant to the query?"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("YES")

# -----------------------------
# Candidate generation
# -----------------------------
def get_candidates(query: str, video_id: str, original_relevant: set) -> dict:
    """
    Returns candidate chunks from two stages:
      Stage 1 — all chunks from the same video (Option 5)
      Stage 2 — union of dense, sparse, hybrid retrieval (Option 1)
    Returns dict of {doc_id: chunk_text} excluding already-relevant docs.
    """
    candidates = {}

    # --- Stage 1: all chunks from same video ---
    for doc in docs_by_video.get(video_id, []):
        doc_id = doc["id"]
        if doc_id not in original_relevant:
            candidates[doc_id] = doc["text"]

    # --- Stage 2: union of all retrievers ---
    for retriever_fn in [retrieve_dense, retrieve_sparse, retrieve_hybrid]:
        try:
            points = retriever_fn(query, top_k=TOP_K)
            for point in points:
                doc_id = point.payload.get("doc_id", str(point.id))
                chunk_text = point.payload.get("text", "")
                if doc_id not in original_relevant and doc_id not in candidates:
                    candidates[doc_id] = chunk_text
        except Exception as e:
            print(f"[WARN] Retriever {retriever_fn.__name__} failed: {e}")

    return candidates

# -----------------------------
# Load ground truth
# -----------------------------
with open(INPUT_PATH) as f:
    ground_truth = json.load(f)

print(f"[INFO] Loaded {len(ground_truth)} queries")
print(f"[INFO] Running two-stage candidate generation + GPT judge...")

# -----------------------------
# Enrich ground truth
# -----------------------------
enriched = []
errors = 0
total_candidates_seen = 0
total_new_relevant = 0

for item in tqdm(ground_truth, desc="Enriching"):
    query = item["query"]
    video_id = item["video_id"]
    original_relevant = set(item["relevant_doc_ids"])

    try:
        # Get candidates from both stages
        candidates = get_candidates(query, video_id, original_relevant)
        total_candidates_seen += len(candidates)

        # Judge each candidate
        judged_relevant = set(original_relevant)
        for doc_id, chunk_text in candidates.items():
            try:
                if judge_relevance(query, chunk_text):
                    judged_relevant.add(doc_id)
                    total_new_relevant += 1
            except Exception as e:
                print(f"[ERROR] Judge failed for {doc_id}: {e}")
                errors += 1

        enriched.append({
            **item,
            "relevant_doc_ids": list(judged_relevant),
            "original_relevant_doc_ids": list(original_relevant),
            "num_relevant": len(judged_relevant),
            "num_candidates_judged": len(candidates),
            "enrichment_stages": ["video_scope", "retrieval_union"],
        })

    except Exception as e:
        print(f"[ERROR] Query failed: {query[:50]}...: {e}")
        enriched.append(item)
        errors += 1

# -----------------------------
# Stats
# -----------------------------
avg_relevant = sum(
    e.get("num_relevant", 1) for e in enriched
) / len(enriched)

avg_candidates = sum(
    e.get("num_candidates_judged", 0) for e in enriched
) / len(enriched)

print(f"\n[INFO] Enriched {len(enriched)} queries ({errors} errors)")
print(f"[INFO] Avg candidates judged per query: {avg_candidates:.1f}")
print(f"[INFO] Avg relevant docs per query: {avg_relevant:.2f} (was 1.00)")
print(f"[INFO] Total new relevant docs found: {total_new_relevant}")

# -----------------------------
# Save
# -----------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(enriched, f, indent=2)

print(f"[OK] Saved to {OUTPUT_PATH}")