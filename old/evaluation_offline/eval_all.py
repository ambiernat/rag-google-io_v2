import json
import numpy as np
from tqdm import tqdm

from retrieval_old.retrievers.retrieve_dense_cli import vector_search  # Your vector search function
from sentence_transformers import SentenceTransformer

# For LLM judge
from openai import OpenAI

from pathlib import Path

# ---------------------------
# Config
# ---------------------------
TOP_K = 5
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embedding model (same one used for retrieval)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Initialize OpenAI client
client = OpenAI()  # Ensure OPENAI_API_KEY is set

# Path to your ground truth dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_FILE = PROJECT_ROOT / "data/eval/ground_truth_gpt5nano.json"
OUTPUT_FILE = PROJECT_ROOT / "data/eval/retrieval_llm_scores_with_truth.json"

# ---------------------------
# Helper Functions
# ---------------------------

def compute_cosines(query, results):
    """Compute cosine similarity between query vector and each retrieved document"""
    query_vec = embedding_model.encode(query, normalize_embeddings=True)
    cosines = []
    for hit in results:
        vec = hit.payload.get("vector")  # <-- Make sure vector is stored in payload
        if vec:
            doc_vec = np.array(vec)
            cos_sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            cosines.append(float(cos_sim))
    return cosines


def get_llm_score(query, retrieved_texts):
    """Ask LLM to rate the quality of retrieval"""
    prompt = f"""
    You are an expert evaluating the quality of document retrieval.
    Question: {query}

    Retrieved documents:
    {retrieved_texts}

    Rate how well the retrieved documents answer the question on a scale from 1 (very poor) to 5 (excellent).
    Only respond with a single integer.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        score_text = response.choices[0].message.content.strip()
        score = int(score_text)
        return max(1, min(score, 5))
    except Exception as e:
        print("LLM scoring error:", e)
        return None


def reciprocal_rank(results, relevant_doc_ids):
    """Compute Reciprocal Rank for a single query"""
    for rank, hit in enumerate(results, start=1):
        doc_id = hit.payload.get("doc_id")
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


# ---------------------------
# Main Evaluation Loop
# ---------------------------

def main():
    with open(GROUND_TRUTH_FILE) as f:
        ground_truth = json.load(f)

    results_all = []
    rr_list = []

    for item in tqdm(ground_truth, desc="Evaluating"):
        query = item["query"]
        relevant_doc_ids = item.get("relevant_doc_ids", [])
        true_docs = item.get("true_documents", [])  # <-- Add ground truth text here

        # Retrieve top-K documents
        retrieved_docs = vector_search(query, top_k=TOP_K)

        # Extract text for LLM judge
        retrieved_texts = "\n---\n".join(
            [hit.payload["text"][:1000] for hit in retrieved_docs if "text" in hit.payload]
        )

        # Compute per-document cosine similarities
        per_doc_cos = compute_cosines(query, retrieved_docs)
        avg_cos = float(np.mean(per_doc_cos)) if per_doc_cos else 0.0

        # Ask LLM to judge quality
        llm_score = get_llm_score(query, retrieved_texts)

        # Compute Recall@K
        retrieved_ids = {hit.payload["doc_id"] for hit in retrieved_docs if hit.payload and "doc_id" in hit.payload}
        recall_at_k = int(any(doc_id in retrieved_ids for doc_id in relevant_doc_ids))

        # Compute Reciprocal Rank
        rr = reciprocal_rank(retrieved_docs, relevant_doc_ids)
        rr_list.append(rr)

        # Include original retrieved answers
        retrieved_answers = [
            {
                "doc_id": hit.payload.get("doc_id"),
                "text": hit.payload.get("text", ""),
                "cosine": per_doc_cos[idx] if idx < len(per_doc_cos) else None
            }
            for idx, hit in enumerate(retrieved_docs)
        ]

        # Save per-query result
        results_all.append({
            "query": query,
            "retrieved_doc_ids": list(retrieved_ids),
            "recall_at_5": recall_at_k,
            "per_doc_cosine": per_doc_cos,
            "avg_cosine": avg_cos,
            "reciprocal_rank": rr,
            "llm_score": llm_score,
            "retrieved_answers": retrieved_answers,
            "true_documents": true_docs  # <-- Added the ground truth documents
        })

        print(f"\nQuery: {query}")
        print(f"Recall@{TOP_K}: {recall_at_k}")
        print(f"Avg Cosine: {avg_cos:.3f}")
        print(f"Reciprocal Rank: {rr:.3f}")
        print(f"LLM Score: {llm_score}")

    # Compute Mean Reciprocal Rank
    mrr = float(np.mean(rr_list)) if rr_list else 0.0

    print("\n======================")
    print(f"MRR@{TOP_K}: {mrr:.3f}")
    print("======================")

    # Save all results to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results_all, f, indent=2)

    print(f"\nSaved evaluation results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()