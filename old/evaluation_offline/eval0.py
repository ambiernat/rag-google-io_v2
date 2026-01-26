import json
from retrieval.vector_search0_2 import vector_search  # your vector search

TOP_K = 5


def recall_at_k(results, relevant_doc_ids):
    retrieved_ids = {
        hit.payload["doc_id"]
        for hit in results
        if hit.payload and "doc_id" in hit.payload
    }
    return int(any(doc_id in retrieved_ids for doc_id in relevant_doc_ids))


def reciprocal_rank(results, relevant_doc_ids):
    """
    Returns reciprocal rank for a single query.
    """
    for rank, hit in enumerate(results, start=1):
        doc_id = hit.payload.get("doc_id")
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def evaluate(ground_truth_path):
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    recalls = []
    reciprocal_ranks = []

    for item in ground_truth:
        question = item["query"]
        relevant_ids = item["relevant_doc_ids"]

        results = vector_search(question, top_k=TOP_K)

        r = recall_at_k(results, relevant_ids)
        rr = reciprocal_rank(results, relevant_ids)

        recalls.append(r)
        reciprocal_ranks.append(rr)

        print("\nQ:", question)
        print(f"Recall@{TOP_K}: {r}")
        print(f"Reciprocal Rank: {rr:.3f}")

    avg_recall = sum(recalls) / len(recalls)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("\n======================")
    print(f"Recall@{TOP_K}: {avg_recall:.3f}")
    print(f"MRR@{TOP_K}:    {mrr:.3f}")
    print("======================")

    return avg_recall, mrr


if __name__ == "__main__":
    evaluate("data/eval/ground_truth_manual.json")