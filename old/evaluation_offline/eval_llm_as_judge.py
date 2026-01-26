from openai import OpenAI
import json
from retrieval_old.vector_search0_2 import vector_search

TOP_K = 5
client = OpenAI()
JUDGE_MODEL = "gpt-5-nano"


def judge_chunk(query: str, chunk_text: str) -> int:
    """
    Returns relevance score:
    0 = not relevant
    1 = partially relevant
    2 = clearly relevant
    """

    prompt = f"""
You are evaluating a semantic search system.

Query:
{query}

Retrieved text chunk:
\"\"\"{chunk_text}\"\"\"

Question:
Does the retrieved chunk help answer the query?

Respond ONLY with a single integer:
0 = not relevant
1 = partially relevant
2 = clearly relevant
"""

    response = client.responses.create(
        model=JUDGE_MODEL,
        input=prompt
    )

    try:
        score = int(response.output_text.strip())
        return score if score in {0, 1, 2} else 0
    except:
        return 0


def evaluate_llm_judge(ground_truth_path):
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    max_scores = []
    avg_scores = []

    for item in ground_truth:
        query = item["query"]

        results = vector_search(query, top_k=TOP_K)

        scores = []
        for hit in results:
            text = hit.payload["text"]
            score = judge_chunk(query, text)
            scores.append(score)

        max_score = max(scores) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0

        max_scores.append(max_score)
        avg_scores.append(avg_score)

        print("\nQ:", query)
        print(f"LLM Judge Scores (top-{TOP_K}): {scores}")
        print(f"Max relevance: {max_score}")
        print(f"Avg relevance: {avg_score:.2f}")

    print("\n======================")
    print(f"Avg Max Relevance: {sum(max_scores)/len(max_scores):.3f}")
    print(f"Avg Relevance@{TOP_K}: {sum(avg_scores)/len(avg_scores):.3f}")
    print("======================")

    return max_scores, avg_scores


if __name__ == "__main__":
    evaluate_llm_judge("data/eval/ground_truth_gpt5nano.json")