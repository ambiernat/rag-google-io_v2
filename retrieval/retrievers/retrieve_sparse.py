# retrieval/retrievers/retrieve_sparse.py
import pickle
import numpy as np
import json
from pathlib import Path

class SparseRetriever:
    """
    BM25-based sparse retriever using pre-built BM25 encoder.
    Fully independent of Qdrant.
    """
    def __init__(
        self,
        bm25_path: str = "data/models/bm25_encoder.pkl",
        docs_path: str = "data/canonical/all_documents.json"
    ):
        # Load BM25 encoder
        if not Path(bm25_path).exists():
            raise FileNotFoundError(f"BM25 pickle not found at {bm25_path}")
        with open(bm25_path, "rb") as f:
            self.bm25_encoder = pickle.load(f)

        # Load documents
        if not Path(docs_path).exists():
            raise FileNotFoundError(f"Documents file not found at {docs_path}")
        with open(docs_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve top-K documents using BM25.
        
        Args:
            query (str): text query
            top_k (int): number of top results

        Returns:
            List of tuples: (score, document dict)
        """
        query_tokens = query.lower().split()
        scores = self.bm25_encoder.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(scores[i], self.documents[i]) for i in top_indices]


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    retriever = SparseRetriever()
    queries = [
        "Who is speaking in the video?",
        "What is Gemma?"
    ]
    for q in queries:
        print(f"\n=== Query: {q} ===")
        results = retriever.retrieve(q, top_k=5)
        if not results:
            print("No results found.")
        else:
            for i, (score, doc) in enumerate(results, 1):
                text_snippet = doc.get("text", "")[:300]
                print(f"\nResult {i}")
                print("-" * 40)
                print(f"Score: {score:.4f}")
                print(f"Doc ID: {doc.get('doc_id', 'N/A')}")
                print(f"Text: {text_snippet}...")
