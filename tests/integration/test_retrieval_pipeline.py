from retrieval.retrievers.retrieve_dense import retrieve_dense
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid

def test_dense_pipeline_end_to_end():
    results = retrieve_dense("What is Gemma?", top_k=3)
    assert len(results) > 0

def test_hybrid_pipeline_end_to_end():
    results = retrieve_hybrid("What is Gemma?", top_k=3)
    assert len(results) > 0
