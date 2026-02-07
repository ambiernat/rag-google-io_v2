from unittest.mock import patch
import numpy as np
from retrieval.rerankers.crossencoder_reranker import crossencoder_rerank


@patch("retrieval.rerankers.crossencoder_reranker.CrossEncoder")
def test_crossencoder_rerank_basic(mock_ce):
    mock_model = mock_ce.return_value
    mock_model.predict.return_value = np.array([0.2, 0.9])

    docs = [
        {"text": "a"},
        {"text": "b"},
    ]

    results = crossencoder_rerank("q", docs, top_k=1)

    assert len(results) == 1
    assert results[0]["text"] == "b"
    assert "rerank_score" in results[0]


@patch("retrieval.rerankers.crossencoder_reranker.CrossEncoder")
def test_crossencoder_empty_docs(mock_ce):
    results = crossencoder_rerank("q", [])
    assert results == []


@patch("retrieval.rerankers.crossencoder_reranker.CrossEncoder")
def test_crossencoder_top_k_respected(mock_ce):
    mock_model = mock_ce.return_value
    mock_model.predict.return_value = np.array([0.1, 0.3, 0.2])

    docs = [
        {"text": "a"},
        {"text": "b"},
        {"text": "c"},
    ]

    results = crossencoder_rerank("q", docs, top_k=2)

    assert len(results) == 2
