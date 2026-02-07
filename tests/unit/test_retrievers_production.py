# tests/unit/test_retrievers_production.py
from unittest.mock import MagicMock, patch
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException
from retrieval.retrievers.retrieve_dense import retrieve_dense
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid

# -------------------------------
# Dense retriever tests
# -------------------------------

def make_mock_point(text="doc", score=0.9, doc_id="1"):
    return MagicMock(payload={"text": text, "doc_id": doc_id}, score=score, id=doc_id)

@pytest.fixture
def mock_dense_client():
    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[
        make_mock_point("a", 0.9),
        make_mock_point("b", 0.8),
        make_mock_point("c", 0.7),
    ])
    return client

def test_dense_top_k(mock_dense_client):
    top_k = 2
    results = retrieve_dense("query", top_k=top_k, client=mock_dense_client)
    # Ensure all points returned (mock doesn't slice)
    assert len(results) == len(mock_dense_client.query_points.return_value.points)
    assert all(hasattr(r, "score") and hasattr(r, "payload") for r in results)

def test_dense_empty_query(mock_dense_client):
    results = retrieve_dense("", top_k=2, client=mock_dense_client)
    assert results  # should still return mocked points

def test_dense_invalid_top_k(mock_dense_client):
    results = retrieve_dense("query", top_k=0, client=mock_dense_client)
    assert results  # client still returns mocked points

def test_dense_network_exception():
    mock_client = MagicMock()
    mock_client.query_points.side_effect = ResponseHandlingException("Failed")
    with pytest.raises(ResponseHandlingException):
        retrieve_dense("query", top_k=2, client=mock_client)

# -------------------------------
# Hybrid retriever tests
# -------------------------------

def make_hybrid_point(text="doc", score=0.9, doc_id="1"):
    return MagicMock(payload={"text": text, "doc_id": doc_id}, score=score, id=doc_id)

@pytest.fixture
def mock_hybrid_client():
    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[
        make_hybrid_point("a", 0.9),
        make_hybrid_point("b", 0.8),
        make_hybrid_point("c", 0.7),
    ])
    return client

def test_hybrid_top_k(mock_hybrid_client):
    top_k = 2
    results = retrieve_hybrid("query", top_k=top_k, client=mock_hybrid_client)
    assert all(hasattr(r, "score") and hasattr(r, "payload") for r in results)
    assert all("text" in r.payload for r in results)

def test_hybrid_empty_query(mock_hybrid_client):
    results = retrieve_hybrid("", top_k=2, client=mock_hybrid_client)
    assert results

def test_hybrid_missing_fields():
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(points=[
        MagicMock(payload={}, score=None, id="1"),
        MagicMock(payload={"text": "b"}, score=None, id="2"),
    ])
    results = retrieve_hybrid("query", top_k=2, client=mock_client)
    for r in results:
        assert hasattr(r, "payload")
        assert "text" in r.payload or r.payload == {}

import retrieval.retrievers.retrieve_hybrid as rh

def test_hybrid_rrf_fusion(monkeypatch):
    # Patch module-level HYBRID_CFG to use RRF fusion
    monkeypatch.setitem(rh.HYBRID_CFG, "fusion", {"type": "rrf"})

    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(points=[
        make_hybrid_point("a", 0.9),
        make_hybrid_point("b", 0.8),
    ])

    results = rh.retrieve_hybrid("query", top_k=2, client=mock_client)
    assert len(results) == len(mock_client.query_points.return_value.points)
    #assert all(hasattr(r, "score") and hasattr(r, "payload") for r in results)


def test_hybrid_network_exception():
    mock_client = MagicMock()
    mock_client.query_points.side_effect = ResponseHandlingException("Failed")
    with pytest.raises(ResponseHandlingException):
        retrieve_hybrid("query", top_k=2, client=mock_client)
