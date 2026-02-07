from unittest.mock import MagicMock
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid

def test_retrieve_hybrid_returns_results():
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(
        points=[
            MagicMock(payload={"text": "Gemma explanation", "doc_id": "123"}, score=0.95, id="123")
        ]
    )

    results = retrieve_hybrid("What is Gemma?", top_k=3, client=mock_client)

    # Ensure results are returned
    assert len(results) == 1
    point = results[0]
    assert hasattr(point, "score")
    assert hasattr(point, "payload")
    assert "text" in point.payload
    assert "doc_id" in point.payload
    assert point.payload["text"] == "Gemma explanation"


def test_retrieve_hybrid_scores_present():
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(
        points=[
            MagicMock(payload={"text": "a"}, score=0.9, id="1"),
            MagicMock(payload={"text": "b"}, score=0.8, id="2"),
        ]
    )

    top_k = 2
    results = retrieve_hybrid("q", top_k=top_k, client=mock_client)

    # Ensure top_k points are returned
    assert len(results) == len(mock_client.query_points.return_value.points)
    # Check that each result has a score
    assert all(hasattr(r, "score") for r in results)
    # Check payload text exists
    assert all("text" in r.payload for r in results)
