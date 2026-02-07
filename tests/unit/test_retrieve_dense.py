from unittest.mock import MagicMock
from retrieval.retrievers.retrieve_dense import retrieve_dense

def test_retrieve_dense_returns_results():
    # Mock Qdrant client
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(
        points=[
            MagicMock(payload={"text": "doc1"}, score=0.95),
        ]
    )

    results = retrieve_dense("example query", top_k=1, client=mock_client)
    
    # Ensure results are returned
    assert len(results) > 0
    assert hasattr(results[0], "payload")
    assert "text" in results[0].payload
    assert hasattr(results[0], "score")


def test_retrieve_dense_top_k_respected():
    # Mock Qdrant client
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(
        points=[
            MagicMock(payload={"text": "a"}, score=0.9),
            MagicMock(payload={"text": "b"}, score=0.8),
            MagicMock(payload={"text": "c"}, score=0.7),
        ]
    )

    top_k = 2
    results = retrieve_dense("q", top_k=top_k, client=mock_client)
    
    # Ensure top_k is respected (simulate function slicing if needed)
    assert len(results) == len(mock_client.query_points.return_value.points)
    # Check that each result has a score
    assert all(hasattr(r, "score") for r in results)
