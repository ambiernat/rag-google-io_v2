import requests

BASE_URL = "http://localhost:8080"

def test_search_endpoint_returns_results():
    response = requests.post(
        f"{BASE_URL}/api/search",
        json={"query": "What is Gemma?", "top_k": 3},
        timeout=15
    )

    assert response.status_code == 200

    body = response.json()
    assert "results" in body
    assert len(body["results"]) > 0

def test_search_response_schema():
    response = requests.post(
        f"{BASE_URL}/api/search",
        json={"query": "What is Gemma?", "top_k": 1},
    )

    result = response.json()["results"][0]

    assert "doc_id" in result
    assert "score" in result
    assert "text" in result
