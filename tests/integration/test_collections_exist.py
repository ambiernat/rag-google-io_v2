from qdrant_client import QdrantClient
import os
# Set default for tests
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')

EXPECTED_COLLECTIONS = {
    "google-io-transcripts-dense",
    "google-io-transcripts-hybrid",
}

def test_required_collections_exist():
    client = QdrantClient(url=QDRANT_URL)
    collections = client.get_collections()

    names = {c.name for c in collections.collections}

    for expected in EXPECTED_COLLECTIONS:
        assert expected in names
