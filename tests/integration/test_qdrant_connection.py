from qdrant_client import QdrantClient
import os
# Set default for tests
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333')

def test_qdrant_is_reachable():
    client = QdrantClient(url=QDRANT_URL)
    health = client.get_collections()
    assert health is not None
