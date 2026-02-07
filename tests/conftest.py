import os
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Set QDRANT_URL for all tests before any imports
os.environ.setdefault('QDRANT_URL', 'http://localhost:6333')

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables"""
    os.environ['QDRANT_URL'] = 'http://localhost:6333'

@pytest.fixture(scope="session")
def test_query():
    return "What is Gemma?"
