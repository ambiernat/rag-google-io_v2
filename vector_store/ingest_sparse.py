#!/usr/bin/env python
# coding: utf-8
"""
ingest_sparse.py
Create and populate a Qdrant collection using native BM25 sparse indexing.
"""
import json
import yaml
import uuid
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "vector_store" / "config.yaml"
CANONICAL_DOCS_PATH = BASE_DIR / "data" / "canonical" / "all_documents.json"

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def string_to_uuid(s: str) -> str:
    """Convert a string to a deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

def main() -> None:
    config = load_config(CONFIG_PATH)
    QDRANT_URL = config["qdrant"]["url"]
    COLLECTION_NAME = config["collections"]["sparse"]["name"]
    SCHEMA_VERSION = config["collections"]["sparse"].get("schema_version", "v1")
    
    print(f"[INFO] Using collection: {COLLECTION_NAME}")
    
    # -----------------------------
    # Load canonical documents
    # -----------------------------
    with open(CANONICAL_DOCS_PATH, "r") as f:
        documents = json.load(f)
    print(f"[INFO] Loaded {len(documents)} canonical documents")
    
    # Calculate average document length for BM25
    avg_doc_length = sum(len(doc["text"].split()) for doc in documents) / len(documents)
    print(f"[INFO] Average document length: {avg_doc_length:.2f} words")
    
    # -----------------------------
    # Connect to Qdrant
    # -----------------------------
    client = QdrantClient(url=QDRANT_URL)
    
    # -----------------------------
    # Recreate collection with sparse BM25 vectors
    # -----------------------------
    if client.collection_exists(COLLECTION_NAME):
        print(f"[WARN] Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(collection_name=COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={},  # No dense vectors
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF  # Enable IDF for BM25
            )
        }
    )
    print(f"[OK] Created sparse BM25 collection: {COLLECTION_NAME}")
    
    # -----------------------------
    # Upload documents with BM25 sparse vectors
    # -----------------------------
    points = []
    for doc in tqdm(documents, desc="Building BM25 points"):
        points.append(
            models.PointStruct(
                id=string_to_uuid(doc["id"]),  # Convert string ID to UUID
                vector={
                    "bm25": models.Document(
                        text=doc["text"],
                        model="Qdrant/bm25",
                        options={"avg_len": avg_doc_length}
                    )
                },
                payload={
                    "doc_id": doc["id"],  # Keep original string ID in payload
                    "text": doc["text"],
                    "video_id": doc["video_id"],
                    "title": doc["title"],
                    "timestamp_start": doc["timestamp_start"],
                    "timestamp_end": doc["timestamp_end"],
                    "source": doc["source"],
                    "speaker": doc["speaker"],
                }
            )
        )
    
    # Upsert in batches for better performance
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=False
        )
    
    print(f"[OK] Indexed {len(points)} documents into sparse BM25 collection")

if __name__ == "__main__":
    main()