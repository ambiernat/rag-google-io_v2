#!/usr/bin/env python
# coding: utf-8
"""
ingest_dense.py
Create and populate a Qdrant collection using dense sentence embeddings.
"""

import json
import yaml
import uuid
from pathlib import Path
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "vector_store" / "config.yaml"
CANONICAL_DOCS_PATH = BASE_DIR / "data" / "canonical" / "all_documents.json"


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def string_to_uuid(s: str) -> str:
    """Convert a string to a deterministic UUID (stable across runs)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


# -------------------------------------------------
# Main
# -------------------------------------------------
def main() -> None:
    # -----------------------------
    # Load config
    # -----------------------------
    config = load_config(CONFIG_PATH)

    QDRANT_URL = config["qdrant"]["url"]
    COLLECTION_NAME = config["collections"]["dense"]["name"]
    EMBEDDING_MODEL_NAME = config["collections"]["dense"]["embedding_model"]

    print(f"[INFO] Using dense collection: {COLLECTION_NAME}")
    print(f"[INFO] Embedding model: {EMBEDDING_MODEL_NAME}")

    # -----------------------------
    # Load canonical documents
    # -----------------------------
    with open(CANONICAL_DOCS_PATH, "r") as f:
        documents = json.load(f)

    print(f"[INFO] Loaded {len(documents)} canonical documents")

    # -----------------------------
    # Load embedding model
    # -----------------------------
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()

    print(f"[INFO] Embedding dimension: {embedding_dim}")

    # -----------------------------
    # Connect to Qdrant
    # -----------------------------
    client = QdrantClient(url=QDRANT_URL)

    # -----------------------------
    # Recreate dense collection
    # -----------------------------
    if client.collection_exists(COLLECTION_NAME):
        print(f"[WARN] Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE,
        ),
    )

    print(f"[OK] Created dense collection: {COLLECTION_NAME}")

    # -----------------------------
    # Embed documents
    # -----------------------------
    texts = [doc["text"] for doc in documents]

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # -----------------------------
    # Build Qdrant points
    # -----------------------------
    points = []
    for doc, vector in tqdm(
        zip(documents, embeddings),
        total=len(documents),
        desc="Building dense points",
    ):
        points.append(
            models.PointStruct(
                id=string_to_uuid(doc["id"]),
                vector=vector.tolist(),
                payload={
                    # Canonical identity
                    "doc_id": doc["id"],
                    "video_id": doc["video_id"],
                    "title": doc["title"],
                    "timestamp_start": doc["timestamp_start"],
                    "timestamp_end": doc["timestamp_end"],
                    # Content
                    "text": doc["text"],
                    # Metadata
                    "source": doc["source"],
                    "speaker": doc["speaker"],
                },
            )
        )

    # -----------------------------
    # Upsert in batches
    # -----------------------------
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
            wait=False,
        )

    print(f"[OK] Indexed {len(points)} documents into dense collection")


if __name__ == "__main__":
    main()
