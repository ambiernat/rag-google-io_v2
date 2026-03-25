from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Union
import yaml
import os

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "retrieval/config.yaml"

with open(CONFIG_PATH, "r") as f:
    _config = yaml.safe_load(f)

QDRANT_URL = os.getenv("QDRANT_URL", _config["qdrant"]["url"])

@router.get("/health")
def health() -> dict:
    return {"status": "ok"}

@router.get("/ready")
def ready() -> Union[dict, JSONResponse]:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL)
        client.get_collections()
        return {"ready": True}
    except Exception as e:
        return JSONResponse(status_code=503, content={"ready": False, "error": "Qdrant not reachable"})
