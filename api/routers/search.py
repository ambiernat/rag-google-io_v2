from fastapi import APIRouter, HTTPException
from typing import Optional
import logging
from pathlib import Path
import yaml
import json

from api.schemas import SearchRequest, SearchResponse, RetrievedDocument
from retrieval.retrievers.retrieve_sparse import SparseRetriever
from retrieval.retrievers.retrieve_dense import retrieve_dense
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid
from retrieval.rerankers.crossencoder_reranker import crossencoder_rerank
from sentence_transformers import CrossEncoder

# -----------------------------
# Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "retrieval/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# -----------------------------
# Logging
# -----------------------------
log_level = config.get("logging", {}).get("level", "INFO")
logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------
# Retriever Setup
# -----------------------------
sparse_cfg = config["sparse"]
dense_cfg = config["dense"]
hybrid_cfg = config["hybrid"]

sparse_retriever = SparseRetriever()

# -----------------------------
# Reranker Setup
# -----------------------------
rerank_cfg = config["rerankers"]["cross_encoder"]
rerank_enabled = rerank_cfg.get("enabled", False)
rerank_model_name = rerank_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
rerank_top_k = rerank_cfg.get("top_k", 10)

if rerank_enabled:
    logger.info(f"Loading CrossEncoder model: {rerank_model_name}")
    reranker_model = CrossEncoder(rerank_model_name)
else:
    reranker_model = None
    logger.info("CrossEncoder reranker is disabled in config.yaml")

# -----------------------------
# Experiment log path
# -----------------------------
EXPERIMENT_LOG = PROJECT_ROOT / "data" / "experiments.log"

def log_experiment(request: SearchRequest, results: list):
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        log_entry = {
            "experiment_id": request.experiment_id,
            "query": request.query,
            "mode": request.mode,
            "rerank": request.rerank,
            "top_k": request.top_k,
            "results": results
        }
        f.write(json.dumps(log_entry) + "\n")

# -----------------------------
# Router
# -----------------------------
router = APIRouter()

# -----------------------------
# Helpers
# -----------------------------
def get_top_k(cfg_top_k: int, request_top_k: Optional[int]) -> int:
    return request_top_k if request_top_k is not None else cfg_top_k

def rerank_docs_with_loaded_model(query: str, docs: list, top_k: int):
    texts = [doc["text"] for doc in docs]
    pairs = [[query, text] for text in texts]

    scores = reranker_model.predict(pairs)
    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [
        {**doc, "score": float(score)}
        for doc, score in ranked
    ]

# -----------------------------
# Search Endpoint
# -----------------------------
@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    top_k = None

    # -----------------------------
    # Retrieval
    # -----------------------------
    if request.mode == "sparse":
        top_k = get_top_k(sparse_cfg.get("top_k", 5), request.top_k)
        hits = sparse_retriever.retrieve(request.query, top_k=top_k)
        docs = [{"doc_id": doc["id"], "score": score, "text": doc["text"]} for score, doc in hits]

    elif request.mode == "dense":
        top_k = get_top_k(dense_cfg.get("top_k", 5), request.top_k)
        hits = retrieve_dense(request.query, top_k=top_k)
        docs = [
            {"doc_id": hit.payload.get("doc_id", str(hit.id)),
             "score": hit.score,
             "text": hit.payload.get("text", "")}
            for hit in hits
        ]

    elif request.mode == "hybrid":
        top_k = get_top_k(hybrid_cfg.get("top_k", 5), request.top_k)
        hits = retrieve_hybrid(request.query, top_k=top_k)
        docs = [
            {"doc_id": hit.payload.get("doc_id", str(hit.id)),
             "score": hit.score,
             "text": hit.payload.get("text", "")}
            for hit in hits
        ]

    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")

    # -----------------------------
    # Optional Reranking
    # -----------------------------
    if request.rerank and rerank_enabled and docs:
        docs = rerank_docs_with_loaded_model(
            query=request.query,
            docs=docs,
            top_k=rerank_top_k,
        )


    # -----------------------------
    # Experiment logging
    # -----------------------------
    if request.experiment_id:
        log_experiment(request, docs)
        logger.info(f"Experiment logged: {request.experiment_id}")

    # -----------------------------
    # Response
    # -----------------------------
    return SearchResponse(
        query=request.query,
        results=[RetrievedDocument(**doc) for doc in docs]
    )
# -----------------------------