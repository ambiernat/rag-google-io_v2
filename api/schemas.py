from pydantic import BaseModel
from typing import List, Optional, Literal

# -----------------------------
# Request Models
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    mode: Literal["sparse", "dense", "hybrid"] = "hybrid"
    rerank: Optional[bool] = False  # trigger reranking
    experiment_id: Optional[str] = None  # NEW: track A/B experiment
# -----------------------------
# Response Models
# -----------------------------
class RetrievedDocument(BaseModel):
    doc_id: str
    score: Optional[float] = None
    text: str
    title: Optional[str] = None
    video_id: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[RetrievedDocument]
