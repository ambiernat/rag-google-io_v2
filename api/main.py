from fastapi import FastAPI
from api.routers import search, health
from api.models import get_embedding_model

app = FastAPI(
    title="RAG Google I/O API",
    version="1.0",
    description="Dense, Sparse and Hybrid Retrieval API"
)

app.include_router(search.router, prefix="/api")
app.include_router(health.router)

@app.on_event("startup")
def preload_models():
    get_embedding_model()
