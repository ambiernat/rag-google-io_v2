from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/ready")
def ready():
    # Later: check Qdrant connectivity
    return {"ready": True}
