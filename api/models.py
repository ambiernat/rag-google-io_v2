from sentence_transformers import SentenceTransformer

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model

def embed_query(query: str) -> list:
    model = get_embedding_model()
    return model.encode(query, normalize_embeddings=True).tolist()
