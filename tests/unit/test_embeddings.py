from api.models import get_embedding_model

def test_embedding_model_loads():
    model = get_embedding_model()
    assert model is not None

def test_embedding_dimension():
    model = get_embedding_model()
    vec = model.encode("test sentence")

    assert isinstance(vec, (list, tuple)) or hasattr(vec, "shape")
    assert len(vec) > 100  # MiniLM = 384
