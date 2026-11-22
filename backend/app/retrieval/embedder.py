from fastembed import TextEmbedding
from backend.app.core.config import settings

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = TextEmbedding(settings.EMBED_MODEL)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return [v.tolist() for v in model.embed(texts)]