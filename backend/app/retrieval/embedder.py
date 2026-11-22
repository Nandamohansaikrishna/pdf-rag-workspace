import threading
from fastembed import TextEmbedding
from backend.app.core.config import settings

_model = None
_model_lock = threading.Lock()

def get_embedder() -> TextEmbedding:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = TextEmbedding(
                    model_name=settings.EMBED_MODEL,
                    threads=1,
                )
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedder()
    return [v.tolist() for v in model.embed(texts, batch_size=8)]