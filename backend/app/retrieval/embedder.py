import math
import os
import threading
import time
from typing import Any
import requests
from backend.app.core.config import settings

_model = None
_model_lock = threading.Lock()
_http_session = requests.Session()

HF_BATCH_SIZE = 32
LOCAL_BATCH_SIZE = 8


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _parse_hf_response(data: Any, expected_count: int) -> list[list[float]]:
    if not isinstance(data, list) or not data:
        raise ValueError("Empty embedding response from Hugging Face API")

    # 1D vector returned for a single input
    if isinstance(data[0], (int, float)):
        return [_l2_normalize([float(x) for x in data])]

    # 2D sentence embeddings: [batch, dim]
    if isinstance(data[0], list) and data[0] and isinstance(data[0][0], (int, float)):
        vectors = [_l2_normalize([float(x) for x in row]) for row in data]
        if len(vectors) != expected_count:
            raise ValueError(f"Expected {expected_count} embeddings, got {len(vectors)}")
        return vectors

    # 3D token embeddings: [batch, tokens, dim] -> mean pool
    if isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
        pooled: list[list[float]] = []
        for token_matrix in data:
            num_tokens = len(token_matrix)
            dim = len(token_matrix[0])
            mean_vec = [
                sum(float(token_matrix[t][d]) for t in range(num_tokens)) / num_tokens
                for d in range(dim)
            ]
            pooled.append(_l2_normalize(mean_vec))
        return pooled

    raise ValueError("Unrecognized embedding response shape from Hugging Face API")


def _embed_via_hf_api(texts: list[str], hf_token: str) -> list[list[float]]:
    url = (
        f"https://router.huggingface.co/hf-inference/models/"
        f"{settings.EMBED_MODEL}/pipeline/feature-extraction"
    )
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), HF_BATCH_SIZE):
        batch = [t[:2000] if t else " " for t in texts[start : start + HF_BATCH_SIZE]]
        last_err = None

        for attempt in range(3):
            try:
                resp = _http_session.post(
                    url,
                    headers=headers,
                    json={"inputs": batch},
                    timeout=30,
                )
                if resp.status_code == 200:
                    all_vectors.extend(_parse_hf_response(resp.json(), len(batch)))
                    last_err = None
                    break
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"HF API {resp.status_code}: {resp.text[:200]}")
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"HF API {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))

        if last_err is not None:
            raise last_err

    return all_vectors


def get_embedder():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from fastembed import TextEmbedding

                _model = TextEmbedding(
                    model_name=settings.EMBED_MODEL,
                    threads=1,
                )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    mode = (os.getenv("EMBED_MODE") or settings.EMBED_MODE or "auto").lower()
    hf_token = (os.getenv("HF_TOKEN") or settings.HF_TOKEN or "").strip()

    if mode == "hf_api" or (mode == "auto" and hf_token):
        try:
            return _embed_via_hf_api(texts, hf_token)
        except Exception as exc:
            print(f"[EMBEDDER] HF API failed ({exc}); falling back to local FastEmbed")

    model = get_embedder()
    return [v.tolist() for v in model.embed(texts, batch_size=LOCAL_BATCH_SIZE)]