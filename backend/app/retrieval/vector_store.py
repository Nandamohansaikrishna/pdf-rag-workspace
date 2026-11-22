import os
import threading
import uuid
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from backend.app.core.config import settings
from backend.app.retrieval.embedder import embed_texts

_client = None
_client_lock = threading.Lock()

_reranker = None
_reranker_lock = threading.Lock()


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = QdrantClient(path=settings.QDRANT_PATH)
    return _client


def get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = TextCrossEncoder(
                    model_name=settings.RERANK_MODEL,
                    threads=1,
                )
    return _reranker


def collection_exists(collection_name: str) -> bool:
    return get_client().collection_exists(collection_name)


def get_collection_count(collection_name: str) -> int:
    client = get_client()
    if not client.collection_exists(collection_name):
        return 0
    return client.count(collection_name=collection_name).count


def upsert_chunks(collection_name: str, chunks: list[dict]):
    if not chunks:
        return
    client = get_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    vectors = embed_texts([c["text"] for c in chunks])
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vec, payload=chunk)
        for chunk, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)


def search(collection_name: str, query: str, top_k: int = 5):
    client = get_client()
    if not client.collection_exists(collection_name):
        return []

    query_vector = embed_texts([query])[0]
    use_reranker = os.getenv("DISABLE_RERANKER", "0") != "1"

    fetch_k = 20 if use_reranker else top_k
    hits = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=fetch_k,
    ).points

    if not hits:
        return []

    retrieved_chunks = [{"score": hit.score, "payload": hit.payload} for hit in hits]

    if not use_reranker:
        return retrieved_chunks[:top_k]

    documents = [chunk["payload"]["text"] for chunk in retrieved_chunks]
    cross_scores = list(get_reranker().rerank(query, documents, batch_size=8))

    for i, cross_score in enumerate(cross_scores):
        retrieved_chunks[i]["cross_score"] = float(cross_score)

    reranked_chunks = sorted(
        retrieved_chunks, key=lambda x: x["cross_score"], reverse=True
    )
    return reranked_chunks[:top_k]