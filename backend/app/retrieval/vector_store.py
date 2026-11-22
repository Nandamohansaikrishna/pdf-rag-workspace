import threading
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import CrossEncoder
from backend.app.core.config import settings
from backend.app.retrieval.embedder import embed_texts

# Thread-safe singletons so Streamlit reruns never collide on disk locks or slow down cold starts
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


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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

    # STAGE 1: Widen the net (Top 50) from Qdrant
    fetch_k = 50
    hits = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=fetch_k,
    ).points

    if not hits:
        return []

    retrieved_chunks = [{"score": hit.score, "payload": hit.payload} for hit in hits]

    # STAGE 2: Cross-Encoder Reranking
    pairs = [[query, chunk["payload"]["text"]] for chunk in retrieved_chunks]
    cross_scores = get_reranker().predict(pairs)

    for i in range(len(retrieved_chunks)):
        retrieved_chunks[i]["cross_score"] = float(cross_scores[i])

    reranked_chunks = sorted(
        retrieved_chunks, key=lambda x: x["cross_score"], reverse=True
    )
    return reranked_chunks[:top_k]