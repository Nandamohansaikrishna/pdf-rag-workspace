import math
import os
import re
import threading
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from backend.app.core.config import settings
from backend.app.retrieval.embedder import embed_texts

_client = None
_client_lock = threading.Lock()

_reranker = None
_reranker_lock = threading.Lock()

UPSERT_BATCH_SIZE = 64

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "pdf", "please", "tell", "that", "the", "this", "to", "what",
    "when", "where", "which", "who", "why", "with", "about", "document",
}


def _tokenize(text: str) -> list[str]:
    return [
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    ]


def _bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    query_terms = _tokenize(query)
    if not query_terms or not documents:
        return [0.0] * len(documents)

    tokenized_docs = [_tokenize(doc) for doc in documents]
    num_docs = len(tokenized_docs)
    avgdl = sum(len(d) for d in tokenized_docs) / max(num_docs, 1)
    if avgdl == 0:
        return [0.0] * num_docs

    doc_freqs: dict[str, int] = {}
    for term in set(query_terms):
        doc_freqs[term] = sum(1 for doc in tokenized_docs if term in doc)

    scores: list[float] = []
    for doc_tokens in tokenized_docs:
        dl = len(doc_tokens)
        if dl == 0:
            scores.append(0.0)
            continue

        tf_map: dict[str, int] = {}
        for tok in doc_tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1

        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            df = doc_freqs.get(term, 0)
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * ((tf * (k1 + 1.0)) / denom)
        scores.append(score)

    max_score = max(scores) if scores else 0.0
    if max_score <= 0.0:
        return [0.0] * num_docs
    return [s / max_score for s in scores]


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = QdrantClient(path=settings.QDRANT_PATH)
    return _client


def get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

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

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        vectors = embed_texts([c["text"] for c in batch])
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload=chunk)
            for chunk, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=collection_name, points=points)


def search(collection_name: str, query: str, top_k: int = 5):
    client = get_client()
    if not client.collection_exists(collection_name):
        return []

    query_vector = embed_texts([query])[0]
    disable_reranker = (
        os.getenv("DISABLE_RERANKER", "").lower() in ("1", "true", "yes")
        or settings.DISABLE_RERANKER
    )

    fetch_k = 25
    hits = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=fetch_k,
    ).points

    if not hits:
        return []

    retrieved_chunks = [{"score": float(hit.score), "payload": hit.payload} for hit in hits]
    documents = [chunk["payload"]["text"] for chunk in retrieved_chunks]

    if disable_reranker:
        # Zero-RAM, <5 ms Hybrid Dense + BM25 Reranking
        lexical_scores = _bm25_scores(query, documents)
        for chunk, lex_score in zip(retrieved_chunks, lexical_scores):
            chunk["cross_score"] = round(0.75 * chunk["score"] + 0.25 * lex_score, 4)
    else:
        cross_scores = list(get_reranker().rerank(query, documents, batch_size=8))
        for i, cross_score in enumerate(cross_scores):
            retrieved_chunks[i]["cross_score"] = float(cross_score)

    reranked_chunks = sorted(
        retrieved_chunks, key=lambda x: x["cross_score"], reverse=True
    )
    return reranked_chunks[:top_k]