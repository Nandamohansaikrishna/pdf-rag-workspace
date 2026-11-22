from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

sentences = ['Paris is the capital of France.', 'Python is a programming language.']
model = TextEmbedding('BAAI/bge-small-en-v1.5')
vectors = [v.tolist() for v in model.embed(sentences)]

client = QdrantClient(path='./qdrant_data')

if client.collection_exists('hello'):
    client.delete_collection('hello')

client.create_collection('hello', vectors_config=VectorParams(size=384, distance=Distance.COSINE))
client.upsert('hello', points=[PointStruct(id=i, vector=v, payload={'text': s}) for i, (v, s) in enumerate(zip(vectors, sentences))])

query = next(iter(model.embed(['What is the capital of France?']))).tolist()
hit = client.query_points('hello', query=query, limit=1).points[0]

print(hit.payload['text'], hit.score)