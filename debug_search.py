from backend.app.retrieval.vector_store import search

question = "How many steps was the base model trained on the Colossal Clean Crawled Corpus?"
results = search("documents", question, top_k=5)

for i, res in enumerate(results):
    page_num = res['payload'].get('page', 'Unknown')
    text = res['payload'].get('text', '')
    cross_score = res.get('cross_score', 'N/A')
    
    print(f"\n=== RANK {i+1} | PAGE {page_num} | SCORE {cross_score} ===")
    print(text)