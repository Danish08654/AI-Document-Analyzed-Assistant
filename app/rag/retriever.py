from app.rag.vector_store import search
from app.rag.embedder import get_embedding

def retrieve(question):
    q_emb = get_embedding(question)

    results, scores = search(q_emb, k=5)

    seen = set()
    final_results = []

    for chunk, score in zip(results, scores):

        # 1. similarity filter (IMPORTANT)
        if score > 1.2:   # adjust threshold
            continue

        # 2. remove duplicates properly
        if chunk not in seen:
            final_results.append(chunk)
            seen.add(chunk)

    return final_results