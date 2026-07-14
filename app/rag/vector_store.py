import faiss
import numpy as np

dimension = 384
index = faiss.IndexFlatL2(dimension)

documents = []

def add_vector(embedding, text):
    index.add(np.array([embedding]).astype("float32"))
    documents.append(text)


def search(query_embedding, k=1):

    D, I = index.search(
        np.array([query_embedding]).astype("float32"),
        k
    )

    results = []
    scores = []

    for score, idx in zip(D[0], I[0]):

        if idx < len(documents):

            # 🔥 FILTER BAD MATCHES (IMPORTANT)
            if score < 1.2:   # tune this
                results.append(documents[idx])
                scores.append(score)

    return results, scores