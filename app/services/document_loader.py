from pypdf import PdfReader
from app.rag.vector_store import add_vector
from app.rag.embedder import get_embedding

def load_pdf(file):

    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = [text[i:i+300] for i in range(0, len(text), 300)]

    for chunk in chunks:
        emb = get_embedding(chunk)
        add_vector(emb, chunk)

    return len(chunks)