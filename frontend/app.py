import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
st.title("AI Business System")
# -----------------------------
# SESSION STATE FIX
# -----------------------------
if "documents" not in st.session_state:
    st.session_state.documents = []
if "index" not in st.session_state:
    dimension = 384
    st.session_state.index = faiss.IndexFlatL2(dimension)
documents = st.session_state.documents
index = st.session_state.index
model = SentenceTransformer("all-MiniLM-L6-v2")
# -------------------------
# FILE UPLOAD
# -------------------------
file = st.file_uploader("Upload PDF", type=["pdf"])
if file:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    st.success("File loaded successfully")
    chunk_size = 300
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    for chunk in chunks:
        emb = model.encode(chunk)
        emb = np.array([emb]).astype("float32")
        index.add(emb)
        documents.append(chunk)
    st.session_state.documents = documents
    st.session_state.index = index
    st.info(f"Stored {len(chunks)} chunks in vector DB")
# -------------------------
# QUESTION ANSWER
# -------------------------
question = st.text_input("Ask Question")
if st.button("Submit"):
    if not question:
        st.warning("Please enter a question")
        st.stop()
    if len(documents) == 0:
        st.warning("Please upload a file first")
        st.stop()
    q_emb = model.encode(question)
    q_emb = np.array([q_emb]).astype("float32")
    D, I = index.search(q_emb, k=3)
    results = []
    for idx in I[0]:
        if idx < len(documents):
            results.append(documents[idx])
    st.subheader("Answer Context")
    for r in results:
        st.write("📄", r)
    st.subheader("Final Answer")
    if results:
        st.success(results[0])
    else:
        st.warning("No relevant info found")
