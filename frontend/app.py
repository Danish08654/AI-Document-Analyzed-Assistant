import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss
import numpy as np


# auth 
os.environ["HF_TOKEN"] = st.secrets.get("HF_TOKEN", "")

st.set_page_config(page_title="DocMind", page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, body { font-family: 'Inter', sans-serif; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .block-container
    { background: #0f172a !important; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div
    { background: #1e293b !important; border-right: 1px solid #334155 !important; }
#MainMenu, footer, header { visibility: hidden; }
p, label, span, div, li { color: #cbd5e1; }
h1, h2, h3 { color: #f1f5f9 !important; }

.stTextInput input { background: #1e293b !important; color: #f1f5f9 !important; border: 1px solid #334155 !important; border-radius: 8px !important; }
.stTextArea textarea { background: #1e293b !important; color: #f1f5f9 !important; border: 1px solid #334155 !important; border-radius: 8px !important; font-size: 0.85rem !important; }
.stButton > button { background: #3b82f6 !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; padding: 10px !important; }
.stButton > button:hover { background: #2563eb !important; }
[data-testid="stFileUploadDropzone"] { background: #1e293b !important; border: 1px dashed #334155 !important; border-radius: 10px !important; }
.stSelectbox > div > div { background: #1e293b !important; border-color: #334155 !important; color: #f1f5f9 !important; }
hr { border-color: #334155 !important; }

.card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; }
.answer-box { background: #0f2744; border: 1px solid #1d4ed8; border-left: 3px solid #3b82f6; border-radius: 10px; padding: 18px 22px; font-size: 0.92rem; line-height: 1.7; color: #e2e8f0; }
.source-box { background: #1e293b; border: 1px solid #334155; border-left: 3px solid #475569; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; font-size: 0.8rem; color: #94a3b8; line-height: 1.6; }
.stat { text-align: center; }
.stat .val { font-size: 1.4rem; font-weight: 700; color: #3b82f6; }
.stat .lbl { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
</style>
""", unsafe_allow_html=True)

# session state 
if "chunks"  not in st.session_state: st.session_state.chunks  = []
if "index"   not in st.session_state: st.session_state.index   = faiss.IndexFlatL2(384)
if "history" not in st.session_state: st.session_state.history = []
if "files"   not in st.session_state: st.session_state.files   = []

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_groq():
    try: return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except: st.error("Add GROQ_API_KEY to Streamlit Secrets."); st.stop()

def embed(text):
    return np.array([load_model().encode(text)], dtype="float32")

def chunk_text(text, size=300, overlap=50):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def ingest(file):
    reader = PdfReader(file)
    text = "".join(p.extract_text() or "" for p in reader.pages)
    chunks = chunk_text(text)
    for c in chunks:
        st.session_state.index.add(embed(c))
        st.session_state.chunks.append(c)
    return len(chunks)

def search(question, k=4):
    if not st.session_state.chunks: return []
    D, I = st.session_state.index.search(embed(question), k=k)
    return [st.session_state.chunks[i] for i in I[0] if i < len(st.session_state.chunks)]

def ask_llm(question, context):
    groq = get_groq()
    prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the answer isn't in the context, say "I couldn't find that in the uploaded documents."

Context:
{chr(10).join(f'[{i+1}] {c}' for i, c in enumerate(context))}

Question: {question}"""
    return groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=1024
    ).choices[0].message.content.strip()


#  sidebar 
with st.sidebar:
    st.markdown("## 📄 DocMind")
    st.caption("Upload PDFs and ask anything.")
    st.divider()

    files = st.file_uploader("Upload PDFs", type=["pdf"],
                             accept_multiple_files=True, label_visibility="collapsed")
    if files:
        for f in files:
            if f.name not in st.session_state.files:
                with st.spinner(f"Indexing {f.name}…"):
                    n = ingest(f)
                    st.session_state.files.append(f.name)
                st.success(f"{f.name} — {n} chunks indexed")

    if st.session_state.files:
        st.divider()
        st.markdown("**Indexed files**")
        for name in st.session_state.files:
            st.markdown(f"📄 {name}")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat"><div class="val">{len(st.session_state.files)}</div><div class="lbl">Files</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat"><div class="val">{len(st.session_state.chunks)}</div><div class="lbl">Chunks</div></div>', unsafe_allow_html=True)

        st.divider()
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.chunks  = []
            st.session_state.index   = faiss.IndexFlatL2(384)
            st.session_state.history = []
            st.session_state.files   = []
            st.rerun()

# ── main ──
st.markdown("## Ask your documents")
st.divider()

if not st.session_state.files:
    st.markdown("""
    <div class="card" style="text-align:center;padding:40px;">
        <div style="font-size:2.5rem;margin-bottom:12px;">📄</div>
        <div style="font-size:1rem;font-weight:600;color:#f1f5f9;margin-bottom:6px;">No documents yet</div>
        <div style="color:#64748b;font-size:0.875rem;">Upload one or more PDFs from the sidebar to get started.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # chat history
    for item in st.session_state.history:
        st.markdown(f'<div class="card"><b style="color:#f1f5f9;">You:</b> {item["q"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box"><b>Answer:</b><br><br>{item["a"]}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # input
    col1, col2 = st.columns([6, 1], gap="small")
    with col1:
        question = st.text_input("", placeholder="Ask a question about your documents…", label_visibility="collapsed")
    with col2:
        submit = st.button("Ask →", use_container_width=True)

    if submit and question:
        with st.spinner("Thinking…"):
            sources  = search(question)
            answer   = ask_llm(question, sources)
        st.session_state.history.append({"q": question, "a": answer, "s": sources})

        st.markdown(f'<div class="card"><b style="color:#f1f5f9;">You:</b> {question}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box"><b>Answer:</b><br><br>{answer}</div>', unsafe_allow_html=True)

        with st.expander("View source chunks"):
            for i, s in enumerate(sources):
                st.markdown(f'<div class="source-box"><b style="color:#64748b;">Source {i+1}</b><br>{s}</div>', unsafe_allow_html=True)
    elif submit and not question:
        st.warning("Please enter a question.")
