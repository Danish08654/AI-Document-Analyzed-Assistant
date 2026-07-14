import ollama
from app.rag.retriever import retrieve

def ask_question(question):

    context = retrieve(question)

    context_text = "\n".join(context)

    prompt = f"""
You are an AI assistant.

Use ONLY this context:

{context_text}

Question:
{question}

Give a short and accurate answer.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"], context