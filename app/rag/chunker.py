import re

def chunk_text(text, max_words=120, overlap=20):

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        sentence_len = len(words)

        # if single sentence is too big, force split
        if sentence_len > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

            # split long sentence
            for i in range(0, sentence_len, max_words):
                chunks.append(" ".join(words[i:i+max_words]))

            continue

        # if adding exceeds limit → save chunk
        if current_len + sentence_len > max_words:
            chunks.append(" ".join(current))

            # overlap logic (important for RAG quality)
            current = current[-overlap:] if overlap > 0 else []
            current_len = len(current)

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks