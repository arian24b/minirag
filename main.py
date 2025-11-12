import json
import os

import faiss
import numpy as np
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_PATH = "./docs"
INDEX_FILE = "./index.faiss"
META_FILE = "./meta.json"
EMBED_MODEL = "embeddinggemma:latest"
LLM_MODEL = "deepseek-r1:latest"


def load_docs(path):
    docs = []
    for file in os.listdir(path):
        if file.endswith(".md") or file.endswith(".mdx"):
            with open(os.path.join(path, file), encoding="utf-8") as f:
                docs.append(f.read())
    return docs


def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return [chunk for doc in docs for chunk in splitter.split_text(doc)]


def build_index(chunks):
    print("Generating embeddings and building FAISS index...")
    vectors = []
    for text in chunks:
        emb = ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]
        vectors.append(emb)

    vectors = np.array(vectors).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "dim": dim}, f, ensure_ascii=False, indent=2)
    print(f"Indexed {len(chunks)} text chunks (dim={dim}).")


def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["chunks"], meta["dim"]


def retrieve(query, index, chunks, top_k=3, expected_dim=None):
    q_emb = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    q_emb = np.array([q_emb]).astype("float32")

    if expected_dim and q_emb.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: index={expected_dim}, query={q_emb.shape[1]}. "
            f"Make sure EMBED_MODEL='{EMBED_MODEL}' is the same for both build and query.",
        )

    distances, ids = index.search(q_emb, top_k)
    return [chunks[i] for i in ids[0]]


def chat_with_context(query, context):
    prompt = f"""You are a helpful assistant.
Use the context below to answer the question accurately.

Context:
{context}

Question: {query}
Answer:"""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    if not os.path.exists(INDEX_FILE):
        docs = load_docs(DOCS_PATH)
        chunks = chunk_docs(docs)
        build_index(chunks)

    index, chunks, dim = load_index()

    print("\nMini RAG Chat is ready. Type 'exit' to quit.\n")
    while True:
        query = input("Ask: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        context = "\n\n".join(retrieve(query, index, chunks, expected_dim=dim))
        answer = chat_with_context(query, context)
        print(f"\n{answer}\n")
