# main.py
# RAG setup WITHOUT torch / onnx / fastembed
# uses scikit-learn TF-IDF as the embedding backend

import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from sklearn.feature_extraction.text import TfidfVectorizer

# ---- 1. read your file ----
docs_dir = "docs"
file_path = os.path.join(docs_dir, "notes.txt")

with open(file_path, "r", encoding="utf-8") as f:
    full_text = f.read()

# ---- 2. make small chunks ----
chunk_size = 400
chunks: List[str] = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

# ---- 3. define a TF-IDF embedding class ----
class TfidfEmbeddings(Embeddings):
    def __init__(self, texts: List[str]):
        # fit TF-IDF on the corpus
        self.vectorizer = TfidfVectorizer().fit(texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        mat = self.vectorizer.transform(texts)
        return mat.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        mat = self.vectorizer.transform([text])
        return mat.toarray()[0].tolist()

# create embedding object trained on our chunks
embeddings = TfidfEmbeddings(chunks)

# ---- 4. create / persist Chroma ----
vectordb = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="rag_collection",
    persist_directory="chroma_db",
)

print("✅ Documents loaded into Chroma using TF-IDF (no torch, no onnx).")
