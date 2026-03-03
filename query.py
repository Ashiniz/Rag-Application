import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_community.llms import Ollama  # <--- local LLM


# ---- TF-IDF embedder (same idea as main.py) ----
class TfidfEmbeddings(Embeddings):
    def __init__(self, texts: List[str]):
        self.vectorizer = TfidfVectorizer().fit(texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.vectorizer.transform([text]).toarray()[0].tolist()


# ---- read your notes and make chunks ----
docs_dir = "docs"
file_path = os.path.join(docs_dir, "notes.txt")
with open(file_path, "r", encoding="utf-8") as f:
    full_text = f.read()

chunk_size = 400  # <-- keep same as in main.py
chunks: List[str] = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

embeddings = TfidfEmbeddings(chunks)

# ---- load chroma ----
vectordb = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

# ---- user question ----
query = input("🔎 Enter your question: ")

# ---- retrieve from chroma ----
docs = vectordb.similarity_search(query, k=3)
context = "\n\n".join(d.page_content for d in docs)

print("\n📄 Retrieved context:\n")
print(context)

# ---- call local ollama model ----
# make sure you ran: `ollama pull phi3` (or llama3)
llm = Ollama(model="phi3")

prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the user's question.
If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

answer = llm.invoke(prompt)

print("\n🤖 Local LLM answer:\n")
print(answer)