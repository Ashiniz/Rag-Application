import os
import uuid
from typing import List

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# LangChain
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

# OpenAI
from openai import OpenAI

# Langfuse
from langfuse import Langfuse

# ---------- Langfuse client (keys inline, local only) ----------
langfuse = Langfuse(
    secret_key="paste your secret key here",
    public_key="paste your public key here",
    host="http://localhost:3001",
)

openai_client = OpenAI(
    api_key="paste your OpenAI API key here"
)


# ---------- TF-IDF embedder ----------
class TfidfEmbeddings(Embeddings):
    def __init__(self, texts: List[str]):
        # Fit TF-IDF on the corpus chunks
        self.vectorizer = TfidfVectorizer().fit(texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.vectorizer.transform([text]).toarray()[0].tolist()


def shorten(text: str, max_chars: int = 500) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    return cut[:last_space] + "..."


# ---------- load notes ----------
docs_dir = "docs"
file_path = os.path.join(docs_dir, "notes.txt")

if not os.path.exists(file_path):
    st.error(f"Could not find docs/notes.txt at: {file_path}")
    st.stop()

with open(file_path, "r", encoding="utf-8") as f:
    full_text = f.read()

# Should match the chunking used when you built the index
chunk_size = 400
chunks: List[str] = [
    full_text[i: i + chunk_size] for i in range(0, len(full_text), chunk_size)
]

# Fit TF-IDF on the chunks
embeddings = TfidfEmbeddings(chunks)

# ---------- vector DB ----------
vectordb = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)

# ---------- UI ----------
st.title("RAG Demo – Local (Chroma + OpenAI + Langfuse)")
st.write("Ask about the content in your `docs/notes.txt`.")

# --- Model selection dropdown ---
MODEL_OPTIONS = {
    "GPT-4.1 mini (cheap & good)": "gpt-4.1-mini",
    "GPT-5 mini (newer, stronger)": "gpt-5-mini",
    "GPT-5.1 (max quality)": "gpt-5.1",
}

model_label = st.selectbox(
    "Choose OpenAI model",
    list(MODEL_OPTIONS.keys()),
    index=0,  # default = GPT-4.1 mini
)
selected_model = MODEL_OPTIONS[model_label]

st.caption(f"Using model: `{selected_model}`")

query = st.text_input("Your question")

if query:
    # ---------- Session + user for Langfuse ----------
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = f"streamlit-session-{uuid.uuid4().hex[:8]}"

    SESSION_ID = st.session_state["session_id"]
    USER_ID = "demo-user-1"  # you can later make this dynamic

    # --- Langfuse trace for this request ---
    trace = langfuse.trace(
        name="rag_app_query",
        input={"question": query, "model": selected_model},
        tags=["demo", "streamlit", "rag", selected_model],
        session_id=SESSION_ID,   # shows in Sessions tab
        user_id=USER_ID,         # shows in Users tab
    )

    # Optional: log the raw user question as an event
    trace.event(
        name="user_question",
        input={"question": query, "model": selected_model},
    )

    # --- Retrieve from Chroma ---
    docs = vectordb.similarity_search(query, k=2)

    if not docs:
        st.subheader("Retrieved")
        st.write("No matching text found in your notes.")
        trace.update(
            output={"answer": "No context found"},
            metadata={"status": "no_context"},
        )
        langfuse.flush()
        st.stop()

    retrieved_text = "\n\n".join(d.page_content for d in docs)

    st.subheader("Retrieved")
    st.write(shorten(retrieved_text, 500))

    # Log retrieval as a span
    trace.span(
        name="retrieval",
        input={"question": query, "k": 2, "model": selected_model},
        output={"retrieved_preview": shorten(retrieved_text, 1000)},
        metadata={"vector_db": "chroma", "embedder": "tfidf"},
    )

    # --- Generate with OpenAI ---
    st.subheader("Answer")
    try:
        system_msg = (
            "You are a helpful assistant. "
            "Use ONLY the provided context to answer. "
            "Write 4–6 clear, beginner-friendly sentences. "
            "Do NOT copy the context; explain it simply."
        )

        user_msg = f"""Context:
{retrieved_text}

Question: {query}
"""

        # Build arguments for chat.completions.create
        kwargs = dict(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )

        # ❗ gpt-5-mini does NOT support temperature=0.2 → let it use default
        if selected_model != "gpt-5-mini":
            kwargs["temperature"] = 0.2

        completion = openai_client.chat.completions.create(**kwargs)

        answer = completion.choices[0].message.content.strip()
        st.write(answer[:1200])

        # Extract token usage
        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None
        total_tokens = usage.total_tokens if usage else None

        # Log as GENERATION so it appears in Generations tab
        trace.generation(
            name="rag_llm_answer",
            model=selected_model,
            input=user_msg[:1200],      # prompt / context preview
            output=answer[:2000],       # answer preview
            metadata={
                "provider": "openai",
                "streamlit_model_label": model_label,
                # record what we *tried* to use for temperature:
                "temperature": 0.2 if selected_model != "gpt-5-mini" else "default",
            },
            usage=(
                {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": total_tokens,
                }
                if total_tokens is not None
                else None
            ),
        )

        trace.update(
            output={"final_answer": answer[:2000]},
            metadata={"status": "ok"},
        )
        trace.score(
            name="length_ok",
            value=1.0 if len(answer) > 50 else 0.0,
        )

    except Exception as e:
        # Show real error so we can debug if anything else breaks
        fallback = (
            "I couldn't call the OpenAI model. "
            "Here's the retrieved note instead:"
        )
        st.error(f"LLM error: {e}")
        st.write(fallback)
        st.write(shorten(retrieved_text))
        trace.update(
            output={"answer": fallback, "error": str(e)},
            metadata={"status": "llm_failed", "model": selected_model},
        )

    # Make sure everything is sent to Langfuse
    langfuse.flush()

else:
    st.info("Try: **what is RAG**, **benefits of RAG**, or **limitations of RAG**.")
