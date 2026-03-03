# 🔎 RAG Application (Local + Observable)

This project is a lightweight Retrieval-Augmented Generation (RAG) system built using Streamlit, LangChain, ChromaDB, TF-IDF embeddings, and Langfuse for observability. It demonstrates how to build a grounded LLM system without heavy embedding models, GPU requirements, or complex infrastructure.

The application loads local documents from the `docs` folder, converts them into TF-IDF embeddings using scikit-learn, stores them in a Chroma vector database, retrieves relevant context based on similarity search, and generates answers using an LLM (OpenAI or Ollama). All interactions are traceable using Langfuse for monitoring token usage, cost, and execution flow.

Project Structure:

- app.py → Streamlit user interface  
- main.py → RAG pipeline setup using TF-IDF embeddings  
- query.py → Retrieval and answer generation logic  
- docs/ → Local knowledge base files  
- requirements.txt → Project dependencies  
- .env.example → Environment variable template  

To run this project:

First install dependencies:

pip install -r requirements.txt

Then create a `.env` file in the project root using the format below:

OPENAI_API_KEY=your_key_here  
LANGFUSE_PUBLIC_KEY=your_key_here  
LANGFUSE_SECRET_KEY=your_key_here  
LANGFUSE_HOST=http://localhost:3001  

These are also included in app.py use that instead 

Make sure Docker Desktop is running. Start Langfuse using:

cd langfuse  
docker compose up -d  

Open Langfuse dashboard at:
http://localhost:3001

Start the RAG application using:

streamlit run app.py  

Open the application at:
http://localhost:8501

To stop the services:

Press Ctrl + C in the Streamlit terminal to stop the app.

To stop Langfuse:

cd langfuse  
docker compose down  

This project avoids heavy embedding models and instead uses TF-IDF to keep the system lightweight, fast, and easy to run locally. It is ideal for small internal knowledge assistants, document QA systems, and rapid RAG prototyping.

Security Note: The `.env` file and `chroma_db` directory are not committed to the repository. No API keys are stored in the source code.

Author:  
Ashiniz
AI Systems | RAG | LLM Applications
