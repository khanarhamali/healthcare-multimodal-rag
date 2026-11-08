# DermaAI — Multimodal Healthvare RAG Backend

An AI-powered dermatology consultation backend built with **FastAPI**, **LangGraph**, **PyTorch (EfficientNet-B0)**, **Pinecone**, **FAISS for Hyperparamter Testing** and **Supabase (PostgreSQL + Storage)**. It processes patient skin images alongside clinical queries, retrieves grounded dermatological knowledge, and streams responses in real time.

---

## Key Features

- **Multimodal Vision + Text Pipeline:** Uses an `EfficientNet-B0` PyTorch classifier to detect potential skin conditions from uploaded images and enriches vector retrieval queries.
- **Dual-Stage RAG Architecture (Local FAISS Testing -> Cloud Pinecone Production):** Benchmark and tune chunking/retrieval hyperparameters locally using FAISS before syncing optimal settings to Pinecone.
- **Persistent Multi-Thread Memory:** Uses LangGraph's `PostgresSaver` backed by Supabase PostgreSQL so patients can maintain isolated threads and resume consultations anytime.
- **Real-Time SSE Streaming:** Streams vision predictions and OpenAI `gpt-4o-mini` tokens live via Server-Sent Events (`/api/chat/stream`).
- **Anti-Hallucination Guardrail:** Enforces a strict **20-message (10-turn) cap** per consultation thread to prevent context drift.
- **HIPAA-Ready Right-to-Erasure:** `DELETE /api/threads/{user_id}/{thread_id}` permanently purges thread metadata, PostgreSQL checkpoints (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`), and patient skin images from Supabase Storage.

---

## RAG Experimentation & Hyperparameter Tuning Workflow

Before indexing documents into the cloud production database, you can safely experiment with and evaluate different RAG hyperparameters locally without incurring cloud vector storage overhead:

1. **Local Evaluation & Tuning (`test_retriever.py` & `test_generator.py`):**
   - Uses a **local FAISS vector database** for rapid, offline experimentation.
   - Tune key hyperparameters such as `chunk_size`, `chunk_overlap`, top-`k` retrieval count, embedding models, and clinical prompt templates.
   - Evaluate retrieval accuracy and response faithfulness using `test_retriever.py` and `test_generator.py`.

2. **Promoting Tuned Parameters to Production (`retriever.py` & `generator.py`):**
   - Once you identify the best-performing hyperparameters in your local FAISS tests, copy those exact `chunk_size`, `chunk_overlap`, and `k` values into `retriever.py`.
   - In production, `retriever.py` uses **Pinecone** (cloud serverless vector database) paired with `generator.py` to serve live patient consultations.

---

## Prerequisites & Environment Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure `.env` in the root directory:**
   ```env
   # OpenAI & Pinecone
   OPENAI_API_KEY="your-openai-api-key"
   PINECONE_API_KEY="your-pinecone-api-key"

   # Supabase Storage & Database
   SUPABASE_URL="https://supabase.co"
   SUPABASE_KEY="your-supabase-service-role-or-secret-key"
   SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-or-secret-key"
   SUPABASE_BUCKET="skincare-images"
   SUPABASE_DB_URL="postgresql://postgres.your-project-ref:URL_ENCODED_PASSWORD@://supabase.com"

   # CORS (comma-separated origins or "*" for development)
   ALLOWED_ORIGINS="http://localhost:3000"
   ```

4. **Create the `chat_threads` table in Supabase SQL Editor:**
   ```sql
   CREATE TABLE IF NOT EXISTS chat_threads (
       id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
       user_id TEXT NOT NULL,
       thread_id TEXT UNIQUE NOT NULL,
       title TEXT DEFAULT 'New Skincare Chat',
       created_at TIMESTAMPTZ DEFAULT NOW(),
       updated_at TIMESTAMPTZ DEFAULT NOW()
   );

   CREATE INDEX IF NOT EXISTS idx_chat_threads_user_id ON chat_threads(user_id);
   ```
   *(LangGraph checkpoint tables are created automatically on startup via `memory.setup()`).*

---

## Running Locally

```bash
python app.py
```
- **API Base URL:** `http://localhost:8000/api`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | Standard synchronous multimodal chat (text + optional image) |
| `POST` | `/api/chat/stream` | Real-time SSE streaming chat with vision badge & token output |
| `GET` | `/api/threads/{user_id}` | List all consultation threads for a patient |
| `GET` | `/api/history/{user_id}/{thread_id}` | Resume a thread by loading saved LangGraph message history |
| `DELETE` | `/api/threads/{user_id}/{thread_id}` | Permanently erase thread metadata, checkpoints, and storage images |

---

## Docker Deployment

```bash
# Build image
docker build -t your-dockerhub-username/derma-rag-backend:latest .

# Push to Docker Hub
docker push your-dockerhub-username/derma-rag-backend:latest

# Run container on server
docker run -d -p 8000:8000 --env-file .env --name derma-backend your-dockerhub-username/derma-rag-backend:latest
```
