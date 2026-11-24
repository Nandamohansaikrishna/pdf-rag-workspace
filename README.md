# PDF RAG Workspace

A production-ready Retrieval-Augmented Generation (RAG) web application for chatting with PDF documents, built with **Streamlit**, **Qdrant**, **PyMuPDF**, and **Groq / Ollama**.

Designed with a **dual-mode architecture** that runs fully offline on local hardware (FastEmbed ONNX + Ollama) and seamlessly switches to an ultra-lightweight **Cloud Mode** (~220 MB RAM footprint) to run reliably within Streamlit Community Cloud's 1 GB free-tier container without Out-of-Memory crashes or CPU throttling.

**Live App:** [pdf-rag-workspace.streamlit.app](https://pdf-rag-workspace.streamlit.app)

---

## Key Features

- **High-Speed PDF Parsing:** Extracts page text and word-level bounding-box coordinates using **PyMuPDF (`pymupdf`)**, processing large PDFs (up to 50 MB / 300 pages) in under a second.
- **Dual-Mode Embeddings (`BAAI/bge-small-en-v1.5`):** Uses the **Hugging Face Inference API** in Cloud Mode (batched requests, zero local ML memory overhead) and automatically falls back to single-threaded local **FastEmbed (ONNX Runtime)** when running offline.
- **Two-Stage Hybrid Retrieval:**
  - **Stage 1 (Semantic Recall):** Retrieves top candidate passages from **Qdrant** using 384-dimensional cosine similarity with batched chunk upserts.
  - **Stage 2 (Precision Reranking):** Supports zero-RAM **Hybrid Dense (75%) + BM25 Lexical (25%)** reranking for fast cloud execution (<5 ms) or local ONNX Cross-Encoder reranking (`Xenova/ms-marco-MiniLM-L-6-v2`).
- **Smart LLM Routing:** Routes queries to **Groq Cloud** (`openai/gpt-oss-20b` with `reasoning_effort="low"`) for low-latency answers, with automatic fallback to local **Ollama** (`qwen2.5:latest`).
- **Source-Grounded Citations:** Returns answers backed by matched PDF passages, page numbers, and retrieval/rerank scores.
- **Google OAuth Authentication:** Built-in OpenID Connect sign-in via `streamlit[auth]` with configurable email allowlisting.

---

## Architecture

```text
PDF Upload (Streamlit UI)
   │
   ▼
PyMuPDF Parser (Text + Word Coordinates) ──► Chunking
   │
   ▼
Embedding Engine (384-dim, BAAI/bge-small-en-v1.5)
   ├── Cloud Mode: Hugging Face Inference API (batched, 0 MB local model RAM)
   └── Local Mode: FastEmbed ONNX Runtime (threads=1, batch_size=8)
   │
   ▼
Qdrant Vector Store (Cosine Similarity, Batched Upserts)
   │
   ▼
Stage 2 Reranking (Top 25 ──► Top 5)
   ├── Cloud Mode: Pure-Python Hybrid Dense (0.75) + BM25 (0.25)
   └── Local Mode: FastEmbed TextCrossEncoder (Xenova/ms-marco-MiniLM-L-6-v2)
   │
   ▼
LLM Generation + Page Citations
   ├── Primary: Groq Cloud (openai/gpt-oss-20b, reasoning_effort="low")
   └── Fallback: Local Ollama (qwen2.5:latest)
```

### Cloud Optimization Summary

Deploying local neural networks on a 1 GB RAM shared container causes OOM crashes and severe CPU throttling. This project resolves both without changing hosts:

| Component | Local Mode | Cloud Mode (`Streamlit Community Cloud`) |
| :--- | :--- | :--- |
| **PDF Parser** | PyMuPDF (`pymupdf`) | PyMuPDF (`pymupdf`) |
| **Embeddings** | FastEmbed ONNX (local, 1 thread) | Hugging Face Inference API (`HF_TOKEN`) |
| **Reranker** | ONNX Cross-Encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) | Zero-RAM Hybrid Dense + BM25 (`DISABLE_RERANKER=1`) |
| **LLM Provider** | Ollama (`qwen2.5:latest`) or Groq | Groq Cloud (`openai/gpt-oss-20b`, `reasoning_effort="low"`) |
| **Peak Container RAM** | ~650 MB | **~220 MB** (lazy imports keep ONNX unloaded) |

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend & Auth** | Streamlit (`streamlit[auth]`), Google OAuth (OIDC) |
| **Document Parsing** | PyMuPDF (`pymupdf`), `pypdf` |
| **Embeddings** | Hugging Face Inference API, FastEmbed (`BAAI/bge-small-en-v1.5`) |
| **Vector Database** | Qdrant (`qdrant-client`) |
| **Reranking** | Custom BM25 + Dense Fusion, FastEmbed `TextCrossEncoder` |
| **LLM Inference** | Groq SDK, Ollama |
| **Configuration** | Pydantic Settings (`pydantic-settings`), `python-dotenv` |

---

## Project Structure

```text
rag-project/
├── .streamlit/
│   └── config.toml                          # Streamlit UI theme configuration
├── backend/
│   └── app/
│       ├── core/
│       │   └── config.py                    # Pydantic settings & cloud/local flags
│       ├── ingestion/
│       │   └── parser.py                    # PyMuPDF text & coordinate extractor
│       ├── retrieval/
│       │   ├── embedder.py                  # Dual-mode HF API / FastEmbed embedder
│       │   └── vector_store.py              # Qdrant client, batched upserts & hybrid reranker
│       └── generation/
│           ├── answer.py                    # RAG orchestration & provider routing
│           ├── prompts.py                   # Grounded system prompt templates
│           └── providers/
│               ├── groq_provider.py         # Cached Groq client with low-latency reasoning
│               └── ollama.py                # Local Ollama inference provider
├── frontend/
│   ├── app.py                               # Streamlit entrypoint (UI, auth, PDF chat)
│   └── requirements.txt                     # Cloud-optimized Python dependencies
└── README.md
```

---

## Getting Started (Local Development)

### Prerequisites
- Python 3.10+
- Optional: [Ollama](https://ollama.com/) installed locally with `qwen2.5:latest` pulled (`ollama pull qwen2.5:latest`)
- Optional: Free API keys from [Groq Console](https://console.groq.com/) and [Hugging Face](https://huggingface.co/settings/tokens) (Fine-grained token with **Inference Providers** permission)

### 1. Clone and Install Dependencies

Using `uv`:
```bash
git clone [https://github.com/Nandamohansaikrishna/pdf-rag-workspace.git](https://github.com/Nandamohansaikrishna/pdf-rag-workspace.git)
cd pdf-rag-workspace
uv venv
uv pip install -r frontend/requirements.txt
```

Or using standard `venv` + `pip`:
```bash
git clone [https://github.com/Nandamohansaikrishna/pdf-rag-workspace.git](https://github.com/Nandamohansaikrishna/pdf-rag-workspace.git)
cd pdf-rag-workspace
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r frontend/requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root (do not commit this file):

```env
# LLM Configuration
LLM_PROVIDER=groq                     # "groq" or "ollama"
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
GROQ_REASONING_EFFORT=low

# Local Ollama Fallback
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:latest

# Embeddings & Retrieval
EMBED_MODE=auto                       # "auto" (uses HF API if HF_TOKEN set), "hf_api", or "local"
HF_TOKEN=hf_your_huggingface_token    # Optional locally; recommended for cloud speed
EMBED_MODEL=BAAI/bge-small-en-v1.5
RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2
DISABLE_RERANKER=0                    # 0 = ONNX Cross-Encoder, 1 = Hybrid Dense + BM25

# Storage
QDRANT_PATH=./qdrant_data
REFUSAL_THRESHOLD=0.0
```

### 3. Run the Application

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

---

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub (ensure `.env`, `.streamlit/secrets.toml`, and `qdrant_data/` are listed in `.gitignore`).
2. Create a new app on [share.streamlit.io](https://share.streamlit.io/) and set **Main file path** to `frontend/app.py`.
3. Add the following to **App Settings → Secrets** (keep app keys above `[auth]`):

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_REASONING_EFFORT = "low"
HF_TOKEN = "hf_your_huggingface_token"
EMBED_MODE = "auto"
DISABLE_RERANKER = "1"
LLM_PROVIDER = "groq"
DISABLE_OLLAMA = "1"
PDF_SESSION_ONLY = "1"
ALLOWED_EMAILS = "your-email@gmail.com"

[auth]
redirect_uri = "https://<your-app-name>.streamlit.app/oauth2callback"
cookie_secret = "your_random_cookie_secret"
client_id = "your_google_oauth_client_id.apps.googleusercontent.com"
client_secret = "your_google_oauth_client_secret"
server_metadata_url = "[https://accounts.google.com/.well-known/openid-configuration](https://accounts.google.com/.well-known/openid-configuration)"
```

4. Register `https://<your-app-name>.streamlit.app/oauth2callback` as an Authorized Redirect URI in your Google Cloud OAuth 2.0 Client settings.

---

## Author

**Nanda Mohan Sai Krishna**
- GitHub: [@Nandamohansaikrishna](https://github.com/Nandamohansaikrishna)
- LinkedIn: [Nanda Mohan Sai Krishna](https://www.linkedin.com/in/nanda-mohan-sai-krishna-357900271)