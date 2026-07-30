---
title: Self-Healing Research Assistant
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Self-Healing RAG

A research assistant that retrieves, generates, and **self-corrects**. When the system detects that its answer isn't well-grounded in the source material, it rewrites the query, re-retrieves, and tries again — automatically.

Built with FastAPI, Qdrant, BM25, and Groq (LLaMA 3.3 70B).

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Most RAG systems retrieve context, generate an answer, and return it — hoping it's correct. This one doesn't hope. It **checks**.

After generating a response, the system runs a grounding evaluation against the retrieved context. If the confidence score falls below a threshold (default: 80%), a healing loop kicks in:

1. The query gets rewritten with expanded terms
2. Retrieval parameters are adjusted (top-k, filters)
3. If needed, new papers are fetched from arXiv on the fly
4. The pipeline runs again with the improved inputs

This repeats until the answer passes validation or the retry limit is hit.

## How it works

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Enhancement   │  ← HyDE, topic extraction, intent detection
│  & Scope Selection   │  ← hybrid / custom-only / arxiv-only
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Hybrid Retrieval    │  ← Qdrant (dense, bge-large-en-v1.5)
│                      │  ← BM25 (sparse, lexical)
│                      │  ← Reciprocal Rank Fusion
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Cross-Encoder       │  ← ms-marco-MiniLM-L-6-v2
│  Reranking           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  LLM Generation      │  ← Groq LLaMA 3.3 70B (or local Ollama)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Grounding Check     │  ← Is the answer supported by context?
│  (CRAG Validation)   │
└─────────┬───────────┘
          │
     ┌────┴────┐
     │         │
  PASS      FAIL
     │         │
  Return    Rewrite query,
  answer    adjust params,
            fetch new data,
            retry (up to N times)
```

## Features

- **Hybrid search**: Dense vector retrieval (Qdrant + BGE-large) combined with BM25 lexical search, merged via Reciprocal Rank Fusion
- **Cross-encoder reranking**: Chunks are re-scored with a fine-tuned cross-encoder before hitting the LLM
- **Knowledge scope control**: Users can restrict search to custom uploads only, arXiv papers only, or both
- **Document ingestion**: Upload PDFs, DOCX, TXT, CSV, code files, or JSON — they get chunked, embedded, and indexed incrementally
- **Dynamic acquisition**: If existing knowledge is insufficient, the system fetches relevant papers from arXiv automatically
- **Multi-user auth**: Registration, login, per-user session history, and isolated telemetry
- **Telemetry dashboard**: Web UI with live metrics — query latency, grounding pass rates, retry counts, and evidence breakdowns

## Tech stack

| Component | Technology |
|-----------|-----------|
| API server | FastAPI + Uvicorn |
| LLM | Groq API (LLaMA 3.3 70B) / Ollama (local fallback) |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Vector store | Qdrant (local persistent) |
| Lexical search | BM25 (rank_bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Persistence | SQLite |
| Frontend | Vanilla HTML/CSS/JS |

## Setup

```bash
git clone https://github.com/ananthu-n/SELF-HEALING-RAG-.git
cd SELF-HEALING-RAG-

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

Run:

```bash
python main.py
```

Open `http://localhost:8000`.

## Docker

```bash
docker-compose up -d --build
```

## Project structure

```
app/
├── api/            # FastAPI routes and request/response models
├── core/           # Config loader, logger
├── db/             # SQLite persistence
├── evaluation/     # Grounding evaluator, CRAG validator, decision engine
├── ingestion/      # PDF/document upload, arXiv fetcher, topic extraction
├── llm/            # Unified LLM client (Groq / OpenAI / Ollama)
├── pipeline/       # RAG pipeline orchestrator, intent detection
├── prompt/         # Prompt builder, templates, validator
├── reranker/       # Cross-encoder reranking
├── retrieval/      # Dense retriever, BM25, RRF fusion, post-processors
├── self_healing/   # Controller, query rewriter, healing planner
└── vectorstore/    # Qdrant client, collection management, indexer

configs/            # config.yaml
static/             # Web UI (HTML, CSS, JS)
scripts/            # Embedding generation, BM25 index building
tests/              # Unit and integration tests
```

## API endpoints

| Method | Endpoint | What it does |
|--------|----------|-------------|
| POST | `/api/query` | Run a query through the self-healing RAG pipeline |
| POST | `/api/upload` | Upload and index a document |
| POST | `/api/auth/register` | Create a user account |
| POST | `/api/auth/login` | Log in and get a session |
| GET | `/api/sessions` | Get session history |
| GET | `/api/telemetry` | System health and query stats |
| GET | `/api/health` | Health check |

## Configuration

All tunable parameters live in `configs/config.yaml`:

- `self_healing.max_retries` — how many healing cycles to attempt (default: 2)
- `self_healing.confidence_threshold` — minimum grounding confidence to accept (default: 0.80)
- `embedding.model_name` — embedding model for dense retrieval
- `reranker.model_name` — cross-encoder model for reranking
- `llm.provider` — `groq`, `openai`, or `ollama`

## License

MIT
