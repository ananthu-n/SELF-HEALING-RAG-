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

A research assistant that retrieves, generates, and **self-corrects**. When the system detects that its answer isn't well-grounded in the source material, it diagnoses the failure mode, rewrites the query, adjusts retrieval parameters, acquires new knowledge if needed, and tries again — automatically.

Built with FastAPI, Qdrant, BM25, CrossEncoder reranking, and Groq (LLaMA 3.3 70B).

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Most RAG systems retrieve context, generate an answer, and return it. This one goes further — it **verifies** the answer against the evidence, and if the grounding is weak, it figures out *why* and fixes it.

The system classifies failures into specific types (low confidence, partial grounding, irrelevant retrieval, insufficient context, hallucination, ambiguous query) and applies targeted recovery strategies for each. It doesn't just retry blindly — it adapts.

## How it works

### End-to-end pipeline

```
User Query
    │
    ▼
┌─────────────────────────┐
│  1. Intent Detection     │  Classifies into: definition, comparison,
│     & Query Enhancement  │  explanation, literature_review, fact_lookup,
│                          │  or research_question
│     - Topic extraction   │  Strips conversational filler
│     - HyDE gating        │  Generates hypothetical doc (only for
│                          │  explanations & research questions)
│     - Decomposition      │  Splits comparisons into sub-queries
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  2. Knowledge Scope      │  User selects: hybrid / custom-only / arxiv-only
│     Selection            │  Filters chunks by source after retrieval
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  3. Hybrid Retrieval     │
│     - Dense search       │  Qdrant + BAAI/bge-large-en-v1.5 (1024-dim)
│     - Sparse search      │  BM25 lexical index (rank_bm25)
│     - RRF fusion         │  Reciprocal Rank Fusion merges both result sets
│     - Post-processing    │  Noise filter, duplicate filter, score filter
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  4. Cross-Encoder        │  ms-marco-MiniLM-L-6-v2
│     Reranking            │  Re-scores and re-orders fused chunks
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  5. Context Building     │  Selects top-k reranked chunks,
│     & Prompt Assembly    │  builds citation-tracked context window,
│                          │  validates token budget
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  6. LLM Generation       │  Groq LLaMA 3.3 70B (cloud)
│                          │  or Ollama qwen2.5 (local fallback)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  7. CRAG Validation      │  Corrective RAG — grades retrieval quality
│     (Document Level)     │  into CORRECT / AMBIGUOUS / INCORRECT
│                          │  based on max cross-encoder score
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  8. Grounding Evaluation │  LLM-as-judge evaluates answer vs context
│     (Answer Level)       │  Returns: is_grounded, confidence,
│                          │  unsupported_claims, should_retry
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  9. Failure Analysis     │  Classifies root cause:
│                          │  - NONE (fully grounded)
│                          │  - PARTIAL_GROUNDING
│                          │  - LOW_CONFIDENCE
│                          │  - INSUFFICIENT_CONTEXT
│                          │  - IRRELEVANT_RETRIEVAL
│                          │  - AMBIGUOUS_QUERY
│                          │  - HALLUCINATION
└────────────┬────────────┘
             │
        ┌────┴────┐
        │         │
     GROUNDED   FAILED
        │         │
     Return    ┌──▼──────────────────┐
     answer    │  10. Healing Loop    │
               │                     │
               │  - Adaptive profile │  Per-failure retrieval params
               │    selection        │  (top-k, dense/bm25 ratios)
               │                     │
               │  - Query rewriting  │  LLM-powered, intent-preserving
               │                     │  scientific search refinement
               │                     │
               │  - Dynamic arXiv    │  CrossEncoder-reranked paper
               │    acquisition      │  search → download → parse →
               │                     │  chunk → embed → index (live)
               │                     │
               │  - Multi-signal     │  Unsupported claims targeting,
               │    analysis         │  low diversity detection,
               │                     │  low rerank score triggering
               │                     │
               │  - Retrieval memory │  Tracks seen chunks/papers
               │                     │  across retry cycles
               └──────────┬──────────┘
                          │
                     Retry (up to N)
```

### Self-healing strategies

The system doesn't retry blindly. Each failure type triggers a specific recovery profile:

| Failure Type | What Happens |
|---|---|
| **Partial Grounding** | Rewrites query targeting unsupported claims, expands dense+BM25 top-k |
| **Low Confidence** | Rewrites query, emphasizes deep semantic vector search |
| **Insufficient Context** | Expands retrieval depth significantly (dense +20, BM25 +20) |
| **Irrelevant Retrieval** | Rewrites query, boosts BM25 lexical retrieval, diversifies across papers |
| **Ambiguous Query** | Disambiguates query before retrying |
| **Hallucination** | Rewrites with scientific precision, activates hybrid diversity strategy |

Additional signals override these defaults:
- If unsupported claims are detected, those concepts are injected into the rewritten query
- If source diversity is low (<3 papers), paper diversification is activated
- If max rerank score is very low (<-2.0), live arXiv acquisition is triggered

## Features

**Retrieval**
- Hybrid dense (Qdrant + BGE-large-en-v1.5) + sparse (BM25) search with Reciprocal Rank Fusion
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- Knowledge scope control — search custom uploads only, arXiv papers only, or both
- Post-retrieval filtering: noise, duplicate, and score-based chunk filtering

**Ingestion**
- PDF parsing via PyMuPDF with page-level text extraction
- Recursive character text splitting with configurable chunk size and overlap
- Multi-rule chunk validation (missing metadata, empty text, short chunks, invalid offsets)
- Custom document upload: PDF, DOCX, TXT, Markdown, CSV, JSON, code files
- Incremental vector + BM25 indexing per document (no full rebuild needed)

**Dynamic knowledge acquisition**
- Extracts research topic from query, searches arXiv metadata
- CrossEncoder reranks candidate paper titles + abstracts
- Top-k papers are downloaded, parsed, chunked, embedded, and indexed live
- BM25 index is rebuilt automatically after acquisition

**Query processing**
- Intent detection: 6 types (definition, comparison, explanation, literature review, fact lookup, research question)
- HyDE (Hypothetical Document Embeddings) — gated by intent, only fires for explanations and research questions
- Query decomposition for comparison queries
- Topic extraction strips conversational filler for cleaner retrieval

**Evaluation & healing**
- CRAG validation: grades document relevance (CORRECT/AMBIGUOUS/INCORRECT) based on reranker scores
- LLM-as-judge grounding evaluation with structured JSON output and multi-attempt parsing
- 7-type failure taxonomy with per-failure adaptive retrieval profiles
- Strategy registry: configurable healing actions per failure type
- Retrieval memory prevents redundant evidence across retry cycles
- Knowledge coverage estimator checks evidence sufficiency before generation

**Application layer**
- Multi-user auth (registration, login, SHA-256 password hashing, API key generation)
- Per-user session history with isolated query logs
- SQLite-backed telemetry: query count, grounding pass rate, avg latency, avg confidence, retry success rate, failure distribution
- Web dashboard with live telemetry gauges

## Tech stack

| Component | Technology |
|---|---|
| API server | FastAPI + Uvicorn |
| LLM inference | Groq API (LLaMA 3.3 70B) / Ollama (local fallback) |
| Embeddings | BAAI/bge-large-en-v1.5 (1024-dim) |
| Vector store | Qdrant (local persistent mode) |
| Lexical search | BM25 via rank_bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF parsing | PyMuPDF |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Persistence | SQLite |
| Config | YAML + Pydantic settings |
| Frontend | Vanilla HTML / CSS / JavaScript |

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

Get a free API key at [console.groq.com](https://console.groq.com).

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
├── api/              FastAPI routes, request/response schemas
├── core/             YAML config loader (Pydantic), structured logger
├── db/               SQLite persistence (users, sessions, query logs, telemetry)
├── embeddings/       Embedding model wrapper, batch generation, storage
├── evaluation/       Grounding evaluator (LLM-as-judge), CRAG validator,
│                     failure analyzer (7 failure types), decision engine,
│                     failure strategy registry
├── ingestion/        arXiv client, PDF downloader, custom document ingestor,
│                     topic extractor, dynamic knowledge acquisitor
├── llm/              Unified LLM client (Groq / OpenAI / Ollama),
│                     generator, generic LLM service
├── pipeline/         RAG pipeline orchestrator, intent detector (6 types),
│                     query enhancer (HyDE, decomposition)
├── preprocessing/    PDF parser (PyMuPDF), document chunker, chunk validator
├── prompt/           Prompt builder, templates, token validator
├── reranker/         CrossEncoder model, reranking logic
├── retrieval/        Dense retriever, BM25 retriever, BM25 index builder,
│                     RRF fusion, post-processors (noise/duplicate/score/topk),
│                     adaptive retrieval profiles, coverage estimator
├── self_healing/     Controller (main orchestrator), healing planner,
│                     query rewriter, strategy registry, retrieval memory
└── vectorstore/      Qdrant client, collection manager, vector indexer

configs/              config.yaml (all tunable parameters)
static/               Web dashboard (HTML, CSS, JS)
scripts/              Embedding generation, BM25 index building, utilities
tests/                Unit and integration tests
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/query` | Run a query through the self-healing pipeline |
| POST | `/api/upload` | Upload and index a document |
| POST | `/api/auth/register` | Create a user account |
| POST | `/api/auth/login` | Log in |
| GET | `/api/sessions` | Get session history |
| GET | `/api/telemetry` | System health and query stats |
| GET | `/api/health` | Health check |

## Configuration

All parameters in `configs/config.yaml`:

```yaml
self_healing:
  max_retries: 2                    # healing cycles before giving up
  confidence_threshold: 0.80        # minimum grounding confidence

embedding:
  model_name: BAAI/bge-large-en-v1.5

reranker:
  model_name: cross-encoder/ms-marco-MiniLM-L-6-v2

llm:
  provider: groq                    # groq, openai, or ollama
  model: llama-3.3-70b-versatile
```

## License

MIT
