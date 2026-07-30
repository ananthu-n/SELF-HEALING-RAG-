from __future__ import annotations

import time
import uuid
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.core.logger import logger
from app.db.database import init_db, DatabaseManager

_upload_ingestor = None

def get_upload_ingestor():
    global _upload_ingestor
    if _upload_ingestor is None:
        from app.ingestion.custom_upload import CustomDatasourceIngestor
        _upload_ingestor = CustomDatasourceIngestor()
    return _upload_ingestor

app = FastAPI(
    title="Autonomous Self-Healing Research Assistant Enterprise API",
    description="Production-grade API with User Authentication, Persistent Sessions, Telemetry Analytics, and Intent Routing",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    logger.info("Initializing SQLite Persistence Database & Telemetry tables...")
    init_db()
    logger.success("Database initialized successfully.")


_controller = None

def get_controller():
    global _controller
    if _controller is None:
        from app.self_healing.controller import SelfHealingController
        _controller = SelfHealingController()
    return _controller


class AuthRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    status: str
    user_id: str
    username: str
    api_key: str


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "demo_user"
    search_scope: Optional[str] = "hybrid"  # "hybrid", "custom_only", or "arxiv_only"


class QueryResponse(BaseModel):
    session_id: str
    user_id: str
    query: str
    answer: str
    intent: str
    search_query: str
    search_scope: str
    retry_count: int
    is_grounded: bool
    grounding_confidence: float
    failure_type: str
    failure_reason: str
    retrieved_chunks: list[dict[str, Any]]
    healing_history: list[dict[str, Any]]
    latency_sec: float


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "Autonomous Self-Healing Research Assistant",
        "version": "2.1.0-Enterprise",
        "auth": "Enabled",
        "persistence": "SQLite Active",
    }


@app.post("/api/auth/register", response_model=AuthResponse)
def register_user(req: AuthRequest):
    if not req.username.strip() or not req.password.strip():
        raise HTTPException(status_code=400, detail="Username and password are required.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    try:
        user = DatabaseManager.register_user(req.username, req.password)
        logger.info(f"User registered: {user['username']} ({user['user_id']})")
        return AuthResponse(
            status="success",
            user_id=user["user_id"],
            username=user["username"],
            api_key=user["api_key"],
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:
        logger.error(f"Registration error: {err}")
        raise HTTPException(status_code=500, detail="Registration failed.")


@app.post("/api/auth/login", response_model=AuthResponse)
def login_user(req: AuthRequest):
    user = DatabaseManager.login_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    logger.info(f"User authenticated: {user['username']} ({user['user_id']})")
    return AuthResponse(
        status="success",
        user_id=user["user_id"],
        username=user["username"],
        api_key=user["api_key"],
    )


@app.post("/api/query", response_model=QueryResponse)
def handle_query(
    req: QueryRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    user_id = x_user_id or req.user_id or "demo_user"
    session_id = x_session_id or req.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    scope = req.search_scope or "hybrid"

    logger.info(f"API Request received [User: {user_id} | Session: {session_id} | Scope: {scope}]: '{req.query}'")
    start_t = time.perf_counter()

    try:
        controller = get_controller()
        state = controller.answer(req.query, search_scope=scope)
        elapsed = time.perf_counter() - start_t


        final_gen = "No answer generated."
        if state.last_generation:
            if hasattr(state.last_generation, "response") and state.last_generation.response:
                final_gen = state.last_generation.response.answer
            elif hasattr(state.last_generation, "answer"):
                final_gen = getattr(state.last_generation, "answer")

        intent_str = "RESEARCH_QUESTION"
        if hasattr(state, "query_intent"):
            intent_str = str(getattr(state, "query_intent"))

        is_grounded = state.last_grounding.response.is_grounded if state.last_grounding else False
        confidence = state.last_grounding.response.confidence if state.last_grounding else 0.0

        failure_type = state.last_failure.failure_type.value if state.last_failure else "none"
        failure_reason = state.last_failure.reason if state.last_failure else "None"

        # Format retrieved chunks
        chunks_data = []
        if state.rerank_result and state.rerank_result.reranked_chunks:
            for chunk in state.rerank_result.reranked_chunks:
                chunks_data.append({
                    "paper_id": chunk.paper_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "reranker_score": round(chunk.reranker_score, 4),
                    "page_number": chunk.page_number,
                })

        # Format healing history
        history_data = []
        for plan in state.healing_history:
            history_data.append({
                "retry_number": plan.retry_number,
                "reason": plan.reason,
                "retrieval_strategy": plan.retrieval_strategy.value,
                "rewrite_strategy": plan.rewrite_strategy.value if plan.rewrite_strategy else "none",
                "top_k": plan.top_k,
                "dense_top_k": plan.dense_top_k,
                "bm25_top_k": plan.bm25_top_k,
            })

        # Persist execution trace & telemetry to SQLite Database
        DatabaseManager.save_query_log(
            user_id=user_id,
            session_id=session_id,
            query=req.query,
            search_query=state.current_query,
            answer=final_gen,
            intent=intent_str,
            retry_count=state.retry_count,
            is_grounded=is_grounded,
            grounding_confidence=confidence,
            failure_type=failure_type,
            failure_reason=failure_reason,
            latency_sec=round(elapsed, 2),
            retrieved_chunks=chunks_data,
            healing_history=history_data,
        )

        return QueryResponse(
            session_id=session_id,
            user_id=user_id,
            query=req.query,
            answer=final_gen,
            intent=intent_str,
            search_query=state.current_query,
            search_scope=scope,
            retry_count=state.retry_count,
            is_grounded=is_grounded,
            grounding_confidence=confidence,
            failure_type=failure_type,
            failure_reason=failure_reason,
            retrieved_chunks=chunks_data,
            healing_history=history_data,
            latency_sec=round(elapsed, 2),
        )
    except Exception as err:
        logger.error(f"API Error processing query: {err}")
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/api/sessions")
def get_user_sessions(user_id: str = "demo_user"):
    """List persistent sessions for a user."""
    sessions = DatabaseManager.get_user_sessions(user_id)
    return {"user_id": user_id, "sessions": sessions}


@app.get("/api/sessions/{session_id}")
def get_session_history(session_id: str):
    """Retrieve full persistent query log history for a session."""
    history = DatabaseManager.get_session_history(session_id)
    if not history:
        return {"session_id": session_id, "history": []}
    return {"session_id": session_id, "history": history}


@app.get("/api/telemetry")
def get_telemetry():
    """Retrieve aggregate production performance metrics and grounding stats."""
    metrics = DatabaseManager.get_telemetry_metrics()
    return {"status": "success", "metrics": metrics}


@app.post("/api/upload")
async def upload_datasource(file: UploadFile = File(...)):
    """Upload custom PDF or text document to index in RAG vector database."""
    from pathlib import Path
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    allowed_exts = {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".csv",
        ".json", ".jsonl", ".html", ".htm", ".xml", ".yaml", ".yml",
        ".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".cpp", ".java",
        ".rst", ".log", ".tsv", ".ini", ".cfg", ".conf"
    }
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Supported formats include PDF, Word (.docx), TXT, Markdown, CSV, JSON, Code files, and Logs.")

    try:
        doc_id = f"custom_{uuid.uuid4().hex[:8]}"
        raw_pdf_dir = Path("data/raw_pdfs")
        raw_pdf_dir.mkdir(parents=True, exist_ok=True)
        save_path = raw_pdf_dir / f"{doc_id}_{file.filename}"

        with open(save_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        ingestor = get_upload_ingestor()
        result = ingestor.process_file(save_path, doc_id)

        return {
            "status": "success",
            "message": f"Successfully indexed '{file.filename}' into RAG knowledge base.",
            "details": result,
        }
    except Exception as err:
        logger.error(f"Error uploading datasource: {err}")
        raise HTTPException(status_code=500, detail=str(err))


# Serve static web frontend files
app.mount("/", StaticFiles(directory="static", html=True), name="static")
