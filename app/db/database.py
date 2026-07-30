import sqlite3
import json
import time
import uuid
import hashlib
from pathlib import Path
from typing import Any, List, Dict, Optional

DB_PATH = Path("data/app_persistence.db")


def get_db_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            created_at REAL NOT NULL
        );
    """)

    # Migration check for password_hash column
    cursor.execute("PRAGMA table_info(users)")
    cols = [row["name"] for row in cursor.fetchall()]
    if "password_hash" not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN password_hash TEXT DEFAULT ''")

    # Sessions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
    """)

    # Query Executions & Telemetry Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            search_query TEXT NOT NULL,
            answer TEXT NOT NULL,
            intent TEXT NOT NULL,
            retry_count INTEGER NOT NULL,
            is_grounded INTEGER NOT NULL,
            grounding_confidence REAL NOT NULL,
            failure_type TEXT NOT NULL,
            failure_reason TEXT NOT NULL,
            latency_sec REAL NOT NULL,
            retrieved_chunks JSON NOT NULL,
            healing_history JSON NOT NULL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        );
    """)

    # Create default demo user if not exists
    cursor.execute("SELECT user_id FROM users WHERE user_id = 'demo_user'")
    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO users (user_id, username, password_hash, api_key, created_at) VALUES (?, ?, ?, ?, ?)",
            ("demo_user", "demo_user", _hash_password("demopass123"), "sk-selfhealing-demo-key", time.time())
        )

    conn.commit()
    conn.close()


class DatabaseManager:
    """
    Manages persistent storage for authentication, sessions, queries, and telemetry.
    """

    @classmethod
    def register_user(cls, username: str, password: str) -> Dict[str, Any]:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()

        username_clean = username.strip().lower()
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username_clean,))
        if cursor.fetchone():
            conn.close()
            raise ValueError("Username already registered.")

        user_id = f"user_{uuid.uuid4().hex[:10]}"
        api_key = f"sk-selfhealing-{uuid.uuid4().hex[:16]}"
        pwd_hash = _hash_password(password)
        now = time.time()

        cursor.execute(
            "INSERT INTO users (user_id, username, password_hash, api_key, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username_clean, pwd_hash, api_key, now)
        )
        conn.commit()
        conn.close()

        return {
            "user_id": user_id,
            "username": username_clean,
            "api_key": api_key,
            "created_at": now
        }

    @classmethod
    def login_user(cls, username: str, password: str) -> Optional[Dict[str, Any]]:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()

        username_clean = username.strip().lower()
        pwd_hash = _hash_password(password)

        cursor.execute(
            "SELECT user_id, username, api_key, created_at FROM users WHERE username = ? AND password_hash = ?",
            (username_clean, pwd_hash)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    @classmethod
    def save_query_log(
        cls,
        user_id: str,
        session_id: str,
        query: str,
        search_query: str,
        answer: str,
        intent: str,
        retry_count: int,
        is_grounded: bool,
        grounding_confidence: float,
        failure_type: str,
        failure_reason: str,
        latency_sec: float,
        retrieved_chunks: List[Dict[str, Any]],
        healing_history: List[Dict[str, Any]],
    ) -> int:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()

        now = time.time()

        # Ensure session exists
        cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
        if not cursor.fetchone():
            title = query[:40] + ("..." if len(query) > 40 else "")
            cursor.execute(
                "INSERT INTO sessions (session_id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, title, now, now)
            )
        else:
            cursor.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id)
            )

        cursor.execute("""
            INSERT INTO query_logs (
                session_id, user_id, query, search_query, answer, intent,
                retry_count, is_grounded, grounding_confidence, failure_type,
                failure_reason, latency_sec, retrieved_chunks, healing_history, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, user_id, query, search_query, answer, intent,
            retry_count, 1 if is_grounded else 0, grounding_confidence, failure_type,
            failure_reason, latency_sec, json.dumps(retrieved_chunks), json.dumps(healing_history), now
        ))

        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return log_id

    @classmethod
    def get_user_sessions(cls, user_id: str) -> List[Dict[str, Any]]:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, title, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    @classmethod
    def get_session_history(cls, session_id: str) -> List[Dict[str, Any]]:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, query, search_query, answer, intent, retry_count,
                   is_grounded, grounding_confidence, failure_type, failure_reason,
                   latency_sec, retrieved_chunks, healing_history, timestamp
            FROM query_logs
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,))
        rows = cursor.fetchall()
        conn.close()
        result = []
        for row in rows:
            d = dict(row)
            d["is_grounded"] = bool(d["is_grounded"])
            d["retrieved_chunks"] = json.loads(d["retrieved_chunks"])
            d["healing_history"] = json.loads(d["healing_history"])
            result.append(d)
        return result

    @classmethod
    def get_telemetry_metrics(cls) -> Dict[str, Any]:
        init_db()
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM query_logs")
        total_queries = cursor.fetchone()["total"]

        if total_queries == 0:
            conn.close()
            return {
                "total_queries": 0,
                "grounding_pass_rate": 0.0,
                "avg_latency_sec": 0.0,
                "avg_confidence": 0.0,
                "retry_success_rate": 0.0,
                "failure_distribution": {},
            }

        cursor.execute("SELECT COUNT(*) as grounded FROM query_logs WHERE is_grounded = 1")
        grounded_count = cursor.fetchone()["grounded"]

        cursor.execute("SELECT AVG(latency_sec) as avg_lat, AVG(grounding_confidence) as avg_conf FROM query_logs")
        row = cursor.fetchone()
        avg_lat = row["avg_lat"] or 0.0
        avg_conf = row["avg_conf"] or 0.0

        cursor.execute("SELECT failure_type, COUNT(*) as cnt FROM query_logs GROUP BY failure_type")
        fail_rows = cursor.fetchall()
        fail_dist = {r["failure_type"]: r["cnt"] for r in fail_rows}

        cursor.execute("SELECT COUNT(*) as retried FROM query_logs WHERE retry_count > 0")
        retried_count = cursor.fetchone()["retried"]

        cursor.execute("SELECT COUNT(*) as retried_grounded FROM query_logs WHERE retry_count > 0 AND is_grounded = 1")
        retried_grounded = cursor.fetchone()["retried_grounded"]

        retry_success = (retried_grounded / retried_count * 100.0) if retried_count > 0 else 100.0

        conn.close()
        return {
            "total_queries": total_queries,
            "grounding_pass_rate": round((grounded_count / total_queries) * 100.0, 1),
            "avg_latency_sec": round(avg_lat, 2),
            "avg_confidence": round(avg_conf, 2),
            "retry_success_rate": round(retry_success, 1),
            "failure_distribution": fail_dist,
        }
