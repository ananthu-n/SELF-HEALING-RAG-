import threading
from app.core.logger import logger

class CancellationManager:
    """
    Thread-safe cancellation tracker for active RAG sessions.
    """
    _lock = threading.Lock()
    _cancelled_sessions = set()

    @classmethod
    def cancel(cls, session_id: str):
        with cls._lock:
            cls._cancelled_sessions.add(session_id)
        logger.info(f"Session '{session_id}' registered in CancellationManager.")

    @classmethod
    def is_cancelled(cls, session_id: str) -> bool:
        if not session_id:
            return False
        with cls._lock:
            cancelled = session_id in cls._cancelled_sessions
        if cancelled:
            logger.warning(f"Session '{session_id}' has a cancellation signal!")
        return cancelled

    @classmethod
    def clear(cls, session_id: str):
        with cls._lock:
            cls._cancelled_sessions.discard(session_id)
