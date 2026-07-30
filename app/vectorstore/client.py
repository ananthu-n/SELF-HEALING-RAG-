import os
from pathlib import Path

from qdrant_client import QdrantClient

from app.core.config import settings
from app.core.logger import logger


class QdrantDB:
    """
    Singleton Qdrant client.

    Supports local embedded Qdrant and Qdrant Host server.
    """

    _client = None

    @classmethod
    def get_client(cls) -> QdrantClient:
        if cls._client is None:
            qdrant_host = os.getenv("QDRANT_HOST")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

            if qdrant_host:
                logger.info(f"Connecting to Qdrant server at {qdrant_host}:{qdrant_port}")
                cls._client = QdrantClient(host=qdrant_host, port=qdrant_port)
            else:
                db_path = Path(settings.vectorstore.location)
                db_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Opening embedded Qdrant database: {db_path}")
                cls._client = QdrantClient(path=str(db_path))

            logger.success("Connected to Qdrant.")

        return cls._client

    @classmethod
    def close(cls):
        if cls._client is not None:
            cls._client.close()
            cls._client = None
            logger.info("Qdrant connection closed.")