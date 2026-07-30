from __future__ import annotations

from qdrant_client.models import ScoredPoint

from app.core.config import settings
from app.core.logger import logger
from app.embeddings.model import EmbeddingModel
from app.vectorstore.client import QdrantDB


class VectorSearchService:
    """
    Low-level vector search service.

    Responsibilities
    ----------------
    • Encode user queries
    • Execute semantic search in Qdrant
    • Return raw ScoredPoint objects

    Does NOT:
        - map payloads
        - rerank
        - filter
        - validate
    """

    def __init__(self) -> None:

        self.client = QdrantDB.get_client()

        self.collection = settings.vectorstore.collection_name

        self.embedding_model = EmbeddingModel()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def embed_query(self, query: str) -> list[float]:
        """
        Convert a query into an embedding vector.
        """

        logger.info("Generating query embedding.")

        embedding = self.embedding_model.encode([query])[0]

        return embedding.tolist()

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[ScoredPoint]:
        """
        Execute semantic vector search.

        Parameters
        ----------
        query:
            User query.

        top_k:
            Number of results to retrieve.
        """

        vector = self.embed_query(query)

        logger.info(
            f"Searching Qdrant (top_k={top_k})"
        )

        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        logger.success(
            f"Retrieved {len(results.points)} candidates."
        )

        return results.points