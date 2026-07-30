from __future__ import annotations

from app.core.logger import logger
from app.retrieval.models import RetrievalResult
from app.retrieval.payload_mapper import PayloadMapper
from app.retrieval.vector_search import VectorSearchService


class DenseRetriever:
    """
    Dense semantic retriever.

    Responsibilities
    ----------------
    • Generate query embedding
    • Execute Qdrant vector search
    • Convert results into RetrievedChunk models

    Does NOT:
        - rerank
        - validate
        - self-heal
        - rewrite queries
    """

    def __init__(self) -> None:

        self.search_service = VectorSearchService()

    # ---------------------------------------------------------
    # Main Retrieval
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve the most semantically similar chunks.
        """

        logger.info(f"Dense retrieval started: '{query}'")

        scored_points = self.search_service.search(
            query=query,
            top_k=top_k,
        )

        chunks = PayloadMapper.map_many(scored_points)



        # ---------------------------------------------------------
        # Preserve retrieval metadata
        # ---------------------------------------------------------

        for rank, chunk in enumerate(chunks, start=1):

            chunk.metadata["retrieval_source"] = "dense"
            chunk.metadata["retrieval_rank"] = rank

        logger.success(
            f"Retrieved {len(chunks)} semantic chunks."
        )

        return RetrievalResult(
            query=query,
            retrieved_chunks=chunks,
            total_results=len(chunks),
        )