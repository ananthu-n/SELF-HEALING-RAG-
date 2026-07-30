from __future__ import annotations

from app.core.logger import logger
from app.retrieval.bm25_payload_mapper import BM25PayloadMapper
from app.retrieval.bm25_search import BM25SearchService
from app.retrieval.models import RetrievalResult



class BM25Retriever:
    """
    Sparse lexical retriever.

    Responsibilities
    ----------------
    • Execute BM25 search
    • Convert search results into RetrievedChunk models
    • Pass retrieved chunks through RetrievalPipeline

    Does NOT:
        - rerank
        - validate
        - self-heal
        - rewrite queries
    """

    def __init__(self) -> None:

        self.search_service = BM25SearchService()

    # ---------------------------------------------------------
    # Main Retrieval
    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve the most lexically relevant chunks.
        """

        logger.info(f"BM25 retrieval started: '{query}'")

        search_results = self.search_service.search(
            query=query,
            top_k=top_k,
        )

        chunks = BM25PayloadMapper.map_many(
            documents=[
                result.document
                for result in search_results
            ],
            scores=[
                result.score
                for result in search_results
            ],
        )



        # ---------------------------------------------------------
        # Preserve retrieval metadata
        # ---------------------------------------------------------

        for rank, chunk in enumerate(chunks, start=1):

            chunk.metadata["retrieval_source"] = "bm25"
            chunk.metadata["retrieval_rank"] = rank

        logger.success(
            f"Retrieved {len(chunks)} lexical chunks."
        )

        return RetrievalResult(
            query=query,
            retrieved_chunks=chunks,
            total_results=len(chunks),
        )