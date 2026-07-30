from __future__ import annotations

from app.core.logger import logger
from app.retrieval.models import RetrievalResult
from app.reranker.cross_encoder import CrossEncoderModel
from app.reranker.models import (
    RerankedChunk,
    RerankResult,
)


class CrossEncoderReranker:
    """
    CrossEncoder-based reranker.

    Pipeline
    --------

    RetrievalResult
            │
            ▼
    CrossEncoder
            │
            ▼
    RerankResult
    """

    def __init__(self) -> None:

        self.model = CrossEncoderModel()

    # ---------------------------------------------------------

    def rerank(
        self,
        retrieval_result: RetrievalResult,
        top_k: int | None = None,
    ) -> RerankResult:
        """
        Rerank retrieved chunks.

        Parameters
        ----------
        retrieval_result
            Output from HybridRetriever.

        top_k
            Keep only the top-k chunks after reranking.
        """

        query = retrieval_result.query

        chunks = retrieval_result.retrieved_chunks

        if not chunks:

            logger.warning(
                "No retrieved chunks received for reranking."
            )

            return RerankResult(
                query=query,
                reranked_chunks=[],
                total_results=0,
            )

        logger.info(
            f"Reranking {len(chunks)} retrieved chunks."
        )

        # -----------------------------------------
        # Build query-document pairs
        # -----------------------------------------

        pairs = [
            (query, chunk.text)
            for chunk in chunks
        ]

        # -----------------------------------------
        # CrossEncoder inference
        # -----------------------------------------

        scores = self.model.predict(pairs)

        # -----------------------------------------
        # Attach scores
        # -----------------------------------------

        reranked_chunks = []

        for chunk, score in zip(chunks, scores):

            reranked_chunks.append(

                RerankedChunk(

                    paper_id=chunk.paper_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=chunk.score,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    metadata=dict(chunk.metadata),
                    reranker_score=float(score),

                )

            )

        # -----------------------------------------
        # Sort
        # -----------------------------------------

        reranked_chunks.sort(
            key=lambda chunk: chunk.reranker_score,
            reverse=True,
        )

        if top_k is not None:

            reranked_chunks = reranked_chunks[:top_k]

        # ---------------------------------------------------------
        # Preserve reranking metadata
        # ---------------------------------------------------------

        for reranker_rank, chunk in enumerate(
            reranked_chunks,
            start=1,
        ):

            chunk.metadata["reranker_rank"] = reranker_rank

            chunk.metadata["reranker_score"] = chunk.reranker_score

        logger.success(
            f"Reranking completed with "
            f"{len(reranked_chunks)} chunks."
        )

        return RerankResult(
            query=query,
            reranked_chunks=reranked_chunks,
            total_results=len(reranked_chunks),
        )

    # ---------------------------------------------------------

    def __call__(
        self,
        retrieval_result: RetrievalResult,
        top_k: int | None = None,
    ) -> RerankResult:

        return self.rerank(
            retrieval_result=retrieval_result,
            top_k=top_k,
        )