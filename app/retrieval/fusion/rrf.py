from __future__ import annotations

from collections import defaultdict

from app.core.logger import logger
from app.retrieval.models import RetrievedChunk, RetrievalResult


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF).

    Reference:
        Cormack, Clarke & Buettcher (SIGIR 2009)

    RRF Score:
        score = Σ (1 / (k + rank))
    """

    def __init__(
        self,
        k: int = 60,
    ) -> None:

        self.k = k

    @staticmethod
    def _chunk_key(chunk: RetrievedChunk) -> str:
        """
        Unique identifier used for fusion.
        """
        return chunk.chunk_id

    def fuse(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Fuse multiple retrieval results using Reciprocal Rank Fusion.
        """

        logger.info(
            f"Running RRF on {len(results)} retrieval result(s)."
        )

        fusion_scores: dict[str, float] = defaultdict(float)

        merged_chunks: dict[str, RetrievedChunk] = {}

        for retrieval_result in results:

            for rank, chunk in enumerate(
                retrieval_result.retrieved_chunks,
                start=1,
            ):

                key = self._chunk_key(chunk)

                fusion_scores[key] += 1.0 / (self.k + rank)

                if key not in merged_chunks:

                    merged_chunks[key] = chunk

        ranked_chunks = sorted(
            merged_chunks.values(),
            key=lambda chunk: fusion_scores[self._chunk_key(chunk)],
            reverse=True,
        )

        fused_chunks: list[RetrievedChunk] = []

        for rrf_rank, chunk in enumerate(
            ranked_chunks[:top_k],
            start=1,
        ):

            # Create a copy of metadata so we don't mutate
            # objects shared with previous stages.
            chunk.metadata = dict(chunk.metadata)

            chunk.metadata["rrf_rank"] = rrf_rank

            chunk.metadata["rrf_score"] = fusion_scores[
                self._chunk_key(chunk)
            ]

            fused_chunks.append(chunk)

        logger.success(
            f"Fusion completed with {len(fused_chunks)} chunks."
        )

        return RetrievalResult(
            query=query,
            retrieved_chunks=fused_chunks,
            total_results=len(fused_chunks),
        )

    def __call__(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> RetrievalResult:

        return self.fuse(
            query=query,
            results=results,
            top_k=top_k,
        )