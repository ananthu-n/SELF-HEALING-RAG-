from __future__ import annotations

from app.core.logger import logger
from app.retrieval.models import RetrievedChunk
from app.retrieval.processors.base import RetrievalProcessor


class DuplicateFilter(RetrievalProcessor):
    """
    Removes duplicate retrieved chunks while keeping
    the highest scoring occurrence.
    """

    def process(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:

        best_chunks: dict[str, RetrievedChunk] = {}

        for chunk in chunks:

            existing = best_chunks.get(chunk.chunk_id)

            if existing is None:
                best_chunks[chunk.chunk_id] = chunk

            elif chunk.score > existing.score:
                best_chunks[chunk.chunk_id] = chunk

        cleaned = list(best_chunks.values())

        logger.info(
            f"DuplicateFilter: {len(cleaned)} chunks remain."
        )

        return cleaned