from __future__ import annotations

from app.core.logger import logger
from app.retrieval.models import RetrievedChunk
from app.retrieval.processors.base import RetrievalProcessor


class ScoreFilter(RetrievalProcessor):
    """
    Removes retrievals below a similarity threshold.
    """

    def __init__(
        self,
        minimum_score: float = 0.55,
    ) -> None:

        self.minimum_score = minimum_score

    def process(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:

        filtered = [
            chunk
            for chunk in chunks
            if chunk.score >= self.minimum_score
        ]

        logger.info(
            f"ScoreFilter: {len(filtered)} chunks remain."
        )

        return filtered
    