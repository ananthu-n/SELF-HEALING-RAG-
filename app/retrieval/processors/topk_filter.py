from __future__ import annotations

from app.retrieval.models import RetrievedChunk
from app.retrieval.processors.base import RetrievalProcessor


class TopKFilter(RetrievalProcessor):
    """
    Keeps only the top-k highest scoring chunks.
    """

    def __init__(
        self,
        top_k: int,
    ) -> None:

        self.top_k = top_k

    def process(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:

        chunks.sort(
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        return chunks[: self.top_k]