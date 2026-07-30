from __future__ import annotations

from abc import ABC, abstractmethod

from app.retrieval.models import RetrievedChunk


class RetrievalProcessor(ABC):
    """
    Base interface for every retrieval processor.
    """

    @abstractmethod
    def process(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Process retrieved chunks.
        """
        raise NotImplementedError