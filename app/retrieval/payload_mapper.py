from __future__ import annotations

from typing import Any

from qdrant_client.models import ScoredPoint

from app.retrieval.models import RetrievedChunk


class PayloadMapper:
    """
    Converts raw Qdrant ScoredPoint objects into
    RetrievedChunk objects.

    This isolates Qdrant-specific payload parsing from
    the rest of the retrieval pipeline.
    """

    @staticmethod
    def map(point: ScoredPoint) -> RetrievedChunk:
        """
        Convert one ScoredPoint into a RetrievedChunk.
        """

        payload: dict[str, Any] = point.payload or {}

        return RetrievedChunk(
            paper_id=payload.get("paper_id", ""),
            chunk_id=payload.get("chunk_id", ""),
            text=payload.get("text", ""),
            score=float(point.score or 0.0),
            page_number=payload.get("page_number", 0),
            chunk_index=payload.get("chunk_index", 0),
            char_start=payload.get("char_start", 0),
            char_end=payload.get("char_end", 0),
            metadata={
                key: value
                for key, value in payload.items()
                if key not in {
                    "paper_id",
                    "chunk_id",
                    "text",
                    "page_number",
                    "chunk_index",
                    "char_start",
                    "char_end",
                }
            },
        )

    @classmethod
    def map_many(
        cls,
        points: list[ScoredPoint],
    ) -> list[RetrievedChunk]:
        """
        Convert multiple search results.
        """

        return [cls.map(point) for point in points]