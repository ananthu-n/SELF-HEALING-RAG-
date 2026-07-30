from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """
    Represents a single retrieved chunk returned from any retriever.
    """

    paper_id: str
    chunk_id: str

    text: str

    score: float = Field(
        default=0.0,
        description="Retriever similarity score."
    )

    page_number: int
    chunk_index: int

    char_start: int
    char_end: int

    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """
    Collection of retrieved chunks.
    """

    query: str

    retrieved_chunks: list[RetrievedChunk]

    total_results: int