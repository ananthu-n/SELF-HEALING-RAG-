from __future__ import annotations

from pydantic import BaseModel


class RetrievalValidation(BaseModel):
    """
    Output of retrieval validation.
    """

    is_valid: bool

    should_retry: bool

    reason: str

    average_score: float

    maximum_score: float

    minimum_score: float

    total_chunks: int

    unique_papers: int