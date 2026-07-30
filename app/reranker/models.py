from __future__ import annotations

from pydantic import BaseModel

from app.retrieval.models import RetrievedChunk


class RerankedChunk(RetrievedChunk):
    """
    Retrieved chunk enriched with a CrossEncoder score.

    Inherits every field from RetrievedChunk:

    - paper_id
    - chunk_id
    - text
    - score               (original retrieval score)
    - page_number
    - chunk_index
    - char_start
    - char_end
    - metadata

    Adds:
    - reranker_score
    """

    reranker_score: float


class RerankResult(BaseModel):
    """
    Output of the CrossEncoder reranker.
    """

    query: str

    reranked_chunks: list[RerankedChunk]

    total_results: int