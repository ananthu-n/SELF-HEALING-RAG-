from __future__ import annotations

from pydantic import BaseModel, Field

from app.reranker.models import RerankedChunk


class ContextStatistics(BaseModel):
    """
    Statistics describing the generated context.
    """

    estimated_tokens: int = 0

    total_characters: int = 0

    unique_papers: int = 0

    token_budget: int = 0

    remaining_budget: int = 0

    truncated: bool = False


class ContextResult(BaseModel):
    """
    Context prepared for LLM generation.
    """

    query: str

    context_chunks: list[RerankedChunk] = Field(
        default_factory=list
    )

    total_chunks: int = 0

    statistics: ContextStatistics = Field(
        default_factory=ContextStatistics
    )