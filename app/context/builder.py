from __future__ import annotations

from app.context.models import (
    ContextResult,
    ContextStatistics,
)
from app.context.utils import TokenEstimator
from app.core.logger import logger
from app.reranker.models import RerankResult


class ContextBuilder:
    """
    Build an LLM-ready context from reranked chunks.

    Responsibilities
    ----------------
    - Respect token budget
    - Preserve ranking order
    - Preserve metadata
    - Compute context statistics
    """

    def __init__(
        self,
        token_budget: int = 4000,
    ) -> None:

        self.token_budget = token_budget

    # ---------------------------------------------------------

    def build(
        self,
        rerank_result: RerankResult,
    ) -> ContextResult:

        logger.info(
            f"Building context from "
            f"{rerank_result.total_results} reranked chunks."
        )

        selected_chunks = []

        used_tokens = 0

        used_characters = 0

        papers = set()

        truncated = False

        for chunk in rerank_result.reranked_chunks:

            tokens = TokenEstimator.estimate(
                chunk.text
            )

            if used_tokens + tokens > self.token_budget:

                truncated = True

                break

            selected_chunks.append(chunk)

            used_tokens += tokens

            used_characters += len(chunk.text)

            papers.add(chunk.paper_id)

        statistics = ContextStatistics(

            estimated_tokens=used_tokens,

            total_characters=used_characters,

            unique_papers=len(papers),

            token_budget=self.token_budget,

            remaining_budget=self.token_budget - used_tokens,

            truncated=truncated,

        )

        logger.success(

            f"Context built with "

            f"{len(selected_chunks)} chunks "

            f"({used_tokens} estimated tokens)."

        )

        return ContextResult(

            query=rerank_result.query,

            context_chunks=selected_chunks,

            total_chunks=len(selected_chunks),

            statistics=statistics,

        )

    # ---------------------------------------------------------

    def __call__(
        self,
        rerank_result: RerankResult,
    ) -> ContextResult:

        return self.build(rerank_result)