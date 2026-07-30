from __future__ import annotations

from enum import Enum
from typing import Sequence
from pydantic import BaseModel
from app.core.logger import logger
from app.reranker.models import RerankedChunk


class CRAGState(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class CRAGResult(BaseModel):
    state: CRAGState
    confidence_score: float
    reason: str
    requires_external_search: bool


class CRAGValidator:
    """
    Corrective RAG (CRAG) Document Validation Engine.

    Grades document relevance to classify retrieval into:
    - CORRECT: High confidence -> proceed directly
    - AMBIGUOUS: Moderate confidence -> refine context & expand queries
    - INCORRECT: Low confidence -> trigger dynamic arXiv search fallback
    """

    @classmethod
    def validate(
        cls,
        query: str,
        reranked_chunks: Sequence[RerankedChunk],
    ) -> CRAGResult:
        if not reranked_chunks:
            return CRAGResult(
                state=CRAGState.INCORRECT,
                confidence_score=0.0,
                reason="No reranked chunks available.",
                requires_external_search=True,
            )

        max_rerank_score = max(c.reranker_score for c in reranked_chunks)

        # Classify based on CrossEncoder reranker score thresholds
        if max_rerank_score >= 3.0:
            state = CRAGState.CORRECT
            requires_search = False
            reason = f"Documents highly relevant (Max Rerank Score: {max_rerank_score:.2f})."
        elif max_rerank_score >= 0.0:
            state = CRAGState.AMBIGUOUS
            requires_search = True
            reason = f"Documents partially relevant (Max Rerank Score: {max_rerank_score:.2f}). arXiv acquisition recommended."
        else:
            state = CRAGState.INCORRECT
            requires_search = True
            reason = f"Documents irrelevant (Max Rerank Score: {max_rerank_score:.2f}). Dynamic arXiv acquisition required."

        confidence = max(0.0, min(1.0, (max_rerank_score + 5.0) / 10.0))

        logger.info(f"CRAG Validation: State={state.value.upper()} | Confidence={confidence:.2f} | Reason={reason}")

        return CRAGResult(
            state=state,
            confidence_score=confidence,
            reason=reason,
            requires_external_search=requires_search,
        )
