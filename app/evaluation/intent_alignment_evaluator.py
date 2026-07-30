from __future__ import annotations

from pydantic import BaseModel
from app.core.logger import logger
from app.pipeline.intent import QueryIntent


class IntentAlignmentResult(BaseModel):
    is_aligned: bool
    confidence_score: float
    reason: str


class IntentAlignmentEvaluator:
    """
    Evaluates whether a generated response satisfies the user's detected query intent.
    """

    @classmethod
    def evaluate(
        cls,
        query: str,
        answer: str,
        intent: QueryIntent,
    ) -> IntentAlignmentResult:
        ans_lower = answer.lower().strip()

        if intent == QueryIntent.DEFINITION:
            aligned = any(kw in ans_lower for kw in ["is a", "is defined as", "refers to", "concept", "technique", "method"])
            reason = "Definition intent satisfied." if aligned else "Answer fails to provide a clear definition."
            return IntentAlignmentResult(is_aligned=aligned, confidence_score=0.9 if aligned else 0.3, reason=reason)

        elif intent == QueryIntent.COMPARISON:
            aligned = any(kw in ans_lower for kw in ["whereas", "compared to", "differ", "unlike", "advantage", "trade-off"])
            reason = "Comparison intent satisfied." if aligned else "Answer fails to compare the requested entities."
            return IntentAlignmentResult(is_aligned=aligned, confidence_score=0.9 if aligned else 0.3, reason=reason)

        elif intent == QueryIntent.EXPLANATION:
            aligned = any(kw in ans_lower for kw in ["by", "through", "mechanism", "process", "works by", "step"])
            reason = "Explanation intent satisfied." if aligned else "Answer lacks mechanical/process explanation."
            return IntentAlignmentResult(is_aligned=aligned, confidence_score=0.85 if aligned else 0.4, reason=reason)

        # Default alignment for GENERAL_QA, LITERATURE_REVIEW, FACT_LOOKUP
        aligned = len(ans_lower) > 30 and "insufficient" not in ans_lower
        return IntentAlignmentResult(
            is_aligned=aligned,
            confidence_score=0.85 if aligned else 0.2,
            reason="Response aligns with general query intent." if aligned else "Response is insufficient.",
        )
