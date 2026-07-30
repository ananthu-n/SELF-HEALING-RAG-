from __future__ import annotations

from app.core.config import settings
from app.core.logger import logger

from app.evaluation.models import GroundingResult
from app.evaluation.failure_models import (
    FailureAnalysis,
    FailureType,
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline.models import PipelineResult


class FailureAnalyzer:
    """
    Analyze the grounding result and determine
    the most probable root cause of failure.
    """

    def analyze(
        self,
        grounding: GroundingResult,
        pipeline_result: PipelineResult | None = None,
    ) -> FailureAnalysis:

        response = grounding.response

        confidence_threshold = (
            settings.self_healing.confidence_threshold
        )

        logger.info("Analyzing grounding result...")

        # -------------------------------------------------
        # Irrelevant Retrieval Check
        # -------------------------------------------------
        if pipeline_result and pipeline_result.rerank:
            chunks = pipeline_result.rerank.reranked_chunks
            if not chunks:
                logger.warning("Failure Type: IRRELEVANT_RETRIEVAL (No chunks retrieved)")
                return FailureAnalysis(
                    failure_type=FailureType.IRRELEVANT_RETRIEVAL,
                    confidence=response.confidence,
                    should_retry=True,
                    reason="No chunks were retrieved.",
                )
            
            max_rerank_score = max(chunk.reranker_score for chunk in chunks)
            # Threshold logit: if max_rerank_score is extremely low (e.g. less than -3.0), retrieval is irrelevant
            if max_rerank_score < -3.0:
                logger.warning(f"Failure Type: IRRELEVANT_RETRIEVAL (max score {max_rerank_score:.2f})")
                return FailureAnalysis(
                    failure_type=FailureType.IRRELEVANT_RETRIEVAL,
                    confidence=response.confidence,
                    should_retry=True,
                    reason=f"Top reranker score {max_rerank_score:.2f} is below relevance threshold.",
                )

        # -------------------------------------------------
        # Ambiguous Query Check
        # -------------------------------------------------
        ambiguity_terms = ["ambiguity", "ambiguous", "unclear query", "multiple interpretation", "clarify", "conflicting interpretation"]
        lower_reason = response.reason.lower()
        if any(term in lower_reason for term in ambiguity_terms):
            logger.warning("Failure Type: AMBIGUOUS_QUERY")
            return FailureAnalysis(
                failure_type=FailureType.AMBIGUOUS_QUERY,
                confidence=response.confidence,
                should_retry=True,
                reason=response.reason,
            )

        for claim in response.unsupported_claims:
            if any(term in claim.reason.lower() for term in ambiguity_terms):
                logger.warning("Failure Type: AMBIGUOUS_QUERY")
                return FailureAnalysis(
                    failure_type=FailureType.AMBIGUOUS_QUERY,
                    confidence=response.confidence,
                    should_retry=True,
                    reason=f"Ambiguity detected in claim: '{claim.claim}'. Reason: {claim.reason}",
                )

        # -------------------------------------------------
        # Fully grounded
        # -------------------------------------------------

        if (
            response.is_grounded
            and len(response.unsupported_claims) == 0
        ):

            logger.success(
                "Failure Type: NONE"
            )

            return FailureAnalysis(
                failure_type=FailureType.NONE,
                confidence=response.confidence,
                should_retry=False,
                reason="Answer is fully grounded.",
            )

        # -------------------------------------------------
        # Partial grounding
        # -------------------------------------------------

        if (
            response.is_grounded
            and len(response.unsupported_claims) > 0
        ):

            logger.warning(
                "Failure Type: PARTIAL_GROUNDING"
            )

            return FailureAnalysis(
                failure_type=FailureType.PARTIAL_GROUNDING,
                confidence=response.confidence,
                should_retry=True,
                reason=(
                    f"{len(response.unsupported_claims)} "
                    "unsupported claims detected."
                ),
            )

        # -------------------------------------------------
        # Low confidence
        # -------------------------------------------------

        if response.confidence < confidence_threshold:

            logger.warning(
                "Failure Type: LOW_CONFIDENCE"
            )

            return FailureAnalysis(
                failure_type=FailureType.LOW_CONFIDENCE,
                confidence=response.confidence,
                should_retry=True,
                reason="Grounding confidence is low.",
            )

        # -------------------------------------------------
        # Insufficient context
        # -------------------------------------------------

        if response.should_retry:

            logger.warning(
                "Failure Type: INSUFFICIENT_CONTEXT"
            )

            return FailureAnalysis(
                failure_type=FailureType.INSUFFICIENT_CONTEXT,
                confidence=response.confidence,
                should_retry=True,
                reason=response.reason,
            )

        # -------------------------------------------------
        # Default
        # -------------------------------------------------

        logger.warning(
            "Failure Type: HALLUCINATION"
        )

        return FailureAnalysis(
            failure_type=FailureType.HALLUCINATION,
            confidence=response.confidence,
            should_retry=True,
            reason=response.reason,
        )
