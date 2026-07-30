from __future__ import annotations

from typing import Sequence
from app.core.logger import logger
from app.ingestion.topic_extractor import TopicExtractor
from app.reranker.models import RerankedChunk


class KnowledgeCoverageEstimator:
    """
    Estimates whether the retrieved reranked chunks contain sufficient coverage
    to answer the user's research query before invoking LLM generation.
    """

    @classmethod
    def estimate_coverage(
        self,
        query: str,
        reranked_chunks: Sequence[RerankedChunk],
    ) -> dict[str, float | bool | str]:
        """
        Estimate coverage based on keyword overlap and cross-encoder score bounds.
        """
        if not reranked_chunks:
            logger.warning("Coverage Estimator: No reranked chunks available.")
            return {
                "has_sufficient_coverage": False,
                "coverage_score": 0.0,
                "max_rerank_score": -99.0,
                "reason": "No reranked chunks retrieved.",
            }

        topic = TopicExtractor.extract_topic(query)
        topic_words = set(topic.lower().split())

        combined_text = " ".join([c.text.lower() for c in reranked_chunks])
        matched_words = {word for word in topic_words if word in combined_text}

        coverage_score = len(matched_words) / max(1, len(topic_words))
        max_score = max(c.reranker_score for c in reranked_chunks)

        # Sufficient if at least 50% topic terms matched AND max rerank score > -2.0
        sufficient = (coverage_score >= 0.50) and (max_score > -2.0)

        reason = (
            f"Coverage acceptable ({coverage_score*100:.0f}%, max score {max_score:.2f})."
            if sufficient
            else f"Low knowledge coverage ({coverage_score*100:.0f}%, max score {max_score:.2f}). Dynamic arXiv acquisition recommended."
        )

        logger.info(f"Knowledge Coverage Estimate: {reason}")

        return {
            "has_sufficient_coverage": sufficient,
            "coverage_score": coverage_score,
            "max_rerank_score": max_score,
            "reason": reason,
        }
