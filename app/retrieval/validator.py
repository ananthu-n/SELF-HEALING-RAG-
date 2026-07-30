from __future__ import annotations

from statistics import mean

from app.core.logger import logger
from app.retrieval.models import RetrievalResult
from app.retrieval.validation import RetrievalValidation


class RetrievalValidator:
    """
    Validates retrieval quality before
    reranking and generation.
    """

    MIN_CHUNKS = 3

    MIN_AVERAGE_SCORE = 0.60

    MIN_MAX_SCORE = 0.70

    def validate(
        self,
        result: RetrievalResult,
    ) -> RetrievalValidation:

        chunks = result.retrieved_chunks

        if not chunks:

            logger.warning("No retrieved chunks.")

            return RetrievalValidation(
                is_valid=False,
                should_retry=True,
                reason="No chunks retrieved.",
                average_score=0.0,
                maximum_score=0.0,
                minimum_score=0.0,
                total_chunks=0,
                unique_papers=0,
            )

        scores = [chunk.score for chunk in chunks]

        avg_score = mean(scores)

        max_score = max(scores)

        min_score = min(scores)

        unique_papers = len(
            {chunk.paper_id for chunk in chunks}
        )

        valid = (
            len(chunks) >= self.MIN_CHUNKS
            and avg_score >= self.MIN_AVERAGE_SCORE
            and max_score >= self.MIN_MAX_SCORE
        )

        validation = RetrievalValidation(
            is_valid=valid,
            should_retry=not valid,
            reason=(
                "Retrieval quality acceptable."
                if valid
                else "Retrieval quality below threshold."
            ),
            average_score=avg_score,
            maximum_score=max_score,
            minimum_score=min_score,
            total_chunks=len(chunks),
            unique_papers=unique_papers,
        )

        logger.info(validation.model_dump())

        return validation