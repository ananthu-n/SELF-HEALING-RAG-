from __future__ import annotations

from pydantic import BaseModel
from app.evaluation.failure_models import FailureType


class AdaptiveRetrievalProfile(BaseModel):
    failure_type: FailureType
    dense_top_k: int
    bm25_top_k: int
    final_top_k: int
    score_threshold: float
    description: str


class AdaptiveProfileRegistry:
    """
    Failure-driven adaptive retrieval profiles.

    Customizes retrieval parameters based on the root cause of failure:
    - LOW_CONFIDENCE -> Heavy dense retrieval emphasis.
    - PARTIAL_GROUNDING -> Expanded evidence context.
    - IRRELEVANT_RETRIEVAL -> Triggers query rewrite & arXiv search.
    - INSUFFICIENT_CONTEXT -> Broad multi-index search.
    - AMBIGUOUS_QUERY -> Disambiguated query expansion.
    """

    PROFILES: dict[FailureType, AdaptiveRetrievalProfile] = {
        FailureType.LOW_CONFIDENCE: AdaptiveRetrievalProfile(
            failure_type=FailureType.LOW_CONFIDENCE,
            dense_top_k=50,
            bm25_top_k=30,
            final_top_k=25,
            score_threshold=-1.0,
            description="Emphasize deep semantic vector search for low confidence.",
        ),
        FailureType.PARTIAL_GROUNDING: AdaptiveRetrievalProfile(
            failure_type=FailureType.PARTIAL_GROUNDING,
            dense_top_k=40,
            bm25_top_k=30,
            final_top_k=20,
            score_threshold=-0.5,
            description="Expand evidence window to capture missing grounding claims.",
        ),
        FailureType.IRRELEVANT_RETRIEVAL: AdaptiveRetrievalProfile(
            failure_type=FailureType.IRRELEVANT_RETRIEVAL,
            dense_top_k=30,
            bm25_top_k=30,
            final_top_k=15,
            score_threshold=0.0,
            description="Trigger search query rewriting and arXiv external acquisition.",
        ),
        FailureType.INSUFFICIENT_CONTEXT: AdaptiveRetrievalProfile(
            failure_type=FailureType.INSUFFICIENT_CONTEXT,
            dense_top_k=50,
            bm25_top_k=50,
            final_top_k=30,
            score_threshold=-2.0,
            description="Maximize dense and BM25 recall breadth for missing context.",
        ),
        FailureType.AMBIGUOUS_QUERY: AdaptiveRetrievalProfile(
            failure_type=FailureType.AMBIGUOUS_QUERY,
            dense_top_k=35,
            bm25_top_k=35,
            final_top_k=15,
            score_threshold=0.0,
            description="Disambiguate acronyms and entity terms.",
        ),
    }

    @classmethod
    def get_profile(cls, failure_type: FailureType) -> AdaptiveRetrievalProfile:
        return cls.PROFILES.get(
            failure_type,
            AdaptiveRetrievalProfile(
                failure_type=failure_type,
                dense_top_k=20,
                bm25_top_k=20,
                final_top_k=10,
                score_threshold=0.0,
                description="Default retrieval profile.",
            ),
        )
