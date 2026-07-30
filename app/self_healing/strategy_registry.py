from __future__ import annotations

from pydantic import BaseModel, Field

from app.evaluation.decision import DecisionAction
from app.evaluation.failure_models import FailureType

from app.self_healing.healing_models import (
    DiversityStrategy,
    RetrievalStrategy,
    RewriteStrategy,
)


class FailureStrategyConfig(BaseModel):
    """
    Adaptive healing configuration for a single failure type.

    This is the single source of truth for:
    - what the DecisionEngine should do next
    - what the HealingPlanner should change on retry
    """

    failure_type: FailureType

    decision_action: DecisionAction

    should_retry: bool

    decision_reason: str

    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.DEFAULT

    rewrite_strategy: RewriteStrategy = RewriteStrategy.NONE

    diversity_strategy: DiversityStrategy = DiversityStrategy.NONE

    rewrite_query: bool = False

    increase_top_k: bool = False

    top_k_delta: int = Field(default=0, ge=0)

    dense_top_k_delta: int = Field(default=0, ge=0)

    bm25_top_k_delta: int = Field(default=0, ge=0)


FAILURE_STRATEGY_REGISTRY: dict[FailureType, FailureStrategyConfig] = {
    FailureType.NONE: FailureStrategyConfig(
        failure_type=FailureType.NONE,
        decision_action=DecisionAction.RETURN_ANSWER,
        should_retry=False,
        decision_reason="Answer is fully grounded.",
    ),
    FailureType.PARTIAL_GROUNDING: FailureStrategyConfig(
        failure_type=FailureType.PARTIAL_GROUNDING,
        decision_action=DecisionAction.REWRITE_QUERY,
        should_retry=True,
        decision_reason="Rewrite query targeting unsupported claims and increase retrieval depth.",
        retrieval_strategy=RetrievalStrategy.BALANCED,
        rewrite_strategy=RewriteStrategy.EXPAND,
        rewrite_query=True,
        increase_top_k=True,
        top_k_delta=5,
        dense_top_k_delta=10,
        bm25_top_k_delta=10,
    ),
    FailureType.LOW_CONFIDENCE: FailureStrategyConfig(
        failure_type=FailureType.LOW_CONFIDENCE,
        decision_action=DecisionAction.REWRITE_QUERY,
        should_retry=True,
        decision_reason="Rewrite query and broaden dense retrieval for low-confidence grounding.",
        retrieval_strategy=RetrievalStrategy.INCREASE_DENSE,
        rewrite_strategy=RewriteStrategy.EXPAND,
        rewrite_query=True,
        increase_top_k=True,
        dense_top_k_delta=10,
    ),
    FailureType.INSUFFICIENT_CONTEXT: FailureStrategyConfig(
        failure_type=FailureType.INSUFFICIENT_CONTEXT,
        decision_action=DecisionAction.RETRY_RETRIEVAL,
        should_retry=True,
        decision_reason="Expand retrieval depth to gather more evidence.",
        retrieval_strategy=RetrievalStrategy.EXPAND_RETRIEVAL,
        increase_top_k=True,
        top_k_delta=10,
        dense_top_k_delta=20,
        bm25_top_k_delta=20,
    ),
    FailureType.IRRELEVANT_RETRIEVAL: FailureStrategyConfig(
        failure_type=FailureType.IRRELEVANT_RETRIEVAL,
        decision_action=DecisionAction.REWRITE_QUERY,
        should_retry=True,
        decision_reason="Rewrite query and increase lexical retrieval depth.",
        retrieval_strategy=RetrievalStrategy.INCREASE_BM25,
        rewrite_strategy=RewriteStrategy.EXPAND,
        diversity_strategy=DiversityStrategy.DIFFERENT_PAPERS,
        rewrite_query=True,
        increase_top_k=True,
        top_k_delta=5,
        bm25_top_k_delta=15,
    ),
    FailureType.AMBIGUOUS_QUERY: FailureStrategyConfig(
        failure_type=FailureType.AMBIGUOUS_QUERY,
        decision_action=DecisionAction.REWRITE_QUERY,
        should_retry=True,
        decision_reason="Disambiguate the user query before retrying.",
        rewrite_strategy=RewriteStrategy.DISAMBIGUATE,
        rewrite_query=True,
    ),
    FailureType.HALLUCINATION: FailureStrategyConfig(
        failure_type=FailureType.HALLUCINATION,
        decision_action=DecisionAction.REWRITE_QUERY,
        should_retry=True,
        decision_reason="Rewrite query with scientific precision and diversify evidence.",
        retrieval_strategy=RetrievalStrategy.DIVERSIFY,
        rewrite_strategy=RewriteStrategy.SCIENTIFIC,
        diversity_strategy=DiversityStrategy.HYBRID,
        rewrite_query=True,
    ),
}


def get_failure_strategy(
    failure_type: FailureType,
) -> FailureStrategyConfig:
    """
    Return the adaptive strategy for a failure type.

    Raises KeyError if the failure type is not registered.
    """

    if failure_type not in FAILURE_STRATEGY_REGISTRY:
        raise KeyError(
            f"No adaptive strategy registered for {failure_type.value}"
        )

    return FAILURE_STRATEGY_REGISTRY[failure_type]
