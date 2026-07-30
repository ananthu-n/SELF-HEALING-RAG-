from __future__ import annotations

from app.evaluation.decision import DecisionAction
from app.evaluation.decision_engine import DecisionEngine
from app.evaluation.failure_analyzer import FailureAnalyzer
from app.evaluation.failure_models import FailureAnalysis, FailureType
from app.evaluation.models import GroundingResponse, GroundingResult
from app.self_healing.healing_models import (
    DiversityStrategy,
    RetrievalStrategy,
    RewriteStrategy,
)
from app.self_healing.healing_planner import HealingPlanner
from app.self_healing.models import SelfHealingState
from app.self_healing.strategy_registry import (
    FAILURE_STRATEGY_REGISTRY,
    get_failure_strategy,
)


def assert_strategy(
    failure_type: FailureType,
    *,
    decision_action: DecisionAction,
    should_retry: bool,
    retrieval_strategy: RetrievalStrategy | None = None,
    rewrite_strategy: RewriteStrategy | None = None,
    diversity_strategy: DiversityStrategy | None = None,
    rewrite_query: bool = False,
    increase_top_k: bool = False,
) -> None:
    strategy = get_failure_strategy(failure_type)

    assert strategy.decision_action == decision_action
    assert strategy.should_retry == should_retry

    if retrieval_strategy is not None:
        assert strategy.retrieval_strategy == retrieval_strategy

    if rewrite_strategy is not None:
        assert strategy.rewrite_strategy == rewrite_strategy

    if diversity_strategy is not None:
        assert strategy.diversity_strategy == diversity_strategy

    assert strategy.rewrite_query == rewrite_query
    assert strategy.increase_top_k == increase_top_k


def test_registry_covers_all_failure_types() -> None:
    for failure_type in FailureType:
        assert failure_type in FAILURE_STRATEGY_REGISTRY


def test_partial_grounding_strategy() -> None:
    assert_strategy(
        FailureType.PARTIAL_GROUNDING,
        decision_action=DecisionAction.RETRY_RETRIEVAL,
        should_retry=True,
        retrieval_strategy=RetrievalStrategy.BALANCED,
        increase_top_k=True,
    )


def test_irrelevant_retrieval_strategy() -> None:
    assert_strategy(
        FailureType.IRRELEVANT_RETRIEVAL,
        decision_action=DecisionAction.REWRITE_QUERY,
        should_retry=True,
        retrieval_strategy=RetrievalStrategy.DIVERSIFY,
        rewrite_strategy=RewriteStrategy.EXPAND,
        diversity_strategy=DiversityStrategy.DIFFERENT_PAPERS,
        rewrite_query=True,
    )


def test_ambiguous_query_strategy() -> None:
    assert_strategy(
        FailureType.AMBIGUOUS_QUERY,
        decision_action=DecisionAction.REWRITE_QUERY,
        should_retry=True,
        rewrite_strategy=RewriteStrategy.DISAMBIGUATE,
        rewrite_query=True,
    )


def test_decision_engine_partial_grounding_retries() -> None:
    engine = DecisionEngine()

    decision = engine.decide(
        FailureAnalysis(
            failure_type=FailureType.PARTIAL_GROUNDING,
            confidence=0.85,
            should_retry=True,
            reason="1 unsupported claims detected.",
        )
    )

    assert decision.should_retry is True
    assert decision.action == DecisionAction.RETRY_RETRIEVAL


def test_healing_planner_partial_grounding_increases_top_k() -> None:
    planner = HealingPlanner()

    state = SelfHealingState(
        original_query="What is Self-RAG?",
        current_query="What is Self-RAG?",
        retry_count=0,
        last_failure=FailureAnalysis(
            failure_type=FailureType.PARTIAL_GROUNDING,
            confidence=0.85,
            should_retry=True,
            reason="1 unsupported claims detected.",
        ),
        last_decision=DecisionEngine().decide(
            FailureAnalysis(
                failure_type=FailureType.PARTIAL_GROUNDING,
                confidence=0.85,
                should_retry=True,
                reason="1 unsupported claims detected.",
            )
        ),
    )

    plan = planner.plan(state)

    assert plan.retry_number == 1
    assert plan.retrieval_strategy == RetrievalStrategy.BALANCED
    assert plan.increase_top_k is True
    assert plan.top_k == 20
    assert plan.dense_top_k == 40
    assert plan.bm25_top_k == 40


def test_failure_analyzer_uses_config_threshold() -> None:
    analyzer = FailureAnalyzer()

    result = analyzer.analyze(
        GroundingResult(
            response=GroundingResponse(
                is_grounded=False,
                confidence=0.75,
                unsupported_claims=[],
                should_retry=False,
                reason="Low confidence answer.",
            )
        )
    )

    assert result.failure_type == FailureType.LOW_CONFIDENCE


def main() -> None:
    test_registry_covers_all_failure_types()
    test_partial_grounding_strategy()
    test_irrelevant_retrieval_strategy()
    test_ambiguous_query_strategy()
    test_decision_engine_partial_grounding_retries()
    test_healing_planner_partial_grounding_increases_top_k()
    test_failure_analyzer_uses_config_threshold()

    print("All healing strategy registry tests passed.")


if __name__ == "__main__":
    main()
