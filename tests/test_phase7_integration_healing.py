import pytest
from app.evaluation.failure_models import FailureType, FailureAnalysis
from app.evaluation.decision_engine import DecisionEngine
from app.self_healing.healing_planner import HealingPlanner
from app.self_healing.models import SelfHealingState
from app.self_healing.healing_models import RetrievalStrategy, RewriteStrategy


def test_integration_low_confidence_healing():
    state = SelfHealingState(original_query="Low confidence test query", current_query="Low confidence test query")
    state.last_failure = FailureAnalysis(
        failure_type=FailureType.LOW_CONFIDENCE,
        confidence=0.3,
        should_retry=True,
        reason="Retrieval and grounding confidence below threshold.",
    )
    engine = DecisionEngine()
    state.last_decision = engine.decide(state.last_failure)

    planner = HealingPlanner()
    plan = planner.plan(state)

    assert plan.dense_top_k >= 40
    assert plan.top_k >= 20


def test_integration_partial_grounding_healing():
    state = SelfHealingState(original_query="Partial grounding test", current_query="Partial grounding test")
    state.last_failure = FailureAnalysis(
        failure_type=FailureType.PARTIAL_GROUNDING,
        confidence=0.5,
        should_retry=True,
        reason="Answer contains unsupported claims.",
    )
    engine = DecisionEngine()
    state.last_decision = engine.decide(state.last_failure)

    planner = HealingPlanner()
    plan = planner.plan(state)

    assert plan.rewrite_query
    assert plan.rewrite_strategy == RewriteStrategy.EXPAND


def test_integration_irrelevant_retrieval_healing():
    state = SelfHealingState(original_query="Irrelevant retrieval test", current_query="Irrelevant retrieval test")
    state.last_failure = FailureAnalysis(
        failure_type=FailureType.IRRELEVANT_RETRIEVAL,
        confidence=0.2,
        should_retry=True,
        reason="No relevant chunks found.",
    )
    engine = DecisionEngine()
    state.last_decision = engine.decide(state.last_failure)

    planner = HealingPlanner()
    plan = planner.plan(state)

    assert plan.retrieval_strategy == RetrievalStrategy.SEARCH_ARXIV
    assert plan.rewrite_query


def test_integration_insufficient_context_healing():
    state = SelfHealingState(original_query="Insufficient context test", current_query="Insufficient context test")
    state.last_failure = FailureAnalysis(
        failure_type=FailureType.INSUFFICIENT_CONTEXT,
        confidence=0.2,
        should_retry=True,
        reason="Context length below threshold.",
    )
    engine = DecisionEngine()
    state.last_decision = engine.decide(state.last_failure)

    planner = HealingPlanner()
    plan = planner.plan(state)

    assert plan.retrieval_strategy == RetrievalStrategy.SEARCH_ARXIV
    assert plan.dense_top_k >= 50
    assert plan.bm25_top_k >= 50


def test_integration_ambiguous_query_healing():
    state = SelfHealingState(original_query="Ambiguous acronym query", current_query="Ambiguous acronym query")
    state.last_failure = FailureAnalysis(
        failure_type=FailureType.AMBIGUOUS_QUERY,
        confidence=0.4,
        should_retry=True,
        reason="Query terms are ambiguous.",
    )
    engine = DecisionEngine()
    state.last_decision = engine.decide(state.last_failure)

    planner = HealingPlanner()
    plan = planner.plan(state)

    assert plan.rewrite_query
