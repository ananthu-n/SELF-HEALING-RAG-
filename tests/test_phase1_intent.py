import pytest
from app.pipeline.intent import QueryIntent, IntentDetector
from app.evaluation.intent_alignment_evaluator import IntentAlignmentEvaluator


def test_intent_detection():
    assert IntentDetector.detect("What is embedding?") == QueryIntent.DEFINITION
    assert IntentDetector.detect("Compare GraphRAG vs Self-RAG") == QueryIntent.COMPARISON
    assert IntentDetector.detect("How does CrossEncoder reranking work?") == QueryIntent.EXPLANATION
    assert IntentDetector.detect("Literature review of sparse retrieval") == QueryIntent.LITERATURE_REVIEW


def test_hyde_gating():
    assert not IntentDetector.should_use_hyde(QueryIntent.DEFINITION)
    assert not IntentDetector.should_use_hyde(QueryIntent.COMPARISON)
    assert IntentDetector.should_use_hyde(QueryIntent.EXPLANATION)
    assert IntentDetector.should_use_hyde(QueryIntent.RESEARCH_QUESTION)


def test_intent_alignment_evaluator():
    def_res = IntentAlignmentEvaluator.evaluate(
        query="What is embedding?",
        answer="An embedding is a dense vector representation of text.",
        intent=QueryIntent.DEFINITION,
    )
    assert def_res.is_aligned

    comp_res = IntentAlignmentEvaluator.evaluate(
        query="Compare A and B",
        answer="System A uses sparse search, whereas System B uses dense vectors.",
        intent=QueryIntent.COMPARISON,
    )
    assert comp_res.is_aligned
