import pytest
from app.evaluation.failure_models import FailureType
from app.retrieval.adaptive_profiles import AdaptiveProfileRegistry


def test_adaptive_profiles():
    low_conf_profile = AdaptiveProfileRegistry.get_profile(FailureType.LOW_CONFIDENCE)
    assert low_conf_profile.dense_top_k == 50
    assert low_conf_profile.final_top_k == 25

    partial_grounding_profile = AdaptiveProfileRegistry.get_profile(FailureType.PARTIAL_GROUNDING)
    assert partial_grounding_profile.dense_top_k == 40
    assert partial_grounding_profile.final_top_k == 20

    insufficient_context_profile = AdaptiveProfileRegistry.get_profile(FailureType.INSUFFICIENT_CONTEXT)
    assert insufficient_context_profile.dense_top_k == 50
    assert insufficient_context_profile.bm25_top_k == 50
    assert insufficient_context_profile.final_top_k == 30
