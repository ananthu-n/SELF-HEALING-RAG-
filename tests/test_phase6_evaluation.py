import pytest
from app.evaluation.metrics import (
    EvaluationMetricsCalculator,
    EvaluationReportGenerator,
    StageLatency,
)


def test_retrieval_metrics():
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
    relevant = {"doc_2", "doc_5"}

    # Recall@4: doc_2 hit out of {doc_2, doc_5} = 0.5
    recall = EvaluationMetricsCalculator.recall_at_k(retrieved, relevant, k=4)
    assert recall == 0.5

    # MRR: first hit doc_2 at rank 2 = 1/2 = 0.5
    mrr = EvaluationMetricsCalculator.mrr(retrieved, relevant)
    assert mrr == 0.5

    # nDCG@4
    ndcg = EvaluationMetricsCalculator.ndcg_at_k(retrieved, relevant, k=4)
    assert 0.0 < ndcg <= 1.0


def test_system_report_generation():
    lat = StageLatency(
        query_enhancement_sec=0.1,
        retrieval_sec=0.5,
        reranking_sec=0.3,
        generation_sec=2.0,
        evaluation_sec=0.2,
        total_sec=3.1,
    )

    report = EvaluationReportGenerator.generate_report(
        total_queries=5,
        recalls=[0.5, 1.0, 0.8],
        mrrs=[0.5, 1.0, 1.0],
        ndcgs=[0.6, 1.0, 0.9],
        grounding_flags=[True, True, True, False, True],
        intent_flags=[True, True, True, True, True],
        retry_successes=[True, True],
        retry_counts=[1, 0, 1, 2, 0],
        latencies=[lat],
    )

    assert report.total_queries == 5
    assert report.grounding_pass_rate == 0.8
    assert report.intent_alignment_rate == 1.0
    assert report.retry_success_rate == 1.0
