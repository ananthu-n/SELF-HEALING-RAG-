from __future__ import annotations

import math
from pydantic import BaseModel, Field
from app.core.logger import logger


class StageLatency(BaseModel):
    query_enhancement_sec: float = 0.0
    retrieval_sec: float = 0.0
    reranking_sec: float = 0.0
    generation_sec: float = 0.0
    evaluation_sec: float = 0.0
    total_sec: float = 0.0


class SystemEvaluationReport(BaseModel):
    total_queries: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    grounding_pass_rate: float
    intent_alignment_rate: float
    retry_success_rate: float
    average_retry_count: float
    average_stage_latency: StageLatency


class EvaluationMetricsCalculator:
    """
    Computes information retrieval and system performance metrics.
    """

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        top_k_retrieved = set(retrieved_ids[:k])
        hits = len(top_k_retrieved.intersection(relevant_ids))
        return hits / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        for idx, item_id in enumerate(retrieved_ids):
            if item_id in relevant_ids:
                return 1.0 / (idx + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        dcg = 0.0
        for i, item_id in enumerate(retrieved_ids[:k]):
            if item_id in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)

        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def grounding_pass_rate(grounding_flags: list[bool]) -> float:
        if not grounding_flags:
            return 0.0
        return sum(grounding_flags) / len(grounding_flags)

    @staticmethod
    def intent_alignment_rate(intent_flags: list[bool]) -> float:
        if not intent_flags:
            return 0.0
        return sum(intent_flags) / len(intent_flags)

    @staticmethod
    def retry_success_rate(retry_results: list[bool]) -> float:
        if not retry_results:
            return 0.0
        return sum(retry_results) / len(retry_results)

    @staticmethod
    def average_retry_count(retry_counts: list[int]) -> float:
        if not retry_counts:
            return 0.0
        return sum(retry_counts) / len(retry_counts)


class EvaluationReportGenerator:
    """
    Generates structured system evaluation reports.
    """

    @classmethod
    def generate_report(
        cls,
        total_queries: int,
        recalls: list[float],
        mrrs: list[float],
        ndcgs: list[float],
        grounding_flags: list[bool],
        intent_flags: list[bool],
        retry_successes: list[bool],
        retry_counts: list[int],
        latencies: list[StageLatency],
    ) -> SystemEvaluationReport:
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

        g_pass = EvaluationMetricsCalculator.grounding_pass_rate(grounding_flags)
        i_pass = EvaluationMetricsCalculator.intent_alignment_rate(intent_flags)
        r_success = EvaluationMetricsCalculator.retry_success_rate(retry_successes)
        avg_retries = EvaluationMetricsCalculator.average_retry_count(retry_counts)

        n = len(latencies) if latencies else 1
        avg_latency = StageLatency(
            query_enhancement_sec=sum(l.query_enhancement_sec for l in latencies) / n,
            retrieval_sec=sum(l.retrieval_sec for l in latencies) / n,
            reranking_sec=sum(l.reranking_sec for l in latencies) / n,
            generation_sec=sum(l.generation_sec for l in latencies) / n,
            evaluation_sec=sum(l.evaluation_sec for l in latencies) / n,
            total_sec=sum(l.total_sec for l in latencies) / n,
        )

        report = SystemEvaluationReport(
            total_queries=total_queries,
            recall_at_k=avg_recall,
            mrr=avg_mrr,
            ndcg_at_k=avg_ndcg,
            grounding_pass_rate=g_pass,
            intent_alignment_rate=i_pass,
            retry_success_rate=r_success,
            average_retry_count=avg_retries,
            average_stage_latency=avg_latency,
        )

        logger.info("=" * 60)
        logger.info("SYSTEM EVALUATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Queries Evaluated : {total_queries}")
        logger.info(f"Recall@K               : {avg_recall:.4f}")
        logger.info(f"MRR                    : {avg_mrr:.4f}")
        logger.info(f"nDCG@K                 : {avg_ndcg:.4f}")
        logger.info(f"Grounding Pass Rate    : {g_pass * 100:.2f}%")
        logger.info(f"Intent Alignment Rate  : {i_pass * 100:.2f}%")
        logger.info(f"Retry Success Rate     : {r_success * 100:.2f}%")
        logger.info(f"Average Retry Count    : {avg_retries:.2f}")
        logger.info(f"Average Total Latency  : {avg_latency.total_sec:.2f}s")
        logger.info("=" * 60)

        return report
