from __future__ import annotations

import time
import math
from statistics import mean
from collections import Counter
from app.core.logger import logger
from app.self_healing.controller import SelfHealingController
from app.evaluation.failure_models import FailureType
from app.self_healing.healing_models import RetrievalStrategy

def run_benchmark() -> None:
    """
    Runs a research-grade benchmark evaluation on the Self-Healing RAG system,
    computing Retrieval, RAG/Generation, Self-Healing, and Performance metrics.
    """
    logger.info("=" * 80)
    logger.info("STARTING SELF-HEALING RAG SYSTEM BENCHMARK WITH COMPREHENSIVE METRICS")
    logger.info("=" * 80)

    # Define benchmark queries and their ground-truth relevant paper IDs
    benchmark_data = [
        {
            "query": "What is Self-RAG?",
            "ground_truth": ["2509.20377", "2408.05933", "2512.10787"]
        },
        {
            "query": "How does Corrective RAG (CRAG) handle low-quality retrieval?",
            "ground_truth": ["2401.15884", "2512.10787"]
        },
        {
            "query": "What is the main advantage of GraphRAG over standard RAG?",
            "ground_truth": ["2411.05844"]
        },
        {
            "query": "Explain the function of critique tokens in Self-RAG.",
            "ground_truth": ["2509.20377", "2512.10787"]
        },
        {
            "query": "How does Active Retrieval-Augmented Generation decide when to retrieve?",
            "ground_truth": ["2512.10787", "2509.20377"]
        }
    ]

    controller = SelfHealingController()
    query_results = []
    
    # Global accumulator for self-healing metrics
    all_failures_encountered = []
    strategy_attempts = Counter()
    strategy_successes = Counter()

    for idx, item in enumerate(benchmark_data, start=1):
        query = item["query"]
        gt = item["ground_truth"]
        
        logger.info("\n" + "-" * 80)
        logger.info(f"BENCHMARK QUERY {idx}/{len(benchmark_data)}: '{query}'")
        logger.info(f"Ground Truth Papers: {gt}")
        logger.info("-" * 80)

        start_time = time.perf_counter()
        state = controller.answer(query)
        elapsed = time.perf_counter() - start_time

        # We must have at least one attempt in attempts_history
        if not state.attempts_history:
            logger.error("No attempts recorded in state.attempts_history!")
            continue

        initial_attempt = state.attempts_history[0]
        final_attempt = state.attempts_history[-1]

        # ---------------------------------------------------------
        # Helper: Extract unique retrieved papers preserving rank
        # ---------------------------------------------------------
        def get_retrieved_papers(attempt) -> list[str]:
            if not attempt.retrieval_result or not attempt.retrieval_result.retrieved_chunks:
                return []
            seen = []
            for chunk in attempt.retrieval_result.retrieved_chunks:
                pid = chunk.paper_id.split("v")[0]
                if pid not in seen:
                    seen.append(pid)
            return seen

        init_papers = get_retrieved_papers(initial_attempt)
        final_papers = get_retrieved_papers(final_attempt)

        # ---------------------------------------------------------
        # Retrieval Metrics Calculations (Recall, Precision, MRR, nDCG, Hit Rate)
        # ---------------------------------------------------------
        def calculate_retrieval_metrics(retrieved: list[str], ground_truth: list[str], k: int = 5) -> dict:
            ret_k = retrieved[:k]
            hits = [p for p in ret_k if p in ground_truth]
            
            recall = len(hits) / len(ground_truth) if len(ground_truth) > 0 else 0.0
            precision = len(hits) / k if k > 0 else 0.0
            hit_rate = 1.0 if len(hits) > 0 else 0.0
            
            # MRR
            mrr = 0.0
            for rank_idx, p in enumerate(retrieved, start=1):
                if p in ground_truth:
                    mrr = 1.0 / rank_idx
                    break
            
            # nDCG@K
            dcg = 0.0
            for r_idx, p in enumerate(ret_k, start=1):
                rel = 1.0 if p in ground_truth else 0.0
                dcg += rel / math.log2(r_idx + 1)
                
            idcg = 0.0
            for r_idx in range(1, min(k, len(ground_truth)) + 1):
                idcg += 1.0 / math.log2(r_idx + 1)
                
            ndcg = dcg / idcg if idcg > 0.0 else 0.0
            
            return {
                "recall": recall,
                "precision": precision,
                "hit_rate": hit_rate,
                "mrr": mrr,
                "ndcg": ndcg
            }

        init_ret_metrics = calculate_retrieval_metrics(init_papers, gt, k=5)
        final_ret_metrics = calculate_retrieval_metrics(final_papers, gt, k=5)

        # ---------------------------------------------------------
        # RAG / Generation Metrics (Faithfulness, Relevance, Citation Accuracy)
        # ---------------------------------------------------------
        def calculate_generation_metrics(attempt, retrieved_papers: list[str], ground_truth: list[str]) -> dict:
            if not attempt.grounding_result:
                return {"faithfulness": 0.0, "hallucination_rate": 1.0, "citation_accuracy": 0.0, "context_precision": 0.0, "context_recall": 0.0}
            
            g_resp = attempt.grounding_result.response
            # Faithfulness: grounded and no unsupported claims
            faithfulness = 1.0 if (g_resp.is_grounded and len(g_resp.unsupported_claims) == 0) else 0.0
            hallucination_rate = len(g_resp.unsupported_claims) / max(1, len(g_resp.unsupported_claims) + 5) if not g_resp.is_grounded else 0.0
            
            # Context Precision: retrieved papers that are relevant
            context_precision = len([p for p in retrieved_papers if p in ground_truth]) / max(1, len(retrieved_papers))
            # Context Recall: ground truth papers retrieved
            context_recall = len([p for p in ground_truth if p in retrieved_papers]) / len(ground_truth) if len(ground_truth) > 0 else 0.0
            
            # Citation Accuracy
            citations = attempt.generation_result.citations if attempt.generation_result else []
            valid_citations = 0
            for c in citations:
                c_pid = c.paper_id.split("v")[0]
                if c_pid in retrieved_papers:
                    valid_citations += 1
            citation_accuracy = valid_citations / len(citations) if len(citations) > 0 else 1.0
            
            return {
                "faithfulness": faithfulness,
                "hallucination_rate": hallucination_rate,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "citation_accuracy": citation_accuracy,
                "answer_relevance": g_resp.confidence  # use LLM confidence score as proxy
            }

        init_gen_metrics = calculate_generation_metrics(initial_attempt, init_papers, gt)
        final_gen_metrics = calculate_generation_metrics(final_attempt, final_papers, gt)

        # ---------------------------------------------------------
        # Track Strategy attempts and successes
        # ---------------------------------------------------------
        for plan_idx, plan in enumerate(state.healing_history):
            # Track failures encountered
            if plan.reason != "Initial execution":
                all_failures_encountered.append(plan.reason)
            
            # Identify strategy applied
            strat = "REWRITE" if plan.rewrite_query else "EXPAND_DEPTH"
            strategy_attempts[strat] += 1
            
            # If the state ended successfully, count final strategies as successful
            if state.last_failure and state.last_failure.failure_type == FailureType.NONE and plan_idx == len(state.healing_history) - 1:
                strategy_successes[strat] += 1

        # Save all results
        query_results.append({
            "idx": idx,
            "query": query,
            "retries": state.retry_count,
            "success": state.last_failure.failure_type == FailureType.NONE if state.last_failure else False,
            "initial_failure": initial_attempt.failure_analysis.failure_type if initial_attempt.failure_analysis else FailureType.NONE,
            "final_failure": final_attempt.failure_analysis.failure_type if final_attempt.failure_analysis else FailureType.NONE,
            "elapsed": elapsed,
            "init_ret": init_ret_metrics,
            "final_ret": final_ret_metrics,
            "init_gen": init_gen_metrics,
            "final_gen": final_gen_metrics
        })

    # ---------------------------------------------------------
    # Aggregate Metrics Calculations
    # ---------------------------------------------------------
    total_queries = len(query_results)
    successful_queries = sum(1 for q in query_results if q["success"])
    overall_accuracy = (successful_queries / total_queries) * 100.0 if total_queries > 0 else 0.0

    # Self-healing metrics
    initially_failed_queries = [q for q in query_results if q["initial_failure"] != FailureType.NONE]
    healed_queries = [q for q in initially_failed_queries if q["success"]]
    
    retry_success_rate = (len(healed_queries) / len(initially_failed_queries) * 100.0) if len(initially_failed_queries) > 0 else 100.0
    recovery_rate = retry_success_rate  # equivalent
    avg_retry_count = mean(q["retries"] for q in query_results)
    
    # Healing effectiveness (improvement in faithfulness)
    avg_init_faith = mean(q["init_gen"]["faithfulness"] for q in query_results)
    avg_final_faith = mean(q["final_gen"]["faithfulness"] for q in query_results)
    healing_effectiveness = avg_final_faith - avg_init_faith

    # Mean Average Precision (MAP)
    mean_ap_init = mean(q["init_ret"]["mrr"] for q in query_results)  # AP proxy
    mean_ap_final = mean(q["final_ret"]["mrr"] for q in query_results)

    # Latencies
    avg_latency = mean(q["elapsed"] for q in query_results)

    # Print Summary report
    print("\n" + "=" * 90)
    print("                    SELF-HEALING RAG SYSTEM BENCHMARK REPORT")
    print("=" * 90)
    print(f"Total Queries Evaluated     : {total_queries}")
    print(f"Overall Grounded Accuracy   : {overall_accuracy:.2f}% ({successful_queries}/{total_queries})")
    print(f"Average Loop Latency        : {avg_latency:.2f} seconds")
    print(f"Average Retry Count         : {avg_retry_count:.2f}")
    print("-" * 90)
    print("1. RETRIEVAL ACCURACY METRICS (Top-5)")
    print("-" * 90)
    print(f"Metric             | Initial Attempt | Final Attempt   | Improvement")
    print(f"Recall@5           | {mean([q['init_ret']['recall'] for q in query_results]):.4f}          | {mean([q['final_ret']['recall'] for q in query_results]):.4f}          | {mean([q['final_ret']['recall'] for q in query_results]) - mean([q['init_ret']['recall'] for q in query_results]):+.4f}")
    print(f"Precision@5        | {mean([q['init_ret']['precision'] for q in query_results]):.4f}          | {mean([q['final_ret']['precision'] for q in query_results]):.4f}          | {mean([q['final_ret']['precision'] for q in query_results]) - mean([q['init_ret']['precision'] for q in query_results]):+.4f}")
    print(f"Hit Rate@5         | {mean([q['init_ret']['hit_rate'] for q in query_results]):.4f}          | {mean([q['final_ret']['hit_rate'] for q in query_results]):.4f}          | {mean([q['final_ret']['hit_rate'] for q in query_results]) - mean([q['init_ret']['hit_rate'] for q in query_results]):+.4f}")
    print(f"MRR                | {mean([q['init_ret']['mrr'] for q in query_results]):.4f}          | {mean([q['final_ret']['mrr'] for q in query_results]):.4f}          | {mean([q['final_ret']['mrr'] for q in query_results]) - mean([q['init_ret']['mrr'] for q in query_results]):+.4f}")
    print(f"nDCG@5             | {mean([q['init_ret']['ndcg'] for q in query_results]):.4f}          | {mean([q['final_ret']['ndcg'] for q in query_results]):.4f}          | {mean([q['final_ret']['ndcg'] for q in query_results]) - mean([q['init_ret']['ndcg'] for q in query_results]):+.4f}")
    print(f"MAP                | {mean_ap_init:.4f}          | {mean_ap_final:.4f}          | {mean_ap_final - mean_ap_init:+.4f}")
    print("-" * 90)
    print("2. RAG / GENERATION QUALITY METRICS")
    print("-" * 90)
    print(f"Metric             | Initial Attempt | Final Attempt   | Improvement")
    print(f"Faithfulness       | {avg_init_faith:.4f}          | {avg_final_faith:.4f}          | {healing_effectiveness:+.4f}")
    print(f"Hallucination Rate | {mean([q['init_gen']['hallucination_rate'] for q in query_results]):.4f}          | {mean([q['final_gen']['hallucination_rate'] for q in query_results]):.4f}          | {mean([q['final_gen']['hallucination_rate'] for q in query_results]) - mean([q['init_gen']['hallucination_rate'] for q in query_results]):+.4f}")
    print(f"Context Precision  | {mean([q['init_gen']['context_precision'] for q in query_results]):.4f}          | {mean([q['final_gen']['context_precision'] for q in query_results]):.4f}          | {mean([q['final_gen']['context_precision'] for q in query_results]) - mean([q['init_gen']['context_precision'] for q in query_results]):+.4f}")
    print(f"Context Recall     | {mean([q['init_gen']['context_recall'] for q in query_results]):.4f}          | {mean([q['final_gen']['context_recall'] for q in query_results]):.4f}          | {mean([q['final_gen']['context_recall'] for q in query_results]) - mean([q['init_gen']['context_recall'] for q in query_results]):+.4f}")
    print(f"Citation Accuracy  | {mean([q['init_gen']['citation_accuracy'] for q in query_results]):.4f}          | {mean([q['final_gen']['citation_accuracy'] for q in query_results]):.4f}          | {mean([q['final_gen']['citation_accuracy'] for q in query_results]) - mean([q['init_gen']['citation_accuracy'] for q in query_results]):+.4f}")
    print(f"Answer Relevance   | {mean([q['init_gen']['answer_relevance'] for q in query_results]):.4f}          | {mean([q['final_gen']['answer_relevance'] for q in query_results]):.4f}          | {mean([q['final_gen']['answer_relevance'] for q in query_results]) - mean([q['init_gen']['answer_relevance'] for q in query_results]):+.4f}")
    print("-" * 90)
    print("3. SELF-HEALING EFFICIENCY METRICS")
    print("-" * 90)
    print(f"Initially Failed Queries   : {len(initially_failed_queries)}")
    print(f"Successfully Healed Queries: {len(healed_queries)}")
    print(f"Recovery Rate (Healing)    : {recovery_rate:.2f}%")
    print(f"Retry Success Rate         : {retry_success_rate:.2f}%")
    print(f"Healing Effectiveness      : {healing_effectiveness:+.4f}")
    print(f"Diagnosed Failure Types    : {dict(Counter(all_failures_encountered))}")
    print("Strategy Success Rates:")
    for strategy in strategy_attempts:
        att = strategy_attempts[strategy]
        succ = strategy_successes[strategy]
        rate = (succ / att * 100.0) if att > 0 else 0.0
        print(f"  - {strategy:<14} : {rate:.2f}% ({succ}/{att})")
    print("-" * 90)
    print("4. SYSTEM PERFORMANCE METRICS")
    print("-" * 90)
    print(f"Average E2E Latency        : {avg_latency:.2f} seconds")
    print("===========================================================================")

if __name__ == "__main__":
    run_benchmark()
