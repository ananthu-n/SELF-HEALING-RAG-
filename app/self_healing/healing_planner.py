from __future__ import annotations

from app.core.logger import logger
from app.evaluation.decision import DecisionAction
from app.evaluation.failure_models import FailureType
from app.retrieval.adaptive_profiles import AdaptiveProfileRegistry
from app.self_healing.healing_models import (
    HealingPlan,
    RetrievalStrategy,
    RewriteStrategy,
    DiversityStrategy,
)
from app.self_healing.models import SelfHealingState
from app.self_healing.strategy_registry import get_failure_strategy


class HealingPlanner:
    """
    Generates the healing strategy for the next retry based on failure modalities.
    """

    MAX_TOP_K = 40
    BASE_TOP_K = 10

    def plan(self, state: SelfHealingState) -> HealingPlan:
        logger.info("Planning healing strategy...")

        failure = state.last_failure
        decision = state.last_decision

        if failure is None or decision is None:
            logger.info("Initial execution.")
            return HealingPlan(
                query=state.current_query,
                retry_number=0,
                reason="Initial execution",
            )

        retry = state.retry_count + 1

        # Fetch failure-driven adaptive profile
        profile = AdaptiveProfileRegistry.get_profile(failure.failure_type)
        top_k = profile.final_top_k
        dense_top_k = profile.dense_top_k
        bm25_top_k = profile.bm25_top_k
        logger.info(f"Adaptive Profile Selected for '{failure.failure_type.value}': {profile.description}")

        strategy = get_failure_strategy(failure.failure_type)

        retrieval_strategy = strategy.retrieval_strategy
        rewrite_strategy = strategy.rewrite_strategy
        diversity_strategy = strategy.diversity_strategy
        rewrite_query = strategy.rewrite_query
        increase_top_k = strategy.increase_top_k

        top_k += strategy.top_k_delta
        dense_top_k += strategy.dense_top_k_delta
        bm25_top_k += strategy.bm25_top_k_delta

        reason = failure.reason

        # Signal A: Unsupported Claims Concept Targeting
        if state.last_grounding and state.last_grounding.response.unsupported_claims:
            unsupported_texts = [c.claim for c in state.last_grounding.response.unsupported_claims]
            reason = f"{failure.reason} Target missing concepts: {'; '.join(unsupported_texts)}"
            rewrite_query = True
            rewrite_strategy = RewriteStrategy.EXPAND

        # Signal B: Low Source Diversity Analysis
        if state.retrieval_result and state.retrieval_result.retrieved_chunks:
            unique_papers = len({c.paper_id for c in state.retrieval_result.retrieved_chunks})
            if unique_papers < 3:
                logger.warning(f"Multi-Signal Analysis: Low paper diversity detected ({unique_papers} papers). Activating paper diversification.")
                diversity_strategy = DiversityStrategy.DIFFERENT_PAPERS
                bm25_top_k += 15

        # Signal C: Low Max Rerank Score Analysis -> Trigger Live arXiv Knowledge Acquisition
        if state.rerank_result and state.rerank_result.reranked_chunks:
            max_score = max(c.reranker_score for c in state.rerank_result.reranked_chunks)
            if max_score < -2.0:
                logger.warning(f"Multi-Signal Analysis: Low rerank score ({max_score:.2f}). Triggering dynamic arXiv paper acquisition.")
                retrieval_strategy = RetrievalStrategy.SEARCH_ARXIV
                rewrite_query = True
                rewrite_strategy = RewriteStrategy.EXPAND

        if failure.failure_type in (FailureType.IRRELEVANT_RETRIEVAL, FailureType.INSUFFICIENT_CONTEXT):
            retrieval_strategy = RetrievalStrategy.SEARCH_ARXIV

        if decision.action == DecisionAction.REWRITE_QUERY:
            rewrite_query = True

        if decision.action == DecisionAction.RETRY_RETRIEVAL:
            increase_top_k = True

        already_rewrote = any(plan.rewrite_query for plan in state.healing_history)
        already_increased = any(plan.increase_top_k for plan in state.healing_history)

        if rewrite_query and already_rewrote:
            logger.info("Healing Memory: Query rewrite was already attempted. Falling back to expanding retrieval depth.")
            rewrite_query = False
            increase_top_k = True
            top_k += 10
            dense_top_k += 20
            bm25_top_k += 20
        elif increase_top_k and already_increased and not rewrite_query:
            logger.info("Healing Memory: Retrieval expansion already attempted. Forcing query rewrite.")
            rewrite_query = True

        return HealingPlan(
            query=state.current_query,
            retry_number=retry,
            reason=reason,
            rewrite_query=rewrite_query,
            increase_top_k=increase_top_k,
            top_k=min(top_k, self.MAX_TOP_K),
            dense_top_k=min(dense_top_k, self.MAX_TOP_K * 2),
            bm25_top_k=min(bm25_top_k, self.MAX_TOP_K * 2),
            retrieval_strategy=retrieval_strategy,
            rewrite_strategy=rewrite_strategy,
            diversity_strategy=diversity_strategy,
        )
