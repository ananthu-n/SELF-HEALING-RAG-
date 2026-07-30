from __future__ import annotations

from app.core.config import settings
from app.core.logger import logger

from app.pipeline.rag_pipeline import RAGPipeline
from app.pipeline.query_enhancement import QueryEnhancer
from app.evaluation.evaluator import GroundingEvaluator
from app.evaluation.crag_validator import CRAGValidator, CRAGState
from app.evaluation.decision_engine import DecisionEngine
from app.evaluation.models import GroundingRequest

from app.ingestion.dynamic_acquisition import DynamicKnowledgeAcquisitor
from app.self_healing.healing_models import RetrievalStrategy
from app.self_healing.models import SelfHealingState
from app.self_healing.query_rewriter import QueryRewriter
from app.self_healing.healing_planner import HealingPlanner
from app.evaluation.failure_analyzer import FailureAnalyzer
from app.retrieval.coverage_estimator import KnowledgeCoverageEstimator


class SelfHealingController:
    """
    Production Executive Orchestrator.
    Executes Query Enhancement -> Knowledge Coverage Check -> CRAG Validation -> Multi-Signal Healing Loop.
    Supports Scope Selection: "hybrid", "custom_only", "arxiv_only".
    """

    def __init__(self):
        self.pipeline = RAGPipeline()
        self.query_enhancer = QueryEnhancer(enable_hyde=True, enable_decomposition=True)
        self.evaluator = GroundingEvaluator()
        self.crag_validator = CRAGValidator()
        self.decision_engine = DecisionEngine()
        self.healing_planner = HealingPlanner()
        self.query_rewriter = QueryRewriter()
        self.failure_analyzer = FailureAnalyzer()
        self.dynamic_acquisitor = DynamicKnowledgeAcquisitor(max_results=10, top_k_download=3)

    def answer(self, query: str, search_scope: str = "hybrid") -> SelfHealingState:
        logger.info("=" * 80)
        logger.info(f"Starting Self-Healing Research Assistant Pipeline (Scope: {search_scope})")
        logger.info("=" * 80)

        # Step 1: Query Enhancement Layer (Topic Extraction, HyDE, Decomposition)
        enhanced_query = self.query_enhancer.enhance(query)
        logger.info(f"Topic Extracted: '{enhanced_query.extracted_topic}'")

        state = SelfHealingState(
            original_query=query,
            current_query=enhanced_query.search_query,
        )

        while state.retry_count < settings.self_healing.max_retries:
            logger.info(f"Self-Healing Execution Cycle [{state.retry_count + 1}/{settings.self_healing.max_retries}]")

            plan = self.healing_planner.plan(state)
            logger.info("=" * 60)
            logger.info(f"Healing Plan Strategy: {plan.retrieval_strategy.value} | Reason: {plan.reason}")
            logger.info("=" * 60)

            state.last_healing_plan = plan
            state.healing_history.append(plan)

            # Strategy Action A: Search arXiv for missing knowledge (only if scope permits arXiv)
            if search_scope != "custom_only" and (plan.retrieval_strategy == RetrievalStrategy.SEARCH_ARXIV or (state.retry_count > 0 and not state.retrieval_result)):
                logger.info("Executing Dynamic Knowledge Acquisition on arXiv...")
                try:
                    acquired_count = self.dynamic_acquisitor.acquire(enhanced_query.extracted_topic)
                    logger.info(f"Dynamic Acquisition indexed {acquired_count} fresh research paper(s).")
                except Exception as err:
                    logger.error(f"Dynamic Acquisition failed: {err}")

            # Strategy Action B: Search Query Keyword Expansion
            if plan.rewrite_query:
                search_terms = self.query_rewriter.rewrite(
                    query=plan.query,
                    reason=plan.reason,
                    rewrite_strategy=plan.rewrite_strategy,
                )
                state.current_query = search_terms
                plan.query = search_terms
                logger.info(f"Search Query Terms: {search_terms}")

            # Execute RAG pass (Hybrid Retrieval + Reranking + Generation)
            pipeline_result = self.pipeline.answer(
                query=plan.query,
                top_k=plan.top_k,
                dense_top_k=plan.dense_top_k,
                bm25_top_k=plan.bm25_top_k,
                search_scope=search_scope,
            )

            state.retrieval_result = pipeline_result.retrieval
            state.rerank_result = pipeline_result.rerank
            state.context_result = pipeline_result.context
            state.prompt_result = pipeline_result.prompt
            state.last_generation = pipeline_result.generation

            # CRAG Validation Layer
            crag_res = self.crag_validator.validate(
                query=plan.query,
                reranked_chunks=pipeline_result.rerank.reranked_chunks,
            )
            if crag_res.requires_external_search and search_scope != "custom_only":
                logger.warning(f"CRAG State {crag_res.state.value.upper()}: Triggering dynamic arXiv search for '{enhanced_query.extracted_topic}'...")
                try:
                    acquired = self.dynamic_acquisitor.acquire(enhanced_query.extracted_topic)
                    logger.info(f"CRAG Acquisition indexed {acquired} fresh paper(s).")
                    if acquired > 0:
                        logger.info("Re-running RAG pipeline with freshly acquired evidence...")
                        pipeline_result = self.pipeline.answer(
                            query=plan.query,
                            top_k=plan.top_k,
                            dense_top_k=plan.dense_top_k,
                            bm25_top_k=plan.bm25_top_k,
                            search_scope=search_scope,
                        )
                        state.retrieval_result = pipeline_result.retrieval
                        state.rerank_result = pipeline_result.rerank
                        state.context_result = pipeline_result.context
                        state.prompt_result = pipeline_result.prompt
                        state.last_generation = pipeline_result.generation
                except Exception as err:
                    logger.error(f"CRAG Acquisition failed: {err}")

            # Grounding Evaluation Layer
            grounding = self.evaluator.evaluate(
                GroundingRequest(
                    query=plan.query,
                    answer=pipeline_result.generation.response.answer,
                    context=pipeline_result.prompt.prompt.context,
                )
            )
            state.last_grounding = grounding

            failure = self.failure_analyzer.analyze(
                grounding=grounding,
                pipeline_result=pipeline_result,
            )
            state.last_failure = failure

            decision = self.decision_engine.decide(failure)
            state.last_decision = decision

            from app.self_healing.models import AttemptResult
            state.attempts_history.append(
                AttemptResult(
                    attempt_number=state.retry_count + 1,
                    query=plan.query,
                    retrieval_result=pipeline_result.retrieval,
                    rerank_result=pipeline_result.rerank,
                    generation_result=pipeline_result.generation,
                    grounding_result=grounding,
                    failure_analysis=failure,
                    decision=decision,
                )
            )

            if not decision.should_retry:
                logger.success("Target Pipeline Execution Succeeded. Response grounded.")
                return state

            state.retry_count += 1

        logger.warning("Maximum retry limit reached.")
        return state