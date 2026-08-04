from __future__ import annotations

from typing import TypedDict, List, Optional
from collections import namedtuple

from app.core.config import settings
from app.core.logger import logger

from app.pipeline.rag_pipeline import RAGPipeline
from app.pipeline.query_enhancement import QueryEnhancer
from app.evaluation.evaluator import GroundingEvaluator
from app.evaluation.crag_validator import CRAGValidator
from app.evaluation.decision_engine import DecisionEngine
from app.evaluation.models import GroundingRequest
from app.pipeline.models import PipelineResult

from app.ingestion.dynamic_acquisition import DynamicKnowledgeAcquisitor
from app.self_healing.healing_models import RetrievalStrategy
from app.self_healing.models import SelfHealingState, AttemptResult
from app.self_healing.query_rewriter import QueryRewriter
from app.self_healing.healing_planner import HealingPlanner
from app.evaluation.failure_analyzer import FailureAnalyzer

from app.retrieval.models import RetrievalResult
from app.reranker.models import RerankResult
from app.context.models import ContextResult
from app.prompt.models import PromptResult
from app.llm.models import GenerationResult
from app.evaluation.models import GroundingResult
from app.evaluation.failure_models import FailureAnalysis
from app.evaluation.decision import RetryDecision
from app.self_healing.healing_models import HealingPlan
from app.self_healing.retrieval_memory import RetrievalMemory

from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    original_query: str
    current_query: str
    extracted_topic: str
    search_scope: str
    retry_count: int
    session_id: Optional[str]

    retrieval_result: Optional[RetrievalResult]
    rerank_result: Optional[RerankResult]
    context_result: Optional[ContextResult]
    prompt_result: Optional[PromptResult]
    last_generation: Optional[GenerationResult]

    last_grounding: Optional[GroundingResult]
    last_failure: Optional[FailureAnalysis]
    last_decision: Optional[RetryDecision]

    last_healing_plan: Optional[HealingPlan]
    healing_history: List[HealingPlan]
    retrieval_memory: RetrievalMemory
    attempts_history: List[AttemptResult]


# ----------------------------------------------------------------------
# Node Functions
# ----------------------------------------------------------------------

from app.core.cancellation import CancellationManager

def check_cancel(state: GraphState):
    session_id = state.get("session_id")
    if session_id and CancellationManager.is_cancelled(session_id):
        raise InterruptedError("Execution stopped by user.")


def enhance_query_node(state: GraphState) -> dict:
    check_cancel(state)
    logger.info("=" * 80)
    logger.info(f"Starting LangGraph Self-Healing RAG Pipeline (Scope: {state['search_scope']})")
    logger.info("=" * 80)

    query_enhancer = QueryEnhancer(enable_hyde=True, enable_decomposition=True)
    enhanced_query = query_enhancer.enhance(state["original_query"])
    logger.info(f"Topic Extracted: '{enhanced_query.extracted_topic}'")

    return {
        "current_query": enhanced_query.search_query,
        "extracted_topic": enhanced_query.extracted_topic,
        "retry_count": 0,
        "healing_history": [],
        "attempts_history": [],
        "retrieval_memory": RetrievalMemory(),
    }


def plan_healing_node(state: GraphState) -> dict:
    check_cancel(state)
    logger.info(f"Self-Healing Execution Cycle [{state['retry_count'] + 1}/{settings.self_healing.max_retries}]")
    
    # Construct transient SelfHealingState for the planner compatibility
    sh_state = SelfHealingState(
        original_query=state["original_query"],
        current_query=state["current_query"],
        retrieval_result=state["retrieval_result"],
        rerank_result=state["rerank_result"],
        context_result=state["context_result"],
        prompt_result=state["prompt_result"],
        last_generation=state["last_generation"],
        last_grounding=state["last_grounding"],
        last_failure=state["last_failure"],
        last_decision=state["last_decision"],
        last_healing_plan=state["last_healing_plan"],
        healing_history=state["healing_history"],
        retrieval_memory=state["retrieval_memory"],
        attempts_history=state["attempts_history"],
        retry_count=state["retry_count"],
    )

    planner = HealingPlanner()
    plan = planner.plan(sh_state)

    logger.info("=" * 60)
    logger.info(f"Healing Plan Strategy: {plan.retrieval_strategy.value} | Reason: {plan.reason}")
    logger.info("=" * 60)

    return {
        "last_healing_plan": plan,
        "healing_history": state["healing_history"] + [plan],
    }


def acquire_knowledge_node(state: GraphState) -> dict:
    check_cancel(state)
    logger.info("Executing Dynamic Knowledge Acquisition on arXiv...")
    dynamic_acquisitor = DynamicKnowledgeAcquisitor(max_results=10, top_k_download=3)
    try:
        acquired_count = dynamic_acquisitor.acquire(state["extracted_topic"])
        logger.info(f"Dynamic Acquisition indexed {acquired_count} fresh research paper(s).")
    except Exception as err:
        logger.error(f"Dynamic Acquisition failed: {err}")
    return {}


def rewrite_query_node(state: GraphState) -> dict:
    check_cancel(state)
    query_rewriter = QueryRewriter()
    plan = state["last_healing_plan"]
    
    search_terms = query_rewriter.rewrite(
        query=plan.query,
        reason=plan.reason,
        rewrite_strategy=plan.rewrite_strategy,
    )
    
    # Mutate plan's query to update the query to be retrieved with
    plan.query = search_terms
    logger.info(f"Search Query Terms: {search_terms}")
    
    return {
        "current_query": search_terms,
        "last_healing_plan": plan,
    }


def rag_pass_node(state: GraphState) -> dict:
    check_cancel(state)
    pipeline = RAGPipeline()
    plan = state["last_healing_plan"]

    pipeline_result = pipeline.answer(
        query=plan.query,
        top_k=plan.top_k,
        dense_top_k=plan.dense_top_k,
        bm25_top_k=plan.bm25_top_k,
        search_scope=state["search_scope"],
    )

    return {
        "retrieval_result": pipeline_result.retrieval,
        "rerank_result": pipeline_result.rerank,
        "context_result": pipeline_result.context,
        "prompt_result": pipeline_result.prompt,
        "last_generation": pipeline_result.generation,
    }


def crag_validation_node(state: GraphState) -> dict:
    check_cancel(state)
    crag_validator = CRAGValidator()
    plan = state["last_healing_plan"]
    rerank_result = state["rerank_result"]

    crag_res = crag_validator.validate(
        query=plan.query,
        reranked_chunks=rerank_result.reranked_chunks,
    )

    if crag_res.requires_external_search and state["search_scope"] != "custom_only":
        check_cancel(state)
        logger.warning(f"CRAG State {crag_res.state.value.upper()}: Triggering dynamic arXiv search for '{state['extracted_topic']}'...")
        try:
            dynamic_acquisitor = DynamicKnowledgeAcquisitor(max_results=10, top_k_download=3)
            acquired = dynamic_acquisitor.acquire(state["extracted_topic"])
            logger.info(f"CRAG Acquisition indexed {acquired} fresh paper(s).")
            
            if acquired > 0:
                check_cancel(state)
                logger.info("Re-running RAG pipeline with freshly acquired evidence...")
                pipeline = RAGPipeline()
                pipeline_result = pipeline.answer(
                    query=plan.query,
                    top_k=plan.top_k,
                    dense_top_k=plan.dense_top_k,
                    bm25_top_k=plan.bm25_top_k,
                    search_scope=state["search_scope"],
                )
                return {
                    "retrieval_result": pipeline_result.retrieval,
                    "rerank_result": pipeline_result.rerank,
                    "context_result": pipeline_result.context,
                    "prompt_result": pipeline_result.prompt,
                    "last_generation": pipeline_result.generation,
                }
        except Exception as err:
            logger.error(f"CRAG Acquisition failed: {err}")
            
    return {}


def evaluate_grounding_node(state: GraphState) -> dict:
    check_cancel(state)
    evaluator = GroundingEvaluator()
    failure_analyzer = FailureAnalyzer()
    decision_engine = DecisionEngine()
    plan = state["last_healing_plan"]

    grounding = evaluator.evaluate(
        GroundingRequest(
            query=plan.query,
            answer=state["last_generation"].response.answer,
            context=state["prompt_result"].prompt.context,
        )
    )

    # Wrap outputs inside a pipeline_result structure
    pipeline_result = PipelineResult(
        retrieval=state["retrieval_result"],
        rerank=state["rerank_result"],
        context=state["context_result"],
        prompt=state["prompt_result"],
        generation=state["last_generation"],
    )

    failure = failure_analyzer.analyze(
        grounding=grounding,
        pipeline_result=pipeline_result,
    )
    decision = decision_engine.decide(failure)

    attempt = AttemptResult(
        attempt_number=state["retry_count"] + 1,
        query=plan.query,
        retrieval_result=state["retrieval_result"],
        rerank_result=state["rerank_result"],
        generation_result=state["last_generation"],
        grounding_result=grounding,
        failure_analysis=failure,
        decision=decision,
    )

    return {
        "last_grounding": grounding,
        "last_failure": failure,
        "last_decision": decision,
        "attempts_history": state["attempts_history"] + [attempt],
        "retry_count": state["retry_count"] + 1,
    }


# ----------------------------------------------------------------------
# Conditional Routers
# ----------------------------------------------------------------------

def route_after_planning(state: GraphState) -> str:
    plan = state["last_healing_plan"]
    search_scope = state["search_scope"]
    retry_count = state["retry_count"]
    retrieval_result = state["retrieval_result"]

    # Check if we should execute arXiv knowledge acquisition
    if search_scope != "custom_only" and (
        plan.retrieval_strategy == RetrievalStrategy.SEARCH_ARXIV or 
        (retry_count > 0 and not retrieval_result)
    ):
        return "acquire_knowledge"

    # Otherwise skip directly to check rewrite
    if plan.rewrite_query:
        return "rewrite_query"
    return "rag_pass"


def route_after_acquisition(state: GraphState) -> str:
    plan = state["last_healing_plan"]
    if plan.rewrite_query:
        return "rewrite_query"
    return "rag_pass"


def route_loop_check(state: GraphState) -> str:
    decision = state["last_decision"]
    retry_count = state["retry_count"]

    if not decision.should_retry:
        logger.success("Target Pipeline Execution Succeeded. Response grounded.")
        return "end"

    if retry_count >= settings.self_healing.max_retries:
        logger.warning("Maximum retry limit reached.")
        return "end"

    return "continue"


# ----------------------------------------------------------------------
# Graph Assembly
# ----------------------------------------------------------------------

def build_self_healing_graph():
    workflow = StateGraph(GraphState)

    # Register Nodes
    workflow.add_node("enhance_query", enhance_query_node)
    workflow.add_node("plan_healing", plan_healing_node)
    workflow.add_node("acquire_knowledge", acquire_knowledge_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("rag_pass", rag_pass_node)
    workflow.add_node("crag_validation", crag_validation_node)
    workflow.add_node("evaluate_grounding", evaluate_grounding_node)

    # Define Connections
    workflow.set_entry_point("enhance_query")
    workflow.add_edge("enhance_query", "plan_healing")

    # Routing from planning
    workflow.add_conditional_edges(
        "plan_healing",
        route_after_planning,
        {
            "acquire_knowledge": "acquire_knowledge",
            "rewrite_query": "rewrite_query",
            "rag_pass": "rag_pass",
        }
    )

    # Routing from acquisition
    workflow.add_conditional_edges(
        "acquire_knowledge",
        route_after_acquisition,
        {
            "rewrite_query": "rewrite_query",
            "rag_pass": "rag_pass",
        }
    )

    # Normal sequential flow
    workflow.add_edge("rewrite_query", "rag_pass")
    workflow.add_edge("rag_pass", "crag_validation")
    workflow.add_edge("crag_validation", "evaluate_grounding")

    # Loop check routing
    workflow.add_conditional_edges(
        "evaluate_grounding",
        route_loop_check,
        {
            "continue": "plan_healing",
            "end": END,
        }
    )

    return workflow.compile()


# ----------------------------------------------------------------------
# Controller Class (Drop-in Replacement)
# ----------------------------------------------------------------------

class LangGraphSelfHealingController:
    """
    Production Executive Orchestrator built using LangGraph.
    Achieves identical logic to SelfHealingController but runs via graph nodes & edges.
    """

    def __init__(self):
        self.app = build_self_healing_graph()

    def answer(self, query: str, search_scope: str = "hybrid", session_id: Optional[str] = None) -> SelfHealingState:
        # Initialize state Dict
        initial_state: GraphState = {
            "original_query": query,
            "current_query": query,
            "extracted_topic": "",
            "search_scope": search_scope,
            "retry_count": 0,
            "session_id": session_id,
            
            "retrieval_result": None,
            "rerank_result": None,
            "context_result": None,
            "prompt_result": None,
            "last_generation": None,
            
            "last_grounding": None,
            "last_failure": None,
            "last_decision": None,
            
            "last_healing_plan": None,
            "healing_history": [],
            "retrieval_memory": RetrievalMemory(),
            "attempts_history": [],
        }

        # Invoke Graph
        final_state = self.app.invoke(initial_state)

        # Convert back to SelfHealingState to preserve external API compatibility
        return SelfHealingState(
            original_query=final_state["original_query"],
            current_query=final_state["current_query"],
            retrieval_result=final_state["retrieval_result"],
            rerank_result=final_state["rerank_result"],
            context_result=final_state["context_result"],
            prompt_result=final_state["prompt_result"],
            last_generation=final_state["last_generation"],
            last_grounding=final_state["last_grounding"],
            last_failure=final_state["last_failure"],
            last_decision=final_state["last_decision"],
            last_healing_plan=final_state["last_healing_plan"],
            healing_history=final_state["healing_history"],
            retrieval_memory=final_state["retrieval_memory"],
            attempts_history=final_state["attempts_history"],
            retry_count=final_state["retry_count"],
        )
