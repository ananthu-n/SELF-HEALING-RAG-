from __future__ import annotations

from pydantic import BaseModel, Field

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


class SelfHealingState(BaseModel):
    """
    Stores the complete execution state of the Self-Healing RAG pipeline.
    """

    original_query: str
    current_query: str

    retrieval_result: RetrievalResult | None = None
    rerank_result: RerankResult | None = None
    context_result: ContextResult | None = None
    prompt_result: PromptResult | None = None
    last_generation: GenerationResult | None = None

    last_grounding: GroundingResult | None = None
    last_failure: FailureAnalysis | None = None
    last_decision: RetryDecision | None = None

    last_healing_plan: HealingPlan | None = None
    healing_history: list[HealingPlan] = Field(default_factory=list)
    retrieval_memory: RetrievalMemory = Field(default_factory=RetrievalMemory)
    attempts_history: list[AttemptResult] = Field(default_factory=list)

    retry_count: int = 0


class AttemptResult(BaseModel):
    """
    Tracks the execution result of a single self-healing loop attempt.
    """
    attempt_number: int
    query: str
    retrieval_result: RetrievalResult | None = None
    rerank_result: RerankResult | None = None
    generation_result: GenerationResult | None = None
    grounding_result: GroundingResult | None = None
    failure_analysis: FailureAnalysis | None = None
    decision: RetryDecision | None = None