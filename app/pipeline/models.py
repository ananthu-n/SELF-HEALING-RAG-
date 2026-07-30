from __future__ import annotations

from pydantic import BaseModel

from app.retrieval.models import RetrievalResult
from app.reranker.models import RerankResult
from app.context.models import ContextResult
from app.prompt.models import PromptResult
from app.llm.models import GenerationResult


class PipelineResult(BaseModel):
    """
    Complete output of one RAG pipeline execution.

    Every stage is preserved so downstream modules
    (evaluation, self-healing, API, metrics, logging)
    can inspect intermediate results.
    """

    retrieval: RetrievalResult

    rerank: RerankResult

    context: ContextResult

    prompt: PromptResult

    generation: GenerationResult