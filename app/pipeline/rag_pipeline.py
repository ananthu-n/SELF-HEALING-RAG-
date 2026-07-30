from __future__ import annotations

from time import perf_counter

from app.core.logger import logger

from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranker.reranker import CrossEncoderReranker
from app.context.builder import ContextBuilder
from app.prompt.builder import PromptBuilder
from app.llm.generator import LLMGenerator
from app.pipeline.models import PipelineResult

class RAGPipeline:
    """
    Executes a single Retrieval-Augmented Generation (RAG) pass with Search Scope configuration.
    """

    def __init__(self) -> None:

        self.retriever = HybridRetriever()

        self.reranker = CrossEncoderReranker()

        self.context_builder = ContextBuilder()

        self.prompt_builder = PromptBuilder()

        self.generator = LLMGenerator()

    def run_once(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        search_scope: str = "hybrid",
    ) -> PipelineResult:

        logger.info(f"Starting RAG pipeline for query: '{query}' (Scope: {search_scope})")

        total_start = perf_counter()

        # ==========================================================
        # Retrieval
        # ==========================================================

        retrieval_start = perf_counter()

        retrieval_result = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            search_scope=search_scope,
        )

        retrieval_time = perf_counter() - retrieval_start

        logger.success(
            f"Retrieved {retrieval_result.total_results} chunks "
            f"in {retrieval_time:.2f}s"
        )

        # ==========================================================
        # Retrieval Validation (Early-Exit)
        # ==========================================================
        from app.retrieval.validator import RetrievalValidator
        
        validator = RetrievalValidator()
        validation = validator.validate(retrieval_result)
        
        if not validation.is_valid:
            logger.warning(f"Retrieval validation failed: {validation.reason}. Early-exiting pipeline.")
            from app.reranker.models import RerankResult
            from app.context.models import ContextResult
            from app.prompt.models import PromptResult, PromptInput
            from app.llm.models import GenerationResult, GenerationResponse
            
            rerank_result = RerankResult(query=query, reranked_chunks=[], total_results=0)
            context_result = ContextResult(query=query, text="", chunks=[], token_estimate=0)
            prompt_result = PromptResult(
                prompt=PromptInput(system_prompt="", user_prompt="", context="", total_chunks=0),
                citations=[]
            )
            generation_result = GenerationResult(
                response=GenerationResponse(
                    answer=f"Insufficient context found in selected scope ('{search_scope}'). Please ensure relevant documents are uploaded or switch search mode.",
                    model="none",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0
                ),
                citations=[]
            )
            
            return PipelineResult(
                retrieval=retrieval_result,
                rerank=rerank_result,
                context=context_result,
                prompt=prompt_result,
                generation=generation_result,
            )

        # ==========================================================
        # Reranking
        # ==========================================================

        rerank_start = perf_counter()

        rerank_result = self.reranker.rerank(
            retrieval_result=retrieval_result,
        )

        rerank_time = perf_counter() - rerank_start

        logger.success(
            f"Reranked {rerank_result.total_results} chunks "
            f"in {rerank_time:.2f}s"
        )

        # ==========================================================
        # Context Building
        # ==========================================================

        context_start = perf_counter()

        context_result = self.context_builder.build(
            rerank_result=rerank_result,
        )

        context_time = perf_counter() - context_start

        logger.success(
            f"Context built in {context_time:.2f}s"
        )

        # ==========================================================
        # Prompt Building
        # ==========================================================

        prompt_start = perf_counter()

        prompt_result = self.prompt_builder.build(
            context_result=context_result,
        )

        prompt_time = perf_counter() - prompt_start

        logger.success(
            f"Prompt built in {prompt_time:.2f}s"
        )

        # ==========================================================
        # Generation
        # ==========================================================

        generation_start = perf_counter()

        generation_result = self.generator.generate(
            prompt_result,
        )

        generation_time = perf_counter() - generation_start

        logger.success(
            f"Generation completed in {generation_time:.2f}s"
        )

        # ==========================================================
        # Performance Summary
        # ==========================================================

        total_time = perf_counter() - total_start

        logger.info("=" * 60)
        logger.info("PIPELINE PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Retrieval : {retrieval_time:.2f}s")
        logger.info(f"Reranking : {rerank_time:.2f}s")
        logger.info(f"Context   : {context_time:.2f}s")
        logger.info(f"Prompt    : {prompt_time:.2f}s")
        logger.info(f"Generation: {generation_time:.2f}s")
        logger.info(f"Total     : {total_time:.2f}s")
        logger.info("=" * 60)

        logger.success("Pipeline execution completed.")

        return PipelineResult(
            retrieval=retrieval_result,
            rerank=rerank_result,
            context=context_result,
            prompt=prompt_result,
            generation=generation_result,
        )

    def answer(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        search_scope: str = "hybrid",
    ):
        return self.run_once(
            query=query,
            top_k=top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            search_scope=search_scope,
        )