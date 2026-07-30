from __future__ import annotations

from app.core.logger import logger
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.fusion.rrf import ReciprocalRankFusion
from app.retrieval.models import RetrievalResult


class HybridRetriever:
    """
    Hybrid Retriever with Search Scope Filtering.

    Search Scopes:
        - "hybrid" (default): Search across both Custom Uploaded Documents & arXiv Research Papers.
        - "custom_only": Restrict knowledge retrieval exclusively to user uploaded documents.
        - "arxiv_only": Restrict knowledge retrieval exclusively to arXiv research papers.
    """

    def __init__(
        self,
        rrf_k: int = 60,
    ) -> None:

        logger.info("Initializing Hybrid Retriever...")

        self.dense_retriever = DenseRetriever()

        self.bm25_retriever = BM25Retriever()

        self.rrf = ReciprocalRankFusion(k=rrf_k)

        logger.success("Hybrid Retriever initialized.")

    # ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
        search_scope: str = "hybrid",
    ) -> RetrievalResult:
        """
        Perform hybrid retrieval using Dense + BM25 + RRF + Scope Filtering.
        """

        dense_top_k = dense_top_k or (top_k * 3)
        bm25_top_k = bm25_top_k or (top_k * 3)

        logger.info(f"Hybrid retrieval started (Scope: {search_scope}): '{query}'")

        # ---------------------------------------------
        # Dense & BM25 Candidate Retrieval
        # ---------------------------------------------
        dense_result = self.dense_retriever.retrieve(
            query=query,
            top_k=dense_top_k,
        )

        bm25_result = self.bm25_retriever.retrieve(
            query=query,
            top_k=bm25_top_k,
        )

        # ---------------------------------------------
        # Reciprocal Rank Fusion
        # ---------------------------------------------
        fused_result = self.rrf.fuse(
            query=query,
            results=[
                dense_result,
                bm25_result,
            ],
            top_k=dense_top_k + bm25_top_k,
        )

        from app.retrieval.pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline(
            top_k=top_k * 2,
        )

        filtered_chunks = pipeline.process(fused_result.retrieved_chunks)

        # ---------------------------------------------
        # Scope-Based Filtering (Custom Only vs arXiv Only vs Hybrid)
        # ---------------------------------------------
        scope_normalized = (search_scope or "hybrid").lower()
        
        if scope_normalized == "custom_only":
            logger.info("Applying Search Scope Filter: CUSTOM UPLOADED KNOWLEDGE BASE ONLY")
            scoped_chunks = [c for c in filtered_chunks if str(getattr(c, "paper_id", "")).startswith("custom_")]
        elif scope_normalized == "arxiv_only":
            logger.info("Applying Search Scope Filter: ARXIV PAPERS ONLY")
            scoped_chunks = [c for c in filtered_chunks if not str(getattr(c, "paper_id", "")).startswith("custom_")]
        else:
            logger.info("Applying Search Scope Filter: HYBRID (arXiv + Custom Knowledge)")
            scoped_chunks = filtered_chunks

        # Truncate to top_k
        final_chunks = scoped_chunks[:top_k]

        hybrid_result = RetrievalResult(
            query=query,
            retrieved_chunks=final_chunks,
            total_results=len(final_chunks),
        )

        logger.success(
            f"Hybrid retrieval ({search_scope}) completed with "
            f"{hybrid_result.total_results} chunks."
        )

        return hybrid_result

    # ---------------------------------------------------------

    def retrieve_all(
        self,
        query: str,
        top_k: int = 10,
    ) -> dict[str, RetrievalResult]:
        """
        Return all intermediate retrieval outputs.
        """

        dense_result = self.dense_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        bm25_result = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        hybrid_result = self.rrf.fuse(
            query=query,
            results=[
                dense_result,
                bm25_result,
            ],
            top_k=top_k,
        )

        return {
            "dense": dense_result,
            "bm25": bm25_result,
            "hybrid": hybrid_result,
        }

    # ---------------------------------------------------------

    def __call__(
        self,
        query: str,
        top_k: int = 10,
        search_scope: str = "hybrid",
    ) -> RetrievalResult:

        return self.retrieve(
            query=query,
            top_k=top_k,
            search_scope=search_scope,
        )