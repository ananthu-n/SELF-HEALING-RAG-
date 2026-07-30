from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Strategy Enums
# ----------------------------------------------------------------------


class RetrievalStrategy(str, Enum):
    """
    Strategy used for the next retrieval attempt.
    """

    DEFAULT = "default"

    EXPAND_RETRIEVAL = "expand_retrieval"

    INCREASE_DENSE = "increase_dense"

    INCREASE_BM25 = "increase_bm25"

    BALANCED = "balanced"

    DIVERSIFY = "diversify"

    SEARCH_ARXIV = "search_arxiv"


class RewriteStrategy(str, Enum):
    """
    Strategy used to rewrite the query.
    """

    NONE = "none"

    REPHRASE = "rephrase"

    EXPAND = "expand"

    DISAMBIGUATE = "disambiguate"

    SCIENTIFIC = "scientific"


class DiversityStrategy(str, Enum):
    """
    Strategy used to avoid repeated retrieval.
    """

    NONE = "none"

    DIFFERENT_PAPERS = "different_papers"

    DIFFERENT_CHUNKS = "different_chunks"

    HYBRID = "hybrid"


# ----------------------------------------------------------------------
# Healing Plan
# ----------------------------------------------------------------------


class HealingPlan(BaseModel):
    """
    Immutable execution plan for the next retry.

    The planner decides WHAT strategy to use.

    The pipeline decides HOW to execute it.
    """

    # ---------------------------------------------------------
    # Query
    # ---------------------------------------------------------

    query: str

    # ---------------------------------------------------------
    # Retry metadata
    # ---------------------------------------------------------

    retry_number: int = 0

    reason: str

    # ---------------------------------------------------------
    # Strategy
    # ---------------------------------------------------------

    retrieval_strategy: RetrievalStrategy = (
        RetrievalStrategy.DEFAULT
    )

    rewrite_strategy: RewriteStrategy = (
        RewriteStrategy.NONE
    )

    diversity_strategy: DiversityStrategy = (
        DiversityStrategy.NONE
    )

    # ---------------------------------------------------------
    # Compatibility
    # ---------------------------------------------------------

    rewrite_query: bool = False

    increase_top_k: bool = False

    # ---------------------------------------------------------
    # Retrieval Parameters
    # ---------------------------------------------------------

    top_k: int = Field(
        default=10,
        ge=1,
    )

    dense_top_k: int = Field(
        default=20,
        ge=1,
    )

    bm25_top_k: int = Field(
        default=20,
        ge=1,
    )

    metadata_filters: dict | None = None