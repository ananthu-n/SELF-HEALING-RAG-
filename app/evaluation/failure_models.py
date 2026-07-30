from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

class FailureType(str, Enum):
    """
    Root cause identified after grounding evaluation.
    """

    NONE = "none"

    PARTIAL_GROUNDING = "partial_grounding"

    HALLUCINATION = "hallucination"

    INSUFFICIENT_CONTEXT = "insufficient_context"

    IRRELEVANT_RETRIEVAL = "irrelevant_retrieval"

    AMBIGUOUS_QUERY = "ambiguous_query"

    LOW_CONFIDENCE = "low_confidence"


class FailureAnalysis(BaseModel):
    """
    Result produced by the Failure Analyzer.
    """

    failure_type: FailureType

    confidence: float

    should_retry: bool

    reason: str

