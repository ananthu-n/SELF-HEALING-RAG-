from enum import Enum

from pydantic import BaseModel


class DecisionAction(str, Enum):
    """
    Possible actions after grounding evaluation.
    """

    RETURN_ANSWER = "return_answer"

    RETRY_RETRIEVAL = "retry_retrieval"

    REWRITE_QUERY = "rewrite_query"

    FAIL = "fail"


class RetryDecision(BaseModel):
    """
    Decision returned by the Decision Engine.
    """

    should_retry: bool

    action: DecisionAction

    reason: str