from pydantic import BaseModel, Field


class UnsupportedClaim(BaseModel):
    """
    Represents a statement in the generated answer
    that is not sufficiently supported by the retrieved context.
    """

    claim: str = Field(..., description="Unsupported statement.")
    reason: str = Field(..., description="Reason why it is unsupported.")


class GroundingRequest(BaseModel):
    """
    Input to the grounding evaluator.
    """

    query: str
    answer: str
    context: str


class GroundingResponse(BaseModel):
    """
    Output of the grounding evaluator.
    """

    is_grounded: bool

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that the answer is grounded."
    )

    unsupported_claims: list[UnsupportedClaim] = Field(
        default_factory=list
    )

    should_retry: bool

    reason: str


class GroundingResult(BaseModel):
    """
    Final evaluation result returned by the evaluator.
    """

    response: GroundingResponse