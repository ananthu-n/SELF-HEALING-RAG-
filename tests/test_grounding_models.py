from app.evaluation.models import (
    GroundingRequest,
    GroundingResponse,
    GroundingResult,
    UnsupportedClaim,
)


def main():

    request = GroundingRequest(
        query="What is Self-RAG?",
        answer="Self-RAG uses reflection tokens.",
        context="Self-RAG introduces reflection tokens."
    )

    response = GroundingResponse(
        is_grounded=True,
        confidence=0.95,
        unsupported_claims=[],
        should_retry=False,
        reason="Answer supported by retrieved context."
    )

    result = GroundingResult(
        response=response
    )

    print("\nREQUEST")
    print(request.model_dump())

    print("\nRESULT")
    print(result.model_dump())


if __name__ == "__main__":
    main()