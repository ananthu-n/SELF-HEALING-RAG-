from app.evaluation.decision_engine import DecisionEngine
from app.evaluation.models import (
    GroundingResponse,
    GroundingResult,
)


def main():

    grounding = GroundingResult(
        response=GroundingResponse(
            is_grounded=False,
            confidence=0.42,
            unsupported_claims=[],
            should_retry=True,
            reason="Answer is only partially supported.",
        )
    )

    engine = DecisionEngine()

    decision = engine.decide(grounding)

    print()

    print(decision.model_dump())


if __name__ == "__main__":
    main()