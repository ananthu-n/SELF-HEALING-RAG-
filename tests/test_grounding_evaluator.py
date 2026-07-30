from app.evaluation.models import GroundingRequest
from app.evaluation.evaluator import GroundingEvaluator


def main():

    request = GroundingRequest(
        query="What is Self-RAG?",
        answer=(
            "Self-RAG improves factual accuracy using "
            "on-demand retrieval and reflection tokens."
        ),
        context=(
            "SELF-RAG enhances factual accuracy through "
            "on-demand retrieval and reflection tokens."
        ),
    )

    evaluator = GroundingEvaluator()

    result = evaluator.evaluate(request)

    print("\nRESULT\n")

    print(result.model_dump())


if __name__ == "__main__":
    main()