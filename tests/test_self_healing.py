from __future__ import annotations

from app.core.logger import logger
from app.self_healing.controller import SelfHealingController


def main() -> None:

    query = "What is Self-RAG?"

    logger.info("=" * 80)
    logger.info("Starting Self-Healing Test")
    logger.info("=" * 80)

    controller = SelfHealingController()

    state = controller.answer(query)

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(state.original_query)

    print("\n" + "=" * 80)
    print("FINAL QUERY")
    print("=" * 80)
    print(state.current_query)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)

    if state.last_generation:
        print(state.last_generation.response.answer)
    else:
        print("No answer generated.")

    print("\n" + "=" * 80)
    print("GROUNDING")
    print("=" * 80)

    if state.last_grounding:
        response = state.last_grounding.response

        print(f"Grounded   : {response.is_grounded}")
        print(f"Confidence : {response.confidence:.2f}")
        print(f"Reason     : {response.reason}")

        if response.unsupported_claims:

            print("\nUnsupported Claims")

            for claim in response.unsupported_claims:

                print(f"- {claim.claim}")

                print(f"  Reason: {claim.reason}")

    else:

        print("Grounding was not executed.")

    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)

    if state.last_failure:

        print(f"Failure Type : {state.last_failure.failure_type}")

        print(f"Reason       : {state.last_failure.reason}")

    else:

        print("No failure detected.")

    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)

    if state.last_decision:

        print(f"Action       : {state.last_decision.action}")

        print(f"Retry        : {state.last_decision.should_retry}")

        print(f"Reason       : {state.last_decision.reason}")

    else:

        print("No decision.")

    print("\n" + "=" * 80)
    print("HEALING")
    print("=" * 80)

    print(f"Attempts : {state.retry_count + 1}")

    print(f"Plans    : {len(state.healing_history)}")

    for index, plan in enumerate(
        state.healing_history,
        start=1,
    ):

        print(f"\nAttempt {index}")

        print(f"Query          : {plan.query}")

        print(f"Rewrite Query  : {plan.rewrite_query}")

        print(f"Top-K          : {plan.top_k}")

        print(f"Dense Top-K    : {plan.dense_top_k}")

        print(f"BM25 Top-K     : {plan.bm25_top_k}")

        print(f"Reason         : {plan.reason}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()