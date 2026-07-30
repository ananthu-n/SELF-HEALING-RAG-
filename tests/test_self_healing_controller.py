from app.self_healing.controller import SelfHealingController


def main():

    controller = SelfHealingController()

    state = controller.answer(

        "What is Self-RAG?"

    )

    print("\n")
    print("=" * 100)
    print("FINAL ANSWER")
    print("=" * 100)

    if state.last_generation:
        print(state.last_generation.response.answer)
    else:
        print("No generation")

    print("\n")
    print("=" * 100)
    print("GROUNDING")
    print("=" * 100)

    if state.last_grounding:
        print(state.last_grounding.response.model_dump())
    else:
        print("No grounding")

    print("\n")
    print("=" * 100)
    print("DECISION")
    print("=" * 100)

    if state.last_decision:
        print(state.last_decision.model_dump())
    else:
        print("No decision")


if __name__ == "__main__":
    main()