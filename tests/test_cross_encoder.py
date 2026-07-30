from app.reranker.cross_encoder import CrossEncoderModel


def main():

    model = CrossEncoderModel()

    pairs = [
        (
            "What is Self-RAG?",
            "Self-RAG is a self-reflective retrieval augmented generation framework.",
        ),
        (
            "What is Self-RAG?",
            "Bananas are yellow fruits.",
        ),
    ]

    scores = model.predict(pairs)

    print()

    for i, score in enumerate(scores, start=1):
        print(f"Pair {i}: {score:.4f}")


if __name__ == "__main__":
    main()