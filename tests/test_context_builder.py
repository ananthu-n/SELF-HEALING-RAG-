from __future__ import annotations

from app.context.builder import ContextBuilder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranker.reranker import CrossEncoderReranker


def main() -> None:

    query = "What is Retrieval Augmented Generation?"

    retriever = HybridRetriever()

    retrieval = retriever.retrieve(
        query=query,
        top_k=10,
    )

    reranker = CrossEncoderReranker()

    reranked = reranker(
        retrieval,
        top_k=10,
    )

    builder = ContextBuilder(
        token_budget=1000,
    )

    context = builder(
        reranked,
    )

    print()

    print("=" * 80)
    print("Context Builder Test")
    print("=" * 80)

    print(f"Query              : {context.query}")
    print(f"Chunks             : {context.total_chunks}")
    print(f"Estimated Tokens   : {context.statistics.estimated_tokens}")
    print(f"Characters         : {context.statistics.total_characters}")
    print(f"Unique Papers      : {context.statistics.unique_papers}")
    print(f"Remaining Budget   : {context.statistics.remaining_budget}")
    print(f"Truncated          : {context.statistics.truncated}")

    print()

    for i, chunk in enumerate(
        context.context_chunks,
        start=1,
    ):

        print("-" * 80)

        print(f"Rank : {i}")

        print(f"Paper : {chunk.paper_id}")

        print(f"Chunk : {chunk.chunk_id}")

        print(
            f"CrossEncoder : {chunk.reranker_score:.4f}"
        )

        print(chunk.text[:150])

        print()

    print("=" * 80)
    print("Context Builder Test Passed")


if __name__ == "__main__":
    main()