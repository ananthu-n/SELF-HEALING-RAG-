from __future__ import annotations

from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranker.reranker import CrossEncoderReranker


def main() -> None:

    retriever = HybridRetriever()

    retrieval_result = retriever.retrieve(
        query="What is Retrieval Augmented Generation?",
        top_k=10,
    )

    reranker = CrossEncoderReranker()

    rerank_result = reranker(
        retrieval_result,
        top_k=5,
    )

    print()

    print("=" * 80)
    print("CrossEncoder Reranker Test")
    print("=" * 80)

    print()

    print(f"Returned {rerank_result.total_results} chunks")

    print()

    for index, chunk in enumerate(
        rerank_result.reranked_chunks,
        start=1,
    ):

        print("-" * 80)

        print(f"Rank            : {index}")

        print(f"Paper           : {chunk.paper_id}")

        print(f"Chunk           : {chunk.chunk_id}")

        print(f"Retriever Score : {chunk.score:.4f}")

        print(f"Reranker Score  : {chunk.reranker_score:.4f}")

        if "rrf_score" in chunk.metadata:

            print(
                f"RRF Score       : "
                f"{chunk.metadata['rrf_score']:.6f}"
            )

        print()

        print(chunk.text[:250])

        print()

    print("=" * 80)

    print("CrossEncoder Test Passed")


if __name__ == "__main__":

    main()