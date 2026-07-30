from __future__ import annotations

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import RetrievalResult


def main() -> None:

    retriever = HybridRetriever()

    result = retriever.retrieve(
        query="What is Retrieval Augmented Generation?",
        top_k=5,
    )

    assert isinstance(result, RetrievalResult)

    assert result.total_results > 0

    assert len(result.retrieved_chunks) == result.total_results

    print()

    print("=" * 80)
    print("Hybrid Retrieval Test")
    print("=" * 80)

    print(f"Retrieved: {result.total_results}")

    for index, chunk in enumerate(result.retrieved_chunks, start=1):

        print("-" * 80)

        print(f"Rank      : {index}")
        print(f"Chunk ID  : {chunk.chunk_id}")
        print(f"Paper ID  : {chunk.paper_id}")
        print(f"Score     : {chunk.score:.4f}")

        if "rrf_score" in chunk.metadata:
            print(f"RRF Score : {chunk.metadata['rrf_score']:.6f}")

        print()

        print(chunk.text[:250])

    print()

    print("Hybrid Retrieval Test Passed")


if __name__ == "__main__":
    main()