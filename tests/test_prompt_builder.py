from app.prompt.builder import PromptBuilder
from app.retrieval.models import RetrievedChunk
from app.reranker.models import (
    RerankedChunk,
    RerankResult,
)


def main():

    chunk = RetrievedChunk(
        paper_id="paper1",
        chunk_id="chunk10",
        text="Self-RAG introduces reflection tokens for self-reflection.",
        score=0.84,
        page_number=2,
        chunk_index=0,
        char_start=0,
        char_end=120,
        metadata={},
    )

    rerank_result = RerankResult(
        query="What is Self-RAG?",
        chunks=[
            RerankedChunk(
                chunk=chunk,
                reranker_score=9.42,
            )
        ],
        total_chunks=1,
    )

    builder = PromptBuilder()

    result = builder.build(rerank_result)

    print("=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(result.prompt.system_prompt)

    print()

    print("=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(result.prompt.user_prompt)

    print()

    print("=" * 80)
    print("CITATIONS")
    print("=" * 80)

    for citation in result.citations:
        print(citation.model_dump())


if __name__ == "__main__":
    main()