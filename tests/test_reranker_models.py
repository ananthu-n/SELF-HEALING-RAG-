from app.retrieval.models import RetrievedChunk
from app.reranker.models import (
    RerankedChunk,
    RerankResult,
)


def main():

    chunk = RetrievedChunk(
        paper_id="paper1",
        chunk_id="chunk1",
        page_number=1,
        chunk_index=0,
        char_start=0,
        char_end=100,
        text="Self-RAG is a retrieval framework.",
        score=0.82,
    )

    reranked = RerankedChunk(
        chunk=chunk,
        reranker_score=8.91,
    )

    result = RerankResult(
        query="What is Self-RAG?",
        chunks=[reranked],
        total_chunks=1,
    )

    print(result.model_dump())


if __name__ == "__main__":
    main()