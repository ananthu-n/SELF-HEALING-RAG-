from app.prompt.formatter import ContextFormatter
from app.retrieval.models import RetrievedChunk
from app.reranker.models import RerankedChunk
from app.reranker.models import RerankResult


def main():

    chunk1 = RetrievedChunk(
        paper_id="paper1",
        chunk_id="chunk1",
        text="Self-RAG introduces reflection tokens.",
        score=0.82,
        page_number=3,
        chunk_index=0,
        char_start=0,
        char_end=100,
        metadata={}
    )

    chunk2 = RetrievedChunk(
        paper_id="paper2",
        chunk_id="chunk5",
        text="GraphRAG uses knowledge graphs.",
        score=0.75,
        page_number=7,
        chunk_index=1,
        char_start=0,
        char_end=120,
        metadata={}
    )

    result = RerankResult(
        query="Explain Self-RAG.",
        chunks=[
            RerankedChunk(
                chunk=chunk1,
                reranker_score=9.75,
            ),
            RerankedChunk(
                chunk=chunk2,
                reranker_score=8.12,
            ),
        ],
        total_chunks=2,
    )

    formatter = ContextFormatter()

    context, citations = formatter.format(result)

    print("=" * 80)
    print(context)

    print()

    print("=" * 80)
    print("CITATIONS")

    for citation in citations:
        print(citation.model_dump())


if __name__ == "__main__":
    main()