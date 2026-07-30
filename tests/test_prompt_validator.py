from app.prompt.builder import PromptBuilder
from app.prompt.validator import PromptValidator
from app.retrieval.models import RetrievedChunk
from app.reranker.models import (
    RerankedChunk,
    RerankResult,
)


def main():

    chunk = RetrievedChunk(
        paper_id="paper1",
        chunk_id="chunk1",
        text="Self-RAG improves factual grounding.",
        score=0.84,
        page_number=1,
        chunk_index=0,
        char_start=0,
        char_end=100,
        metadata={},
    )

    rerank = RerankResult(
        query="What is Self-RAG?",
        chunks=[
            RerankedChunk(
                chunk=chunk,
                reranker_score=9.2,
            )
        ],
        total_chunks=1,
    )

    builder = PromptBuilder()

    prompt = builder.build(rerank)

    validator = PromptValidator()

    result = validator.validate(prompt)

    print(result.model_dump())
    

if __name__ == "__main__":
    main()