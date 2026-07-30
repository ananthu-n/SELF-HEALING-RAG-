from app.prompt.models import (
    Citation,
    ContextChunk,
    PromptInput,
    PromptResult,
)


def main():

    chunk = ContextChunk(
        paper_id="paper1",
        chunk_id="chunk10",
        page_number=5,
        score=9.83,
        content="Self-RAG introduces reflection tokens."
    )

    prompt = PromptInput(
        system_prompt="You are a research assistant.",
        user_prompt="Explain Self-RAG.",
        context=chunk.content,
        total_chunks=1
    )

    citation = Citation(
        paper_id="paper1",
        chunk_id="chunk10",
        page_number=5
    )

    result = PromptResult(
        prompt=prompt,
        citations=[citation]
    )

    print(result.model_dump())


if __name__ == "__main__":
    main()