from app.llm.models import (
    GenerationRequest,
    GenerationResponse,
    GenerationResult,
)
from app.prompt.models import (
    Citation,
    PromptInput,
    PromptResult,
)


def main():

    prompt = PromptResult(
        prompt=PromptInput(
            system_prompt="You are a research assistant.",
            user_prompt="Explain Self-RAG.",
            context="Self-RAG introduces reflection tokens.",
            total_chunks=1,
        ),
        citations=[
            Citation(
                paper_id="paper1",
                chunk_id="chunk1",
                page_number=2,
            )
        ],
    )

    request = GenerationRequest(
        prompt=prompt,
    )

    response = GenerationResponse(
        answer="Self-RAG is a retrieval-augmented generation framework that introduces reflection tokens.",
        model="llama3.1",
        citations=prompt.citations,
    )

    result = GenerationResult(
        request=request,
        response=response,
    )

    print(result.model_dump())


if __name__ == "__main__":
    main()