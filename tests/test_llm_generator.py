from app.llm.generator import LLMGenerator
from app.prompt.models import (
    Citation,
    PromptInput,
    PromptResult,
)


def main():

    prompt = PromptResult(
        prompt=PromptInput(
            system_prompt="You are a research assistant.",
            user_prompt="What is Self-RAG?",
            context="Self-RAG is a retrieval-augmented generation framework that introduces reflection tokens for self-correction.",
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

    generator = LLMGenerator()

    result = generator.generate(prompt)

    print("\n")
    print("=" * 100)
    print("ANSWER")
    print("=" * 100)

    print(result.response.answer)

    print("\n")
    print("=" * 100)
    print("CITATIONS")
    print("=" * 100)

    for citation in result.citations:
        print(citation.model_dump())


if __name__ == "__main__":
    main()