from app.core.config import settings
from app.llm.client import OllamaClient
from app.llm.models import GenerationRequest


def main():

    request = GenerationRequest(
        system_prompt="You are a research assistant.",
        user_prompt="What is Self-RAG?",
        model=settings.llm.model,
        temperature=settings.llm.temperature,
    )

    client = OllamaClient()

    response = client.generate(request)

    print("=" * 80)
    print(response.answer)
    print("=" * 80)

    print(response.model)

    print(response.prompt_tokens)

    print(response.completion_tokens)

    print(response.total_tokens)


if __name__ == "__main__":
    main()