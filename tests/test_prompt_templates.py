from app.prompt.templates import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)


def main():

    context = """
Paper A:
Self-RAG introduces reflection tokens.

Paper B:
GraphRAG retrieves graph-structured evidence.
"""

    prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        query="Explain Self-RAG."
    )

    print("=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(SYSTEM_PROMPT)

    print()

    print("=" * 80)
    print("USER PROMPT")
    print("=" * 80)
    print(prompt)


if __name__ == "__main__":
    main()