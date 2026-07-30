from __future__ import annotations

from app.core.logger import logger
from app.pipeline.rag_pipeline import RAGPipeline


def main() -> None:
    """
    End-to-end integration test for the RAG pipeline.
    """

    query = "What is Self-RAG?"

    logger.info("=" * 80)
    logger.info("Starting RAG Pipeline Test")
    logger.info("=" * 80)

    pipeline = RAGPipeline()

    result = pipeline.answer(query)

    logger.success("Pipeline execution completed.")

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(query)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result.generation.response.answer)

    print("\n" + "=" * 80)
    print("MODEL")
    print("=" * 80)
    print(result.generation.response.model)

    print("\n" + "=" * 80)
    print("TOKEN USAGE")
    print("=" * 80)
    print(f"Prompt Tokens     : {result.generation.response.prompt_tokens}")
    print(f"Completion Tokens : {result.generation.response.completion_tokens}")
    print(f"Total Tokens      : {result.generation.response.total_tokens}")

    print("\n" + "=" * 80)
    print("CITATIONS")
    print("=" * 80)

    if not result.generation.citations:
        print("No citations.")
    else:
        for i, citation in enumerate(
            result.generation.citations,
            start=1,
        ):
            print(
                f"{i}. "
                f"{citation.paper_id} | "
                f"Page {citation.page_number} | "
                f"Chunk {citation.chunk_id}"
            )

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()