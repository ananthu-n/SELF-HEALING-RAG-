from __future__ import annotations

from app.core.config import settings
from app.prompt.models import Citation
from app.context.models import ContextResult

class ContextFormatter:
    """
    Formats reranked chunks into a prompt-ready context.
    """


    def __init__(self):

        self.max_chunks = settings.prompt.max_chunks

        self.max_context_characters = (
            settings.prompt.max_context_characters
        )

        self.include_metadata = (
            settings.prompt.include_metadata
        )

    def format(
        self,
        context_result: ContextResult,
    ) -> tuple[str, list[Citation]]:

        context_blocks: list[str] = []

        citations: list[Citation] = []

        for rank, chunk in enumerate(
            context_result.context_chunks[: self.max_chunks],
            start=1,
        ):

            context_blocks.append(
                f"""[Paper {chunk.paper_id} | Page {chunk.page_number}]
{chunk.text.strip()}""".strip()
            )
            

            citations.append(
                Citation(
                    paper_id=chunk.paper_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                )
            )

        if not context_blocks:
            return "", []

        separator = "\n\n" + ("=" * 80) + "\n\n"

        context = separator.join(context_blocks)

        return context, citations