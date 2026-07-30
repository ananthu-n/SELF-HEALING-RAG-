from __future__ import annotations

from app.prompt.formatter import ContextFormatter
from app.prompt.models import PromptInput, PromptResult
from app.prompt.templates import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    NO_CONTEXT_PROMPT,
)
from app.context.models import ContextResult

class PromptBuilder:
    """
    Builds the final prompt for the LLM.
    """

    def __init__(self) -> None:
        self.formatter = ContextFormatter()

    def build(
        self,
        context_result: ContextResult,
    ) -> PromptResult:

        if not context_result.context_chunks:

            prompt = PromptInput(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=NO_CONTEXT_PROMPT,
                context="",
                total_chunks=0,
            )

            return PromptResult(
                prompt=prompt,
                citations=[],
            )

        context, citations = self.formatter.format(
            context_result
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            query=context_result.query,
        )

        prompt = PromptInput(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            context=context,
            total_chunks=len(citations),
        )

        return PromptResult(
            prompt=prompt,
            citations=citations,
        )