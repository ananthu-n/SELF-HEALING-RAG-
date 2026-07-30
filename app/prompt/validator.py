from __future__ import annotations

from app.prompt.models import (
    PromptResult,
    PromptValidationResult,
)


class PromptValidator:
    """
    Validates prompts before sending them to the LLM.
    """

    AVG_CHAR_PER_TOKEN = 4

    def validate(
        self,
        prompt_result: PromptResult,
    ) -> PromptValidationResult:

        context = prompt_result.prompt.context

        if not prompt_result.prompt.system_prompt.strip():

            return PromptValidationResult(
                is_valid=False,
                reason="System prompt is empty.",
                total_chunks=0,
                context_length=0,
                estimated_tokens=0,
            )

        if not prompt_result.prompt.user_prompt.strip():

            return PromptValidationResult(
                is_valid=False,
                reason="User prompt is empty.",
                total_chunks=0,
                context_length=0,
                estimated_tokens=0,
            )



        context_length = len(context)

        estimated_tokens = (
            context_length // self.AVG_CHAR_PER_TOKEN
        )

        return PromptValidationResult(
            is_valid=True,
            reason="Prompt validation successful.",
            total_chunks=prompt_result.prompt.total_chunks,
            context_length=context_length,
            estimated_tokens=estimated_tokens,
        )