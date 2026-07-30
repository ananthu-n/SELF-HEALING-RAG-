from __future__ import annotations

from app.core.config import settings
from app.core.logger import logger

from app.llm.client import OllamaClient
from app.llm.models import (
    GenerationRequest,
    GenerationResult,
)

from app.prompt.models import PromptResult
from app.prompt.validator import PromptValidator


class LLMGenerator:
    """
    Orchestrates prompt validation and LLM generation.
    """

    def __init__(self) -> None:

        self.validator = PromptValidator()

        self.client = OllamaClient()

    def generate(
        self,
        prompt: PromptResult,
    ) -> GenerationResult:

        logger.info("Validating prompt...")

        validation = self.validator.validate(prompt)

        if not validation.is_valid:
            raise ValueError(validation.reason)

        logger.success("Prompt validation successful.")

        request = GenerationRequest(
            system_prompt=prompt.prompt.system_prompt,
            user_prompt=prompt.prompt.user_prompt,
            model=settings.llm.model,
            temperature=settings.llm.temperature,
        )

        response = self.client.generate(request)

        logger.success("LLM generation completed.")

        return GenerationResult(
            response=response,
            citations=prompt.citations,
        )