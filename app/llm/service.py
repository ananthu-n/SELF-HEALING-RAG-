from __future__ import annotations

from app.core.config import settings
from app.core.logger import logger

from app.llm.client import OllamaClient
from app.llm.models import (
    GenerationRequest,
    GenerationResponse,
)


class LLMService:
    """
    Generic LLM interface.

    This service is intended for modules that need
    direct LLM reasoning without going through the
    RAG PromptBuilder.

    Examples:
        - Query Rewriter
        - HyDE
        - Reflection
        - CRAG Judge
        - Context Distillation
    """

    def __init__(self) -> None:

        self.client = OllamaClient()

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> GenerationResponse:

        logger.info("Sending request to LLM...")

        request = GenerationRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=settings.llm.model,
            temperature=temperature,
        )

        response = self.client.generate(request)

        logger.success("LLM request completed.")

        return response