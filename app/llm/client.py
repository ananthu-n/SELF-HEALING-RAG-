from __future__ import annotations

import os
from ollama import Client
from openai import OpenAI

from app.core.config import settings
from app.core.logger import logger

from app.llm.models import (
    GenerationRequest,
    GenerationResponse,
)


class OllamaClient:
    """
    Client responsible for communicating with Ollama or Cloud LLM Providers (Groq / OpenAI).
    """

    def __init__(self) -> None:
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        provider_setting = getattr(settings.llm, "provider", "groq").lower()

        if groq_key or provider_setting == "groq":
            self.provider = "groq"
            key = groq_key or os.getenv("GROQ_API_KEY", "")
            self.client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=key if key else "gsk_placeholder",
            )
            self.model = os.getenv("GROQ_MODEL", getattr(settings.llm, "model", "llama-3.3-70b-versatile"))
            logger.info(f"LLM Client initialized with Groq Cloud API (Model: {self.model})")
        elif openai_key or provider_setting == "openai":
            self.provider = "openai"
            self.client = OpenAI(api_key=openai_key or "")
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            logger.info(f"LLM Client initialized with OpenAI API (Model: {self.model})")
        else:
            self.provider = "ollama"
            self.client = Client(host=settings.llm.base_url)
            self.model = settings.llm.model
            logger.info(f"LLM Client initialized with Local Ollama (Host: {settings.llm.base_url}, Model: {self.model})")

        self.timeout = settings.llm.timeout

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:

        logger.info(
            f"Generating response using {self.provider}:{self.model}"
        )

        if self.provider in ("groq", "openai"):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=512,
            )
            answer_text = completion.choices[0].message.content or ""
            prompt_tokens = completion.usage.prompt_tokens if completion.usage else 0
            completion_tokens = completion.usage.completion_tokens if completion.usage else 0
            logger.success("Cloud Generation completed.")
            return GenerationResponse(
                answer=answer_text,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        # Default Ollama Generation
        response = self.client.chat(
            model=self.model,
            messages=request.messages,
            options={
                "temperature": request.temperature,
                "num_predict": 512,
            },
        )

        logger.success("Ollama Generation completed.")

        return GenerationResponse(
            answer=response["message"]["content"],
            model=self.model,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
            total_tokens=(
                (response.get("prompt_eval_count") or 0)
                + (response.get("eval_count") or 0)
            ),
        )