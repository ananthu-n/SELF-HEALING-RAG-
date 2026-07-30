from __future__ import annotations

from pydantic import BaseModel, Field

from app.prompt.models import Citation


class GenerationRequest(BaseModel):
    """
    Request sent to the LLM.
    """

    system_prompt: str
    user_prompt: str
    model: str
    temperature: float

    @property
    def messages(self) -> list[dict[str, str]]:
        """
        Convert the request into a chat format.
        """

        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self.user_prompt,
            },
        ]


class GenerationResponse(BaseModel):
    """
    Raw response from the LLM.
    """

    answer: str

    model: str

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class GenerationResult(BaseModel):
    """
    Final output of the generation stage.
    """

    response: GenerationResponse

    citations: list[Citation] = Field(default_factory=list)