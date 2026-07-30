from __future__ import annotations

from pydantic import BaseModel, Field




class PromptInput(BaseModel):
    """
    Final prompt sent to the LLM.

    The prompt is stored as separate system and user prompts.
    This keeps it compatible with chat-based LLM APIs such as
    Ollama, OpenAI, Anthropic and others.
    """

    system_prompt: str

    user_prompt: str

    context: str

    total_chunks: int

    @property
    def messages(self) -> list[dict[str, str]]:
        """
        Convert the prompt into a chat message format.
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


class Citation(BaseModel):
    """
    Citation information returned alongside the answer.
    """

    paper_id: str

    chunk_id: str

    page_number: int


class PromptResult(BaseModel):
    """
    Output of the Prompt Builder.
    """

    prompt: PromptInput

    citations: list[Citation] = Field(default_factory=list)

class PromptValidationResult(BaseModel):

    is_valid: bool

    reason: str

    total_chunks: int

    context_length: int

    estimated_tokens: int