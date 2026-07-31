import json
import os
import re
from app.llm.client import OllamaClient
from app.llm.models import GenerationRequest

from app.core.config import settings
from app.core.logger import logger

from app.evaluation.models import (
    GroundingRequest,
    GroundingResponse,
    GroundingResult,
)
from app.evaluation.prompts import GROUNDING_SYSTEM_PROMPT


class GroundingEvaluator:
    """
    Evaluates whether a generated answer is grounded in the retrieved context.
    Supports local Ollama or Cloud LLM Providers (Groq / OpenAI).
    """

    def __init__(self) -> None:
        self.client = OllamaClient()

    def evaluate(
        self,
        request: GroundingRequest,
    ) -> GroundingResult:
        """
        Evaluate grounding of a generated answer.
        """
        if request.answer == "Insufficient context/retrieval quality.":
            logger.warning("Bypassing grounding evaluation due to early-exit retrieval validation failure.")
            return GroundingResult(
                response=GroundingResponse(
                    is_grounded=False,
                    confidence=0.0,
                    unsupported_claims=[],
                    should_retry=True,
                    reason="Retrieval validation failed: insufficient or irrelevant context retrieved.",
                )
            )

        user_prompt = f"""
Question

{request.query}

--------------------------------

Retrieved Context

{request.context}

--------------------------------

Generated Answer

{request.answer}
"""

        logger.info("Running grounding evaluation...")

        _default_model = os.getenv("GROQ_MODEL", getattr(settings.llm, "model", "llama-3.3-70b-versatile"))
        gen_req = GenerationRequest(
            system_prompt=GROUNDING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=_default_model,
            temperature=0.0,
        )

        gen_res = self.client.generate(gen_req)
        raw_content = gen_res.answer

        def extract_json(text: str) -> dict | None:
            text = text.strip()

            # Remove markdown code fences if present
            if text.startswith("```"):
                lines = text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

            # Attempt 1: Direct JSON load
            try:
                return json.loads(text)
            except Exception:
                pass

            # Attempt 2: Outer braces
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                candidate = text[start : end + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    pass

                # Attempt 3: Clean trailing commas
                try:
                    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
                    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
                    cleaned = re.sub(r"\bFalse\b", "false", cleaned)
                    cleaned = re.sub(r"\bNone\b", "null", cleaned)
                    return json.loads(cleaned)
                except Exception:
                    pass

                # Attempt 4: AST evaluation
                try:
                    import ast
                    val = ast.literal_eval(candidate)
                    if isinstance(val, dict):
                        return val
                except Exception:
                    pass

            # Attempt 5: Regex extraction fallback for truncated/malformed JSON
            try:
                is_grounded = bool(re.search(r'"is_grounded"\s*:\s*true', text, re.IGNORECASE))
                conf_match = re.search(r'"confidence"\s*:\s*([0-9\.]+)', text)
                confidence = float(conf_match.group(1)) if conf_match else 0.5
                reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
                reason = reason_match.group(1) if reason_match else "Extracted grounding result from LLM output."
                return {
                    "is_grounded": is_grounded,
                    "confidence": confidence,
                    "unsupported_claims": [],
                    "should_retry": not is_grounded,
                    "reason": reason
                }
            except Exception:
                pass

            return None

        result = extract_json(raw_content)

        if result is None:
            logger.error(f"Failed to parse grounding response JSON. Raw output: {raw_content}")
            grounding_response = GroundingResponse(
                is_grounded=False,
                confidence=0.0,
                unsupported_claims=[],
                should_retry=True,
                reason=f"Grounding evaluator JSON parse failure. Raw: {raw_content[:150]}"
            )
        else:
            try:
                grounding_response = GroundingResponse(**result)
            except Exception as e:
                logger.error(f"Failed to validate GroundingResponse schema: {e}. Result: {result}")
                grounding_response = GroundingResponse(
                    is_grounded=False,
                    confidence=0.0,
                    unsupported_claims=[],
                    should_retry=True,
                    reason=f"Grounding validation schema error: {str(e)}"
                )

        logger.success("Grounding evaluation completed.")

        return GroundingResult(
            response=grounding_response,
        )