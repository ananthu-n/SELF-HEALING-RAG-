from __future__ import annotations

from app.core.logger import logger
from app.llm.service import LLMService
from app.pipeline.intent import QueryIntent, IntentDetector
from app.self_healing.healing_models import RewriteStrategy


class QueryRewriter:
    """
    Intelligent Failure-Aware Query Rewriter.

    Generates natural, intent-preserving scientific search queries.
    Strictly forbids comma-separated keyword stuffing.
    """

    def __init__(self) -> None:
        self.llm = LLMService()

    def rewrite(
        self,
        query: str,
        reason: str,
        rewrite_strategy: RewriteStrategy | None = None,
        intent: QueryIntent | None = None,
    ) -> str:
        if intent is None:
            intent = IntentDetector.detect(query)

        logger.info(f"QueryRewriter: Intent={intent.value.upper()} | Strategy={rewrite_strategy} | Reason='{reason}'")

        system_prompt = (
            "You are an expert scientific search query refiner. "
            "Your job is to rewrite a natural language query into a SINGLE natural scientific search phrase. "
            "\n\nSTRICT RULES:\n"
            "1. Output ONLY ONE natural scientific query phrase (max 12 words).\n"
            "2. DO NOT output comma-separated keyword lists or word dumps (e.g. NEVER write 'term1, term2, term3').\n"
            "3. DO NOT output conversational sentences or explanatory claims.\n"
            "4. Preserve the user's intent (definition, comparison, or mechanism explanation).\n"
            "5. DO NOT drift or pivot to unrelated subjects mentioned in the Failure Reason. Keep the search query strictly focused on the core subject of the Original Question."
        )

        user_prompt = (
            f"Original Question: {query}\n"
            f"Query Intent: {intent.value}\n"
            f"Failure Reason: {reason}\n"
            f"Generate a single, refined scientific search query:"
        )

        try:
            response = self.llm.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            rewritten = response.answer.strip()
            # Clean quotes or newlines if any
            rewritten = rewritten.replace('"', '').replace("'", '').split("\n")[0].strip()

            # Safeguard: if LLM returned comma-separated list, clean it
            if "," in rewritten and len(rewritten.split(",")) > 2:
                rewritten = " ".join([term.strip() for term in rewritten.split(",") if term.strip()])

            logger.success(f"Query Rewritten cleanly: '{rewritten}'")
            return rewritten
        except Exception as err:
            logger.warning(f"Query rewriting failed: {err}. Fallback to original query.")
            return query

    def __call__(
        self,
        query: str,
        reason: str,
        rewrite_strategy: RewriteStrategy | None = None,
        intent: QueryIntent | None = None,
    ) -> str:
        return self.rewrite(query=query, reason=reason, rewrite_strategy=rewrite_strategy, intent=intent)