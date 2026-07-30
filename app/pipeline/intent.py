from __future__ import annotations

from enum import Enum
from app.core.logger import logger


class QueryIntent(str, Enum):
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    LITERATURE_REVIEW = "literature_review"
    FACT_LOOKUP = "fact_lookup"
    RESEARCH_QUESTION = "research_question"


class IntentDetector:
    """
    Classifies user queries into distinct research intents and applies HyDE gating rules.
    """

    @classmethod
    def detect(cls, query: str) -> QueryIntent:
        q_lower = query.lower().strip()

        # Comparison intent
        if any(kw in q_lower for kw in [" vs ", "compare", "versus", "difference between"]):
            return QueryIntent.COMPARISON

        # Definition intent
        if q_lower.startswith(("what is", "what are", "define ", "meaning of")):
            return QueryIntent.DEFINITION

        # Literature review intent
        if any(kw in q_lower for kw in ["literature review", "overview of papers", "survey of", "recent work on"]):
            return QueryIntent.LITERATURE_REVIEW

        # Fact lookup intent
        if q_lower.startswith(("who invented", "when was", "which dataset", "which model achieved")):
            return QueryIntent.FACT_LOOKUP

        # Explanation intent
        if q_lower.startswith(("how does", "how do", "explain ", "why does")):
            return QueryIntent.EXPLANATION

        # Default fallback: Exploratory Research Question
        return QueryIntent.RESEARCH_QUESTION

    @classmethod
    def should_use_hyde(cls, intent: QueryIntent) -> bool:
        """
        HyDE Gating Rule:
        - Only use HyDE for exploratory/research questions and explanations.
        - Skip HyDE for definitions, comparisons, literature reviews, and factual lookups.
        """
        use_hyde = intent in (QueryIntent.EXPLANATION, QueryIntent.RESEARCH_QUESTION)
        logger.info(f"HyDE Gating check for intent '{intent.value}': use_hyde={use_hyde}")
        return use_hyde
