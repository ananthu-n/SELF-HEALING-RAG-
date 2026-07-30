from __future__ import annotations

from typing import NamedTuple
from app.core.logger import logger
from app.llm.service import LLMService
from app.ingestion.topic_extractor import TopicExtractor
from app.pipeline.intent import QueryIntent, IntentDetector


class EnhancedQuery(NamedTuple):
    original_query: str
    intent: QueryIntent
    search_query: str
    extracted_topic: str
    hypothetical_document: str | None
    sub_queries: list[str]


class QueryEnhancer:
    """
    Production Intent-Aware Query Processing Layer.

    Routes queries intelligently based on detected intent:
    - DEFINITION: Preserves semantic context ('vector embedding definition') instead of bare keyword stuffing.
    - COMPARISON: Triggers query decomposition.
    - EXPLANATION / RESEARCH_QUESTION: Triggers HyDE gating and abstract generation.
    - LITERATURE_REVIEW / FACT_LOOKUP: Preserves structured research query parameters.
    """

    def __init__(self, enable_hyde: bool = True, enable_decomposition: bool = True) -> None:
        self.enable_hyde = enable_hyde
        self.enable_decomposition = enable_decomposition
        self.llm = LLMService()

    def enhance(self, query: str) -> EnhancedQuery:
        intent = IntentDetector.detect(query)
        logger.info(f"Query Processing: Query='{query}' | Detected Intent={intent.value.upper()}")

        extracted_topic = TopicExtractor.extract_topic(query)
        hypothetical_doc = None
        sub_queries = [query]

        # Check HyDE Gating
        use_hyde = self.enable_hyde and IntentDetector.should_use_hyde(intent)

        # Query Routing based on Intent
        if intent == QueryIntent.DEFINITION:
            search_query = f"{extracted_topic} definition"

        elif intent == QueryIntent.COMPARISON:
            search_query = query
            if self.enable_decomposition:
                try:
                    system_prompt = (
                        "You are a query decomposer. "
                        "Break comparison questions into 2-3 atomic search queries. "
                        "Output ONLY sub-queries separated by newlines."
                    )
                    response = self.llm.complete(
                        system_prompt=system_prompt,
                        user_prompt=f"Decompose: {query}",
                        temperature=0.0,
                    )
                    lines = [line.strip() for line in response.answer.split("\n") if line.strip()]
                    if len(lines) > 1:
                        sub_queries = lines
                        logger.info(f"Comparison Decomposed: {sub_queries}")
                except Exception as err:
                    logger.warning(f"Decomposition error: {err}")

        elif intent in (QueryIntent.EXPLANATION, QueryIntent.RESEARCH_QUESTION):
            search_query = query
            if use_hyde:
                try:
                    system_prompt = (
                        "Write a short hypothetical paragraph (2-3 sentences) from a scientific research paper "
                        "that explains the mechanisms answering the query."
                    )
                    response = self.llm.complete(
                        system_prompt=system_prompt,
                        user_prompt=f"Generate hypothetical abstract for: {query}",
                        temperature=0.3,
                    )
                    hypothetical_doc = response.answer.strip()
                    logger.info(f"HyDE Abstract Generated: {hypothetical_doc[:100]}...")
                except Exception as err:
                    logger.warning(f"HyDE error: {err}")
        else:
            search_query = query

        return EnhancedQuery(
            original_query=query,
            intent=intent,
            search_query=search_query,
            extracted_topic=extracted_topic,
            hypothetical_document=hypothetical_doc,
            sub_queries=sub_queries,
        )
