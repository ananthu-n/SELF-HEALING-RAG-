from __future__ import annotations

import re


class TopicExtractor:
    """
    Extracts clean research topics and entity terms from natural language queries.
    Prevents natural language questions from polluting arXiv and retriever searches.
    """

    STOP_PATTERNS = [
        r"^what\s+is\s+a\s+",
        r"^what\s+is\s+an\s+",
        r"^what\s+is\s+",
        r"^what\s+are\s+",
        r"^how\s+does\s+",
        r"^how\s+do\s+",
        r"^explain\s+",
        r"^tell\s+me\s+about\s+",
        r"^describe\s+",
        r"\?$",
    ]

    @classmethod
    def extract_topic(cls, query: str) -> str:
        """
        Strip conversational filler and return core research topic keywords.
        """
        clean_query = query.strip()
        for pattern in cls.STOP_PATTERNS:
            clean_query = re.sub(pattern, "", clean_query, flags=re.IGNORECASE).strip()

        # If stripping removed everything, fallback to original query
        return clean_query if clean_query else query.strip()
