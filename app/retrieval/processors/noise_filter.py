from __future__ import annotations

import re

from app.core.logger import logger
from app.retrieval.models import RetrievedChunk
from app.retrieval.processors.base import RetrievalProcessor


class NoiseFilter(RetrievalProcessor):
    """
    Removes low-value retrievals such as references,
    bibliographies, URL-heavy chunks, and citation lists.
    """

    KEYWORDS = (
        "references",
        "bibliography",
        "acknowledgements",
        "appendix",
        "table of contents",
        "copyright",
    )

    URL_PATTERN = re.compile(
        r"(https?://|www\.|github|youtube|doi\.org|arxiv\.org)",
        re.IGNORECASE,
    )

    CITATION_PATTERN = re.compile(
        r"\[\d+\]|\(\d{4}\)|et al\.",
        re.IGNORECASE,
    )

    AVAILABLE_PATTERN = re.compile(
        r"available\s*:",
        re.IGNORECASE,
    )

    MIN_TEXT_LENGTH = 120
    MAX_URLS = 2
    MAX_CITATIONS = 8
    MIN_ALPHA_RATIO = 0.45

    def process(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:

        cleaned = []

        for chunk in chunks:

            text = chunk.text.strip()

            if len(text) < self.MIN_TEXT_LENGTH:
                continue

            alpha = sum(c.isalpha() for c in text)

            if alpha / max(len(text), 1) < self.MIN_ALPHA_RATIO:
                continue

            lower = text.lower()

            if any(word in lower for word in self.KEYWORDS):
                continue

            if len(self.URL_PATTERN.findall(text)) > self.MAX_URLS:
                continue

            if len(self.CITATION_PATTERN.findall(text)) > self.MAX_CITATIONS:
                continue

            if self.AVAILABLE_PATTERN.search(text):
                continue

            cleaned.append(chunk)

        logger.info(
            f"NoiseFilter: {len(cleaned)} chunks remain."
        )

        return cleaned