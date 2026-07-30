"""
Tokenizer utilities for BM25 retrieval.
"""

from __future__ import annotations

import re

from app.core.logger import logger


class BM25Tokenizer:
    """
    Lightweight tokenizer for BM25.
    """

    def __init__(self) -> None:
        logger.info("Initialized BM25 tokenizer.")

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = BM25Tokenizer.normalize(text)

        # Extract alphanumeric words
        return re.findall(r"[a-z0-9]+", text)

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)