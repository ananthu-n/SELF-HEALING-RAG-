
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.logger import logger
from app.retrieval.bm25_index import BM25Index
from app.retrieval.tokenizer import BM25Tokenizer


@dataclass(slots=True)
class BM25SearchResult:
    """
    Represents one BM25 search result before payload mapping.
    """

    document: dict
    score: float


class BM25SearchService:
    """
    Executes sparse lexical search using BM25.

    Responsibilities
    ----------------
    • Load BM25 index
    • Tokenize query
    • Execute BM25 search
    • Return ranked documents

    Does NOT:
        - map payloads
        - validate results
        - rerank
        - filter
    """

    def __init__(self) -> None:

        logger.info("Loading BM25 search service...")

        index = BM25Index()

        self.bm25 = index.load_index()

        self.documents = index.load_saved_documents()

        self.tokenizer = BM25Tokenizer()

        logger.success(
            f"Loaded BM25 search service ({len(self.documents)} documents)."
        )

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[BM25SearchResult]:
        """
        Execute BM25 search.

        Parameters
        ----------
        query
            User query.

        top_k
            Number of candidates.

        Returns
        -------
        list[BM25SearchResult]
        """

        logger.info(f"BM25 search started: '{query}'")

        query_tokens = self.tokenizer(query)

        scores = self.bm25.get_scores(query_tokens)

        if len(scores) == 0:

            logger.warning("BM25 returned no candidates.")

            return []

        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            BM25SearchResult(
                document=self.documents[int(index)],
                score=float(scores[int(index)]),
            )
            for index in ranked_indices
        ]

        logger.success(
            f"Retrieved {len(results)} BM25 candidates."
        )

        return results