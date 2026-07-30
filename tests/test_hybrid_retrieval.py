"""
Test Hybrid Retrieval Pipeline

This script validates the complete retrieval pipeline:

User Query
      │
      ▼
Dense Retriever
      │
      ▼
BM25 Retriever
      │
      ▼
Reciprocal Rank Fusion
      │
      ▼
Hybrid Results
"""

from __future__ import annotations

import sys

from app.core.logger import logger
from app.retrieval.hybrid_retriever import HybridRetriever


def print_results(title: str, results: list[dict]) -> None:
    """
    Pretty-print retrieval results.
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info(title)
    logger.info("=" * 80)

    if not results:
        logger.warning("No results found.")
        return

    for result in results:

        logger.info(f"Rank        : {result.get('rank')}")
        logger.info(f"Retriever   : {result.get('retriever')}")
        logger.info(f"Paper ID    : {result.get('paper_id')}")
        logger.info(f"Chunk ID    : {result.get('chunk_id')}")

        if "score" in result:
            logger.info(f"Score       : {result['score']:.4f}")

        if "rrf_score" in result:
            logger.info(f"RRF Score   : {result['rrf_score']:.6f}")

        text = result.get("text", "")
        preview = text.replace("\n", " ")

        if len(preview) > 250:
            preview = preview[:250] + "..."

        logger.info(f"Preview     : {preview}")

        logger.info("-" * 80)


def main() -> int:

    logger.info("=" * 80)
    logger.info("HYBRID RETRIEVAL TEST")
    logger.info("=" * 80)

    try:

        retriever = HybridRetriever()

        query = input("\nEnter query: ").strip()

        if not query:
            logger.warning("Query cannot be empty.")
            return 1

        logger.info(f"Query: {query}")

        outputs = retriever.retrieve_all(
            query=query,
            top_k=5,
        )

        print_results(
            "DENSE RETRIEVAL",
            outputs["dense"],
        )

        print_results(
            "BM25 RETRIEVAL",
            outputs["bm25"],
        )

        print_results(
            "HYBRID RETRIEVAL",
            outputs["hybrid"],
        )

        logger.success("Hybrid retrieval test completed.")

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 1

    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())