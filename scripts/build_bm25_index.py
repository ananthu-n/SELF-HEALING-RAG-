"""
Build BM25 Index

Reads all validated chunks and builds a persistent BM25 index.

Run:

    python -m scripts.build_bm25_index
"""

from __future__ import annotations

import sys

from app.core.logger import logger
from app.retrieval.bm25_index import BM25Index


def main() -> int:
    """
    Build and persist the BM25 index.
    """

    logger.info("=" * 60)
    logger.info("BM25 INDEX BUILDER")
    logger.info("=" * 60)

    try:
        builder = BM25Index()

        builder.build()

        logger.success("=" * 60)
        logger.success("BM25 index successfully built.")
        logger.success("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        return 1

    except Exception as e:
        logger.exception(f"BM25 index build failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())