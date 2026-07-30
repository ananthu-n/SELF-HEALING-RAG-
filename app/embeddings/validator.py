from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.core.logger import logger
from app.embeddings.storage import (
    load_embeddings,
    load_embedding_metadata,
)


class EmbeddingValidator:
    def __init__(self):
        self.expected_dimension = 1024

    def validate(self, paper_id: str) -> bool:
        try:
            embeddings = load_embeddings(paper_id)
            metadata = load_embedding_metadata(paper_id)

        except Exception as e:
            logger.error(f"{paper_id}: failed to load files ({e})")
            return False

        if embeddings.ndim != 2:
            logger.error(f"{paper_id}: embeddings must be 2-dimensional")
            return False

        if embeddings.shape[1] != self.expected_dimension:
            logger.error(
                f"{paper_id}: expected dimension "
                f"{self.expected_dimension}, got {embeddings.shape[1]}"
            )
            return False

        if np.isnan(embeddings).any():
            logger.error(f"{paper_id}: contains NaN values")
            return False

        if np.isinf(embeddings).any():
            logger.error(f"{paper_id}: contains Infinite values")
            return False

        if embeddings.shape[0] != metadata["num_chunks"]:
            logger.error(
                f"{paper_id}: chunk count mismatch "
                f"({embeddings.shape[0]} != {metadata['num_chunks']})"
            )
            return False

        if metadata["dimension"] != self.expected_dimension:
            logger.error(
                f"{paper_id}: metadata dimension mismatch"
            )
            return False

        logger.success(f"{paper_id}: validation passed")

        return True