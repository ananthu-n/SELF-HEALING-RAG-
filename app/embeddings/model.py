from __future__ import annotations

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logger import logger


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer.
    """

    def __init__(self) -> None:

        self.device = self._resolve_device()

        logger.info(f"Using device: {self.device}")

        logger.info(
            f"Loading embedding model: {settings.embedding.model_name}"
        )

        self.model = SentenceTransformer(
            settings.embedding.model_name,
            device=self.device,
        )

        logger.success("Embedding model loaded successfully.")

    def _resolve_device(self) -> str:

        if settings.embedding.device.lower() == "auto":

            return "cuda" if torch.cuda.is_available() else "cpu"

        return settings.embedding.device

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:

        logger.info(f"Encoding {len(texts)} texts.")

        embeddings = self.model.encode(
            texts,
            batch_size=settings.embedding.batch_size,
            normalize_embeddings=settings.embedding.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        logger.success(
            f"Generated embeddings with shape {embeddings.shape}"
        )

        return embeddings