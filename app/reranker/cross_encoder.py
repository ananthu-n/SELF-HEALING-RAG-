from __future__ import annotations

from typing import Sequence

import torch
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logger import logger


class CrossEncoderModel:
    """
    Singleton wrapper around HuggingFace CrossEncoder.

    Responsibilities
    ----------------
    - Load model only once
    - Manage device
    - Run inference
    - Return relevance scores
    """

    _instance = None
    _initialized = False

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):

        if self._initialized:
            return

        self._load_model()

        self.__class__._initialized = True

    def _load_model(self):

        requested_device = settings.reranker.device.lower()

        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = requested_device

        logger.info(f"Requested device: {requested_device}")
        logger.info(f"Using device: {device}")

        logger.info(
            f"Loading CrossEncoder: {settings.reranker.model_name}"
        )

        logger.info(
            f"Using reranker device: {device}"
        )

        self.model = CrossEncoder(
                        model_name_or_path=settings.reranker.model_name,
                        device=device,
                        max_length=settings.reranker.max_length,
                    )
        self.batch_size = settings.reranker.batch_size

        logger.success(
            "CrossEncoder loaded successfully."
        )

    @torch.inference_mode()
    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[float]:
        """
        Compute relevance scores.

        Parameters
        ----------
        pairs
            [(query, document), ...]

        Returns
        -------
        list[float]
        """

        if not pairs:
            return []

        logger.info(
            f"Scoring {len(pairs)} query-document pairs."
        )

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return scores.tolist()