from __future__ import annotations

import os
import requests
from typing import Sequence

from app.core.config import settings
from app.core.logger import logger


class CrossEncoderModel:
    """
    Singleton wrapper around HuggingFace CrossEncoder or API.
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

        self.is_production = os.getenv("ENV", "development").lower() == "production"
        self.hf_token = os.getenv("HF_TOKEN", "")

        self._load_model()
        self.__class__._initialized = True

    def _load_model(self):
        if self.is_production:
            logger.info("Initializing CrossEncoder via Hugging Face API (Production mode)")
            if not self.hf_token:
                logger.warning("HF_TOKEN missing! API calls will fail.")
            self.api_url = f"https://api-inference.huggingface.co/models/{settings.reranker.model_name}"
        else:
            logger.info("Initializing CrossEncoder via Local PyTorch (Development mode)")
            import torch
            from sentence_transformers import CrossEncoder

            requested_device = settings.reranker.device.lower()
            if requested_device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = requested_device

            logger.info(f"Requested device: {requested_device}")
            logger.info(f"Using device: {device}")
            logger.info(f"Loading CrossEncoder: {settings.reranker.model_name}")

            self.model = CrossEncoder(
                model_name_or_path=settings.reranker.model_name,
                device=device,
                max_length=settings.reranker.max_length,
            )
            self.batch_size = settings.reranker.batch_size
            logger.success("CrossEncoder loaded successfully.")

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[float]:
        """Compute relevance scores."""
        if not pairs:
            return []

        logger.info(f"Scoring {len(pairs)} query-document pairs.")

        if self.is_production:
            query = pairs[0][0]
            docs = [p[1] for p in pairs]
            
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": {"source_sentence": query, "sentences": docs},
                "options": {"wait_for_model": True}
            }
            response = requests.post(self.api_url, headers=headers, json=payload)
            if response.status_code != 200:
                logger.error(f"HF API Reranker Error: {response.text}")
                response.raise_for_status()
            
            # API returns a list of dicts like [{'score': 0.9}, {'score': 0.1}] or just list of floats
            result = response.json()
            if isinstance(result, list) and isinstance(result[0], dict) and 'score' in result[0]:
                return [item['score'] for item in result]
            return result
        else:
            import torch
            with torch.inference_mode():
                scores = self.model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            return scores.tolist()