from __future__ import annotations

import os
import requests
import numpy as np

from app.core.config import settings
from app.core.logger import logger


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer or HuggingFace API based on ENV.
    """

    def __init__(self) -> None:
        self.is_production = os.getenv("ENV", "development").lower() == "production"
        self.hf_token = os.getenv("HF_TOKEN", "")

        if self.is_production:
            logger.info("Initializing EmbeddingModel via Hugging Face API (Production mode)")
            if not self.hf_token:
                logger.warning("HF_TOKEN environment variable is missing! API calls will fail.")
            self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{settings.embedding.model_name}"
        else:
            logger.info("Initializing EmbeddingModel via Local PyTorch (Development mode)")
            import torch
            from sentence_transformers import SentenceTransformer
            
            self.device = self._resolve_device(torch)
            logger.info(f"Using device: {self.device}")
            logger.info(f"Loading embedding model: {settings.embedding.model_name}")
            
            self.model = SentenceTransformer(
                settings.embedding.model_name,
                device=self.device,
            )
            logger.success("Embedding model loaded successfully.")

    def _resolve_device(self, torch_module) -> str:
        if settings.embedding.device.lower() == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        return settings.embedding.device

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        logger.info(f"Encoding {len(texts)} texts.")

        if self.is_production:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            if response.status_code != 200:
                logger.error(f"HF API Error: {response.text}")
                response.raise_for_status()
            embeddings = np.array(response.json())
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=settings.embedding.batch_size,
                normalize_embeddings=settings.embedding.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        logger.success(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings