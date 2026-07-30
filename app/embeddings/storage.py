"""
Storage utilities for embedding vectors and embedding metadata.

Responsibilities:
- Save embeddings (.npy)
- Load embeddings (.npy)
- Save metadata (.json)
- Load metadata (.json)

This module does NOT generate embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logger import logger


# ---------------------------------------------------------------------
# Internal Paths
# ---------------------------------------------------------------------

EMBEDDINGS_DIR = settings.embedding.embeddings_dir
METADATA_DIR = settings.embedding.metadata_dir

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Embedding Storage
# ---------------------------------------------------------------------

def save_embeddings(paper_id: str, embeddings: np.ndarray) -> Path:
    """
    Save embedding vectors as a NumPy file.

    Args:
        paper_id: arXiv paper id
        embeddings: Shape (num_chunks, embedding_dim)

    Returns:
        Path to saved file.
    """

    output_path = EMBEDDINGS_DIR / f"{paper_id}.npy"

    np.save(output_path, embeddings)

    logger.info(f"Saved embeddings: {output_path}")

    return output_path


def load_embeddings(paper_id: str) -> np.ndarray:
    """
    Load embedding vectors for a paper.
    """

    input_path = EMBEDDINGS_DIR / f"{paper_id}.npy"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    logger.info(f"Loading embeddings: {input_path}")

    return np.load(input_path)


# ---------------------------------------------------------------------
# Metadata Storage
# ---------------------------------------------------------------------

def save_embedding_metadata(
    paper_id: str,
    metadata: dict[str, Any],
) -> Path:
    """
    Save embedding metadata as JSON.
    """

    output_path = METADATA_DIR / f"{paper_id}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved embedding metadata: {output_path}")

    return output_path


def load_embedding_metadata(
    paper_id: str,
) -> dict[str, Any]:
    """
    Load embedding metadata.
    """

    input_path = METADATA_DIR / f"{paper_id}.json"

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"Loaded embedding metadata: {input_path}")

    return metadata