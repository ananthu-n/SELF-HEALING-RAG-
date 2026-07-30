from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings
from app.core.logger import logger
from app.embeddings.model import EmbeddingModel
from app.embeddings.storage import (
    save_embeddings,
    save_embedding_metadata,
)


class EmbeddingGenerator:
    """
    Generates embeddings for validated chunks.

    Pipeline

    validated_chunks JSON
            ↓
        extract text
            ↓
    EmbeddingModel.encode()
            ↓
       save .npy vectors
            ↓
      save metadata json
    """

    def __init__(self) -> None:
        self.model = EmbeddingModel()

        self.chunk_dir = Path("data/validated_chunks")

        self.embedding_dir = settings.embedding.embeddings_dir

        self.metadata_dir = settings.embedding.metadata_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_chunk_files(self) -> list[Path]:
        files = sorted(self.chunk_dir.glob("*.json"))

        logger.info(f"Found {len(files)} validated chunk files.")

        return files

    def _load_chunks(self, file_path: Path) -> list[dict]:
        with file_path.open("r", encoding="utf-8") as f:
            chunks = json.load(f)

        return chunks

    @staticmethod
    def _extract_texts(chunks: list[dict]) -> list[str]:
        return [chunk["text"] for chunk in chunks]

    @staticmethod
    def _paper_id(file_path: Path) -> str:
        return file_path.stem

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_paper(self, file_path: Path) -> None:
        paper_id = self._paper_id(file_path)

        embedding_file = self.embedding_dir / f"{paper_id}.npy"

        if embedding_file.exists():
            logger.info(f"Skipping {paper_id} (already embedded).")
            return

        logger.info(f"Processing paper: {paper_id}")

        chunks = self._load_chunks(file_path)

        texts = self._extract_texts(chunks)

        logger.info(f"Loaded {len(texts)} chunks.")

        embeddings = self.model.encode(texts)

        save_embeddings(
            paper_id=paper_id,
            embeddings=embeddings,
        )

        metadata = {
            "paper_id": paper_id,
            "model_name": settings.embedding.model_name,
            "dimension": int(embeddings.shape[1]),
            "num_chunks": int(embeddings.shape[0]),
            "normalized": settings.embedding.normalize,
            "dtype": str(embeddings.dtype),
        }

        save_embedding_metadata(
            paper_id=paper_id,
            metadata=metadata,
        )

        logger.success(f"Finished {paper_id}")

    # ------------------------------------------------------------------
    # Main Entry
    # ------------------------------------------------------------------

    def process_all(self) -> None:
        files = self._get_chunk_files()

        if not files:
            logger.warning("No validated chunk files found.")
            return

        for file_path in files:
            try:
                self.process_paper(file_path)

            except Exception as e:
                logger.exception(
                    f"Failed processing {file_path.name}: {e}"
                )

        logger.success("Embedding generation completed.")