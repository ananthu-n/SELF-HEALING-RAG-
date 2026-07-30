"""
BM25 Index Builder

Responsibilities
----------------
- Load validated chunks
- Tokenize chunk text
- Build BM25 index
- Save BM25 index
- Load BM25 index
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.core.logger import logger
from app.retrieval.tokenizer import BM25Tokenizer


class BM25Index:
    """
    Builds and manages a BM25 index over validated chunks.
    """

    def __init__(
        self,
        chunk_dir: str | Path = "data/validated_chunks",
        index_dir: str | Path = "data/bm25",
    ) -> None:

        self.chunk_dir = Path(chunk_dir)
        self.index_dir = Path(index_dir)

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_dir / "bm25.pkl"
        self.documents_file = self.index_dir / "documents.json"

        self.tokenizer = BM25Tokenizer()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chunk_files(self) -> list[Path]:
        files = sorted(self.chunk_dir.glob("*.json"))

        logger.info(f"Found {len(files)} validated chunk files.")

        return files

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Document Loading
    # ------------------------------------------------------------------

    def load_documents(self) -> list[dict[str, Any]]:
        """
        Load every validated chunk into memory.
        """

        documents: list[dict[str, Any]] = []

        total_chunks = 0

        for file in self._chunk_files():

            chunks = self._load_json(file)

            documents.extend(chunks)

            total_chunks += len(chunks)

        logger.info(f"Loaded {total_chunks} validated chunks.")

        return documents

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    def build_corpus(
        self,
        documents: list[dict[str, Any]],
    ) -> list[list[str]]:
        """
        Convert documents into tokenized corpus.
        """

        logger.info("Tokenizing corpus...")

        corpus = [
            self.tokenizer(doc["text"])
            for doc in documents
        ]

        logger.success("Corpus tokenization completed.")

        return corpus

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """
        Build BM25 index and save it.
        """

        logger.info("Loading documents...")

        documents = self.load_documents()

        if not documents:
            raise RuntimeError("No validated chunks found.")

        corpus = self.build_corpus(documents)

        logger.info("Building BM25 index...")

        bm25 = BM25Okapi(corpus)

        logger.success("BM25 index built successfully.")

        self.save_index(bm25)

        self.save_documents(documents)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_index(self, bm25: BM25Okapi) -> None:
        with self.index_file.open("wb") as f:
            pickle.dump(bm25, f)

        logger.success(f"Saved BM25 index -> {self.index_file}")

    def save_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> None:

        with self.documents_file.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                documents,
                f,
                indent=4,
                ensure_ascii=False,
            )

        logger.success(
            f"Saved document metadata -> {self.documents_file}"
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_index(self) -> BM25Okapi:
        if not self.index_file.exists():
            raise FileNotFoundError(self.index_file)

        with self.index_file.open("rb") as f:
            bm25 = pickle.load(f)

        logger.info("Loaded BM25 index.")

        return bm25

    def load_saved_documents(self) -> list[dict[str, Any]]:
        if not self.documents_file.exists():
            raise FileNotFoundError(self.documents_file)

        with self.documents_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            documents = json.load(f)

        logger.info(
            f"Loaded {len(documents)} indexed documents."
        )

        return documents