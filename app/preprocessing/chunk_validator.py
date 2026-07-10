from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logger import logger
from app.preprocessing.validation_report import ValidationReport


class ChunkValidator:
    """
    Validates generated document chunks before embedding generation.
    """

    def __init__(self) -> None:

        self.input_dir = Path(settings.storage.chunks_dir)

        self.output_dir = self.input_dir.parent / "validated_chunks"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.report = ValidationReport()
        self.required_fields = {
            "chunk_id",
            "paper_id",
            "page_number",
            "chunk_index",
            "char_start",
            "char_end",
            "text",
        }

    def validate_chunk(self, chunk: dict[str, Any]) -> bool:
        """
        Validate a single chunk.

        Returns
        -------
        bool
            True if the chunk is valid.
        """

        self.report.chunks_processed += 1

        # -----------------------------
        # Required metadata
        # -----------------------------
        missing = self.required_fields - chunk.keys()

        if missing:
            self.report.missing_metadata += 1
            self.report.rejected_chunks += 1
            return False

        text = chunk["text"]

        # -----------------------------
        # Empty text
        # -----------------------------
        if text == "":
            self.report.empty_chunks += 1
            self.report.rejected_chunks += 1
            return False

        # -----------------------------
        # Whitespace only
        # -----------------------------
        if not text.strip():
            self.report.whitespace_chunks += 1
            self.report.rejected_chunks += 1
            return False

        # -----------------------------
        # Very short chunk
        # -----------------------------
        if len(text.strip()) < 50:
            self.report.short_chunks += 1
            self.report.rejected_chunks += 1
            return False

        # -----------------------------
        # Character offsets
        # -----------------------------
        char_start = chunk["char_start"]
        char_end = chunk["char_end"]

        if char_start < 0 or char_start >= char_end:
            self.report.invalid_offsets += 1
            self.report.rejected_chunks += 1
            return False

        # -----------------------------
        # Page number
        # -----------------------------
        if chunk["page_number"] <= 0:
            self.report.invalid_pages += 1
            self.report.rejected_chunks += 1
            return False

        self.report.valid_chunks += 1

        return True

    def load_chunks(self, file_path: Path) -> list[dict[str, Any]]:
        """
        Load chunks from a JSON file.

        Parameters
        ----------
        file_path : Path

        Returns
        -------
        list[dict]
        """

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
        

    def validate_file(self, file_path: Path) -> None:
        """
        Validate a single chunk file.

        Parameters
        ----------
        file_path : Path
        """

        logger.info(f"Validating {file_path.name}")

        self.report.files_processed += 1

        chunks = self.load_chunks(file_path)

        valid_chunks: list[dict[str, Any]] = []

        for chunk in chunks:

            if self.validate_chunk(chunk):
                valid_chunks.append(chunk)

        paper_id = file_path.stem

        self.save_valid_chunks(
            paper_id=paper_id,
            valid_chunks=valid_chunks,
        )

        logger.success(
            f"{paper_id}: "
            f"{len(valid_chunks)}/{len(chunks)} chunks valid"
        )

    def validate_corpus(self) -> None:
        """
        Validate all chunk files.
        """

        chunk_files = sorted(self.input_dir.glob("*.json"))

        logger.info(f"Found {len(chunk_files)} chunk files.")

        for file_path in chunk_files:

            try:
                self.validate_file(file_path)

            except Exception as exc:

                self.report.failed_files += 1

                logger.exception(
                    f"Failed to validate {file_path.name}: {exc}"
                )

        self.generate_report()

    def save_valid_chunks(
        self,
        paper_id: str,
        valid_chunks: list[dict[str, Any]]
    ) -> None:
        """
        Save validated chunks.
        """
        output_file = self.output_dir / f"{paper_id}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(valid_chunks, f, indent=2, ensure_ascii=False)

    def generate_report(self) -> None:
        """
        Print validation statistics.
        """

        logger.info("=" * 50)

        logger.info("Validation Summary")

        logger.info("=" * 50)

        for key, value in self.report.to_dict().items():
            logger.info(f"{key:25}: {value}")

        logger.info(f"Output directory : {self.output_dir}")