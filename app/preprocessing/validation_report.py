from dataclasses import dataclass


@dataclass
class ValidationReport:
    """Stores validation statistics for a validation run."""

    files_processed: int = 0

    chunks_processed: int = 0

    valid_chunks: int = 0

    rejected_chunks: int = 0

    empty_chunks: int = 0

    whitespace_chunks: int = 0

    short_chunks: int = 0

    duplicate_chunks: int = 0

    missing_metadata: int = 0

    invalid_offsets: int = 0

    invalid_pages: int = 0

    duplicate_ids: int = 0

    failed_files: int = 0

    def to_dict(self) -> dict:
        return {
            "files_processed": self.files_processed,
            "failed_files": self.failed_files,
            "chunks_processed": self.chunks_processed,
            "valid_chunks": self.valid_chunks,
            "rejected_chunks": self.rejected_chunks,
            "empty_chunks": self.empty_chunks,
            "whitespace_chunks": self.whitespace_chunks,
            "short_chunks": self.short_chunks,
            "duplicate_chunks": self.duplicate_chunks,
            "missing_metadata": self.missing_metadata,
            "invalid_offsets": self.invalid_offsets,
            "invalid_pages": self.invalid_pages,
            "duplicate_ids": self.duplicate_ids,
        }