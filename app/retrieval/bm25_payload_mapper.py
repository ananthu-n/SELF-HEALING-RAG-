from __future__ import annotations

from typing import Any

from app.retrieval.models import RetrievedChunk


class BM25PayloadMapper:
    """
    Converts BM25 document dictionaries into RetrievedChunk objects.

    Unlike PayloadMapper, this mapper works on documents loaded from
    data/bm25/documents.json rather than Qdrant ScoredPoint objects.

    This isolates BM25-specific parsing from the retrieval pipeline.
    """

    @staticmethod
    def map(
        document: dict[str, Any],
        score: float,
    ) -> RetrievedChunk:
        """
        Convert one BM25 document into a RetrievedChunk.
        """

        return RetrievedChunk(
            paper_id=document.get("paper_id", ""),
            chunk_id=document.get("chunk_id", ""),
            text=document.get("text", ""),
            score=float(score),
            page_number=document.get("page_number", 0),
            chunk_index=document.get("chunk_index", 0),
            char_start=document.get("char_start", 0),
            char_end=document.get("char_end", 0),
            metadata={
                key: value
                for key, value in document.items()
                if key not in {
                    "paper_id",
                    "chunk_id",
                    "text",
                    "page_number",
                    "chunk_index",
                    "char_start",
                    "char_end",
                }
            },
        )

    @classmethod
    def map_many(
        cls,
        documents: list[dict[str, Any]],
        scores: list[float],
    ) -> list[RetrievedChunk]:
        """
        Convert multiple BM25 documents into RetrievedChunk objects.

        Parameters
        ----------
        documents
            Ranked BM25 documents.

        scores
            BM25 scores corresponding to each document.

        Returns
        -------
        list[RetrievedChunk]
        """

        if len(documents) != len(scores):
            raise ValueError(
                "documents and scores must have identical lengths."
            )

        return [
            cls.map(document, score)
            for document, score in zip(documents, scores)
        ]