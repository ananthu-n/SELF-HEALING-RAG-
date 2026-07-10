from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.core.logger import logger


class DocumentChunker:
    """Split processed documents into retrieval-ready chunks."""

    def __init__(self) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
        )

    def chunk_document(self, document: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Split one processed paper into chunks.

        Parameters
        ----------
        document
            Parsed paper JSON.

        Returns
        -------
        list
            Retrieval-ready chunks.
        """

        paper_id = document["paper_id"]

        chunks: list[dict[str, Any]] = []

        chunk_index = 0

        for page in document["pages"]:

            page_number = page["page"]

            page_text = page["text"]

            if not page_text.strip():
                continue

            split_chunks = self.splitter.split_text(page_text)

            cursor = 0

            for chunk in split_chunks:

                start = page_text.find(chunk, cursor)

                if start == -1:
                    start = cursor

                end = start + len(chunk)

                cursor = end

                chunks.append(
                    {
                        "chunk_id": f"{paper_id}_p{page_number}_c{chunk_index}",
                        "paper_id": paper_id,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "char_start": start,
                        "char_end": end,
                        "text": chunk,
                    }
                )

                chunk_index += 1

        logger.info(
            f"Created {len(chunks)} chunks for {paper_id}"
        )

        return chunks