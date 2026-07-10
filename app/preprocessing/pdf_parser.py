
from pathlib import Path
from typing import Any

import pymupdf

from app.core.logger import logger


class PDFParser:
    """Extract text from PDF files."""

    def parse(self, pdf_path: Path) -> dict[str, Any]:
        """
        Extract text from every page of a PDF.

        Parameters
        ----------
        pdf_path : Path
            PDF file path.

        Returns
        -------
        dict
            Parsed paper with page-aware structure.
        """

        document = pymupdf.open(pdf_path)

        pages = []

        for page_number in range(len(document)):
            page = document.load_page(page_number)

            text = page.get_text("text").strip()

            pages.append(
                {
                    "page": page_number + 1,
                    "text": text,
                }
            )

        parsed = {
            "paper_id": pdf_path.stem,
            "num_pages": len(document),
            "pages": pages,
        }

        document.close()

        logger.info(
            f"Parsed {pdf_path.name} ({parsed['num_pages']} pages)"
        )
        return parsed