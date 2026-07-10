import json
from pathlib import Path

from app.core.config import settings
from app.core.logger import logger
from app.preprocessing.pdf_parser import PDFParser


def main():

    parser = PDFParser()

    pdf_dir = Path(settings.storage.raw_pdf_dir)

    processed_dir = Path(settings.storage.processed_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    logger.info(f"Found {len(pdf_files)} PDF files.")

    for pdf in pdf_files:

        logger.info(f"Processing {pdf.name}")

        result = parser.parse(pdf)

        output = {
            "paper_id": result["paper_id"],
            "num_pages": result["num_pages"],
            "pages": result["pages"],
        }

        output_file = processed_dir / f"{pdf.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:

            json.dump(
                output,
                f,
                indent=4,
                ensure_ascii=False
            )

        logger.success(f"Saved {output_file.name}")


if __name__ == "__main__":
    main()