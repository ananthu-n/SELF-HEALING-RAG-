import json
from pathlib import Path

from app.core.logger import logger
from app.preprocessing.document_chunker import DocumentChunker
from app.core.config import settings
from pathlib import Path

PROCESSED_DIR = Path(settings.storage.processed_dir)
OUTPUT_DIR = Path(settings.storage.chunks_dir)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunker = DocumentChunker()

    files = sorted(PROCESSED_DIR.glob("*.json"))

    logger.info(f"Found {len(files)} processed papers.")

    for file in files:

        logger.info(f"Chunking {file.name}")

        with file.open("r", encoding="utf-8") as f:
            document = json.load(f)

        chunks = chunker.chunk_document(document)

        output_path = OUTPUT_DIR / file.name

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                chunks,
                f,
                indent=4,
                ensure_ascii=False,
            )

        logger.success(f"Saved {output_path.name}")


if __name__ == "__main__":
    main()