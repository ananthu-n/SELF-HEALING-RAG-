from pathlib import Path
import json

from app.core.config import settings
from app.core.logger import logger


def load_chunks(file_path: Path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:

    chunk_dir = Path(settings.storage.chunks_dir)
    validated_dir = chunk_dir.parent / "validated_chunks"

    original_files = sorted(chunk_dir.glob("*.json"))

    total_rejected = 0

    logger.info("=" * 80)
    logger.info("Rejected Chunk Inspection")
    logger.info("=" * 80)

    for original_file in original_files:

        validated_file = validated_dir / original_file.name

        if not validated_file.exists():
            logger.warning(f"Validated file missing: {original_file.name}")
            continue

        original_chunks = load_chunks(original_file)
        validated_chunks = load_chunks(validated_file)

        valid_ids = {
            chunk["chunk_id"]
            for chunk in validated_chunks
        }

        rejected_chunks = [
            chunk
            for chunk in original_chunks
            if chunk["chunk_id"] not in valid_ids
        ]

        if not rejected_chunks:
            continue

        logger.info("")
        logger.info(f"Paper : {original_file.stem}")
        logger.info(f"Rejected Chunks : {len(rejected_chunks)}")

        for chunk in rejected_chunks:

            total_rejected += 1

            logger.info("-" * 60)

            logger.info(f"Chunk ID : {chunk['chunk_id']}")
            logger.info(f"Page     : {chunk['page_number']}")
            logger.info(f"Length   : {len(chunk['text'].strip())}")

            preview = chunk["text"].strip()

            if len(preview) > 300:
                preview = preview[:300] + "..."

            logger.info("Text:")
            logger.info(preview)

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Total Rejected Chunks : {total_rejected}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()