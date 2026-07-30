from pathlib import Path

from app.core.logger import logger
from app.embeddings.validator import EmbeddingValidator


def main():
    validator = EmbeddingValidator()

    embedding_dir = Path("data/embeddings")

    embedding_files = sorted(embedding_dir.glob("*.npy"))

    logger.info(f"Found {len(embedding_files)} embedding files.")

    passed = 0
    failed = 0

    failed_papers = []

    for file in embedding_files:

        paper_id = file.stem

        if validator.validate(paper_id):
            passed += 1
        else:
            failed += 1
            failed_papers.append(paper_id)

    logger.info("=" * 60)
    logger.info("Embedding Validation Summary")
    logger.info("=" * 60)

    logger.success(f"Passed : {passed}")
    logger.error(f"Failed : {failed}")

    if failed_papers:
        logger.warning("Failed Papers:")

        for paper in failed_papers:
            logger.warning(f" - {paper}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()