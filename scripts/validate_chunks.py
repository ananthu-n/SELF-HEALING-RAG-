from app.core.logger import logger
from app.preprocessing.chunk_validator import ChunkValidator


def main() -> None:
    logger.info("Starting chunk validation...")

    validator = ChunkValidator()

    validator.validate_corpus()

    logger.success("Chunk validation completed.")


if __name__ == "__main__":
    main()