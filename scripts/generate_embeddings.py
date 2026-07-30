from app.core.logger import logger
from app.embeddings.generator import EmbeddingGenerator


def main() -> None:
    logger.info("Starting embedding generation...")

    generator = EmbeddingGenerator()

    generator.process_all()

    logger.success("Embedding generation finished.")


if __name__ == "__main__":
    main()