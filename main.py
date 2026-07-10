from app.core.config import settings
from app.core.logger import logger


def main():
    logger.info(settings.get("project.name"))
    logger.info(settings.get("embedding.model"))
    logger.info(settings.get("arxiv.papers_per_query"))


if __name__ == "__main__":
    main()