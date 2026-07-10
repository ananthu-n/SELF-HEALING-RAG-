from app.core.config import settings
from app.core.logger import logger
from app.ingestion.arxiv_client import ArxivClient
from app.ingestion.pdf_downloader import PDFDownloader


def main() -> None:
    downloader = PDFDownloader()

    downloaded = 0
    skipped = 0
    failed = 0

    logger.info("========== Starting arXiv Paper Download ==========")

    for query in settings.arxiv.queries:

        logger.info(f"Searching: {query}")

        client = ArxivClient(
            max_results=settings.arxiv.papers_per_query
        )

        papers = client.search(query)

        logger.info(f"Found {len(papers)} papers")

        for paper in papers:

            try:

                if downloader.download(paper):
                    downloaded += 1
                else:
                    skipped += 1

            except Exception as e:
                failed += 1
                logger.exception(
                    f"Failed to download '{paper.title}': {e}"
                )

    logger.success("========== Download Summary ==========")
    logger.info(f"Downloaded : {downloaded}")
    logger.info(f"Skipped    : {skipped}")
    logger.info(f"Failed     : {failed}")
    logger.info(f"Total      : {downloaded + skipped + failed}")


if __name__ == "__main__":
    main()