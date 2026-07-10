from pathlib import Path
import json
import requests

from app.core.config import settings
from app.core.logger import logger
from pathlib import Path
from app.models.paper import Paper


class PDFDownloader:
    def __init__(self):
        self.pdf_dir = Path(settings.storage.raw_pdf_dir)
        self.metadata_dir = Path(settings.storage.metadata_dir)

        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    from app.models.paper import Paper


    def download(self, paper: Paper):

        paper_id = paper.entry_id.split("/")[-1]
        pdf_path = self.pdf_dir / f"{paper_id}.pdf"
        metadata_path = self.metadata_dir / f"{paper_id}.json"
        if pdf_path.exists() and metadata_path.exists():
            logger.info(f"Skipping existing paper: {paper_id}")
            return False
        
        ...

        logger.info(f"Downloading: {paper.title}")

        response = requests.get(
            paper.pdf_url,
            timeout=60
        )

        response.raise_for_status()

        # Save PDF
        with open(pdf_path, "wb") as f:
            f.write(response.content)

        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                paper.model_dump(mode="json"),
                f,
                indent=4,
                ensure_ascii=False
)

        logger.success(f"Saved {paper_id}")
        return True