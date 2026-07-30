from qdrant_client.http.models import Distance, VectorParams

from app.core.config import settings
from app.core.logger import logger
from app.vectorstore.client import QdrantDB


class CollectionManager:
    def __init__(self):
        self.client = QdrantDB.get_client()
        self.collection = settings.vectorstore.collection_name

    def exists(self) -> bool:
        collections = self.client.get_collections().collections

        return any(
            collection.name == self.collection
            for collection in collections
        )

    def create(self):

        if self.exists():
            logger.info(
                f"Collection '{self.collection}' already exists."
            )
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=settings.vectorstore.vector_size,
                distance=Distance.COSINE,
            ),
        )

        logger.success(
            f"Collection '{self.collection}' created successfully."
        )

    def delete(self):

        if not self.exists():
            logger.warning(
                f"Collection '{self.collection}' does not exist."
            )
            return

        self.client.delete_collection(self.collection)

        logger.success(
            f"Collection '{self.collection}' deleted."
        )

    def info(self):

        if not self.exists():
            logger.warning(
                f"Collection '{self.collection}' not found."
            )
            return None

        info = self.client.get_collection(self.collection)

        logger.info(info)

        return info