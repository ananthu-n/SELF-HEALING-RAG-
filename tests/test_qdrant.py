from app.core.logger import logger
from app.vectorstore.client import QdrantDB


def main():

    client = QdrantDB.get_client()

    logger.success("Qdrant client created.")

    collections = client.get_collections()

    logger.info(collections)

    QdrantDB.close()


if __name__ == "__main__":
    main()