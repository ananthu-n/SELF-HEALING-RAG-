from app.core.logger import logger
from app.vectorstore.client import QdrantDB
from app.vectorstore.indexer import VectorIndexer
from app.vectorstore.collection import CollectionManager


def main():

    logger.info("Starting vector indexing...")
    
    CollectionManager().create()

    indexer = VectorIndexer()

    indexer.index_all()

    QdrantDB.close()


if __name__ == "__main__":
    main()