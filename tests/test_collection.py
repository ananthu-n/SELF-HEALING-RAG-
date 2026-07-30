from app.vectorstore.client import QdrantDB
from app.vectorstore.collection import CollectionManager


def main():

    manager = CollectionManager()

    manager.create()

    manager.info()

    QdrantDB.close()


if __name__ == "__main__":
    main()