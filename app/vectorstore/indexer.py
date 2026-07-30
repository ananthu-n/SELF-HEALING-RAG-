import json
import uuid
from pathlib import Path

import numpy as np
from qdrant_client.models import PointStruct
from tqdm import tqdm

from app.core.config import settings
from app.core.logger import logger
from app.vectorstore.client import QdrantDB


class VectorIndexer:
    """
    Production Vector Indexer

    Reads:
        data/chunks/
        data/embeddings/
        data/embedding_metadata/

    Uploads:
        Qdrant PointStruct
    """

    def __init__(self):

        self.client = QdrantDB.get_client()

        self.collection = settings.vectorstore.collection_name

        self.chunk_dir = Path(settings.storage.chunks_dir)

        self.embedding_dir = Path(settings.embedding.embeddings_dir)

        self.embedding_metadata_dir = Path(
            settings.embedding.metadata_dir
        )

        self.batch_size = 256

    def _embedding_files(self):

        return sorted(self.embedding_dir.glob("*.npy"))

    def _load_embeddings(self, path: Path):

        return np.load(path)

    def _load_chunks(self, paper_id):

        file = self.chunk_dir / f"{paper_id}.json"

        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_embedding_metadata(self, paper_id):

        file = self.embedding_metadata_dir / f"{paper_id}.json"

        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_points(self, vectors, chunks):

        points = []

        for vector, chunk in zip(vectors, chunks):

            payload = {

                "paper_id": chunk["paper_id"],

                "chunk_id": chunk["chunk_id"],

                "page_number": chunk["page_number"],

                "chunk_index": chunk["chunk_index"],

                "char_start": chunk["char_start"],

                "char_end": chunk["char_end"],

                "text": chunk["text"],
            }

            points.append(

                PointStruct(

                    id=str(uuid.uuid4()),

                    vector=vector.tolist(),

                    payload=payload,
                )
            )

        return points

    def _upload(self, points):

        for i in range(0, len(points), self.batch_size):

            batch = points[i:i + self.batch_size]

            self.client.upsert(

                collection_name=self.collection,

                wait=True,

                points=batch,
            )

    def index_single_document(self, paper_id: str):
        """Indexes a single document into Qdrant without re-indexing the entire database."""
        emb_file = self.embedding_dir / f"{paper_id}.npy"
        if not emb_file.exists():
            logger.error(f"Embedding file not found for paper_id: {paper_id}")
            return

        embeddings = self._load_embeddings(emb_file)
        chunks = self._load_chunks(paper_id)

        if len(chunks) != len(embeddings):
            logger.error(f"{paper_id}: {len(chunks)} chunks != {len(embeddings)} embeddings")
            return

        points = self._build_points(embeddings, chunks)
        self._upload(points)
        logger.success(f"Single paper '{paper_id}' indexed ({len(points)} vectors) into Qdrant.")

    def index_all(self):

        embedding_files = self._embedding_files()

        logger.info(f"Found {len(embedding_files)} embedding files.")

        total_vectors = 0

        for embedding_file in tqdm(embedding_files):

            paper_id = embedding_file.stem

            embeddings = self._load_embeddings(embedding_file)

            chunks = self._load_chunks(paper_id)

            embedding_meta = self._load_embedding_metadata(paper_id)

            expected = embedding_meta["num_chunks"]

            if expected != len(chunks):

                logger.error(

                    f"{paper_id}: metadata says {expected} chunks "
                    f"but chunk file has {len(chunks)}"

                )

                continue

            if len(chunks) != len(embeddings):

                logger.error(

                    f"{paper_id}: "

                    f"{len(chunks)} chunks != "

                    f"{len(embeddings)} embeddings"

                )

                continue

            points = self._build_points(

                embeddings,

                chunks,
            )

            self._upload(points)

            total_vectors += len(points)

            logger.success(

                f"{paper_id} indexed ({len(points)} vectors)"

            )

        info = self.client.get_collection(

            self.collection

        )

        logger.success("Indexing completed.")

        logger.info(f"Uploaded vectors : {total_vectors}")

        logger.info(f"Qdrant points    : {info.points_count}")