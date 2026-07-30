from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from app.core.config import settings
from app.core.logger import logger
from app.ingestion.arxiv_client import ArxivClient
from app.ingestion.pdf_downloader import PDFDownloader
from app.ingestion.topic_extractor import TopicExtractor
from app.preprocessing.pdf_parser import PDFParser
from app.preprocessing.document_chunker import DocumentChunker
from app.embeddings.model import EmbeddingModel
from app.vectorstore.indexer import VectorIndexer
from app.retrieval.bm25_index import BM25Index
from app.reranker.cross_encoder import CrossEncoderModel


class DynamicKnowledgeAcquisitor:
    """
    On-the-fly dynamic knowledge acquisition service.

    1. Extracts research topic (stripping natural language question words).
    2. Searches arXiv metadata (fetching top candidates).
    3. Reranks candidate titles + abstracts with CrossEncoder.
    4. Downloads top-reranked PDFs, chunks, embeds, and updates indices live.
    """

    def __init__(self, max_results: int = 10, top_k_download: int = 3) -> None:
        self.arxiv_client = ArxivClient(max_results=max_results)
        self.top_k_download = top_k_download
        self.downloader = PDFDownloader()
        self.parser = PDFParser()
        self.chunker = DocumentChunker()
        self.embedding_model = EmbeddingModel()
        self.vector_indexer = VectorIndexer()
        self.bm25_index = BM25Index()
        self.cross_encoder = CrossEncoderModel()

    def acquire(self, query: str) -> int:
        """
        Extract topic, search arXiv, rerank titles/summaries, download top papers, and index.
        """
        topic = TopicExtractor.extract_topic(query)
        logger.info("=" * 70)
        logger.info(f"DYNAMIC KNOWLEDGE ACQUISITION: Extracted topic '{topic}' from query '{query}'")
        logger.info("=" * 70)

        # Step 1: Search arXiv metadata (fetch top 10)
        candidate_papers = self.arxiv_client.search(topic)
        if not candidate_papers:
            logger.warning(f"No papers found on arXiv for topic '{topic}'.")
            return 0

        logger.info(f"Retrieved {len(candidate_papers)} paper metadata candidates from arXiv. Reranking...")

        # Step 2: Rerank metadata (title + summary) using CrossEncoder
        pairs = [
            (topic, f"{paper.title}\n{paper.summary}")
            for paper in candidate_papers
        ]
        scores = self.cross_encoder.predict(pairs)

        # Pair papers with scores and sort
        scored_papers = list(zip(candidate_papers, scores))
        scored_papers.sort(key=lambda item: item[1], reverse=True)

        # Select top reranked papers to download
        selected_papers = [paper for paper, score in scored_papers[: self.top_k_download]]

        logger.info(f"Selected Top-{len(selected_papers)} reranked papers for acquisition:")
        for rank, (paper, score) in enumerate(scored_papers[: self.top_k_download], start=1):
            logger.info(f"  [{rank}] Score: {score:.4f} | Title: {paper.title}")

        new_papers_count = 0
        raw_pdf_dir = Path(settings.storage.raw_pdf_dir)
        chunk_dir = Path(settings.storage.chunks_dir)
        validated_chunk_dir = Path("data/validated_chunks")
        embedding_dir = Path(settings.embedding.embeddings_dir)
        metadata_dir = Path(settings.embedding.metadata_dir)

        chunk_dir.mkdir(parents=True, exist_ok=True)
        validated_chunk_dir.mkdir(parents=True, exist_ok=True)
        embedding_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        for paper in selected_papers:
            paper_id = paper.entry_id.split("/")[-1]
            pdf_path = raw_pdf_dir / f"{paper_id}.pdf"

            # Download PDF
            self.downloader.download(paper)
            if not pdf_path.exists():
                continue

            # Parse PDF
            parsed_doc = self.parser.parse(pdf_path)
            chunks = self.chunker.chunk_document(parsed_doc)

            if not chunks:
                continue

            # Save chunk files
            chunk_file = chunk_dir / f"{paper_id}.json"
            val_chunk_file = validated_chunk_dir / f"{paper_id}.json"
            
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=4, ensure_ascii=False)
            with open(val_chunk_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=4, ensure_ascii=False)

            # Generate Embeddings
            texts = [c["text"] for c in chunks]
            embeddings = self.embedding_model.encode(texts)
            
            emb_file = embedding_dir / f"{paper_id}.npy"
            np.save(emb_file, embeddings)

            # Save embedding metadata
            meta_file = metadata_dir / f"{paper_id}.json"
            meta_data = {
                "paper_id": paper_id,
                "num_chunks": len(chunks),
                "embedding_dim": embeddings.shape[1],
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=4)

            new_papers_count += 1

        if new_papers_count > 0:
            logger.info("Indexing new vectors into Qdrant...")
            self.vector_indexer.index_all()

            logger.info("Rebuilding BM25 index...")
            self.bm25_index.build()
            logger.success(f"Dynamic acquisition completed: {new_papers_count} new paper(s) indexed.")

        return new_papers_count
