from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from app.core.config import settings
from app.core.logger import logger
from app.preprocessing.pdf_parser import PDFParser
from app.preprocessing.document_chunker import DocumentChunker
from app.embeddings.model import EmbeddingModel
from app.vectorstore.indexer import VectorIndexer
from app.retrieval.bm25_index import BM25Index


class CustomDatasourceIngestor:
    """
    Ingest custom uploaded PDF / datasource files into the Self-Healing RAG pipeline.
    Parses, chunks, embeds, and updates both Qdrant Vector Store and BM25 index.
    """

    def __init__(self) -> None:
        self.parser = PDFParser()
        self.chunker = DocumentChunker()
        self.embedding_model = EmbeddingModel()
        self.vector_indexer = VectorIndexer()
        self.bm25_index = BM25Index()

    def process_file(self, file_path: Path, doc_id: str) -> dict:
        logger.info(f"Processing uploaded document: {file_path.name} (Doc ID: {doc_id})")

        raw_pdf_dir = Path(settings.storage.raw_pdf_dir)
        chunk_dir = Path(settings.storage.chunks_dir)
        validated_chunk_dir = Path("data/validated_chunks")
        embedding_dir = Path(settings.embedding.embeddings_dir)
        metadata_dir = Path(settings.embedding.metadata_dir)

        raw_pdf_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        validated_chunk_dir.mkdir(parents=True, exist_ok=True)
        embedding_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Parse document based on file type
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            parsed_doc = self.parser.parse(file_path)
            parsed_doc["paper_id"] = doc_id
        elif suffix in (".docx", ".doc"):
            try:
                import docx
                doc = docx.Document(file_path)
                full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            except Exception as err:
                logger.warning(f"python-docx parsing failed, falling back to raw text read: {err}")
                full_text = file_path.read_text(encoding="utf-8", errors="ignore")
            parsed_doc = {
                "paper_id": doc_id,
                "num_pages": 1,
                "pages": [{"page": 1, "text": full_text}],
            }
        else:
            # Universal text parser for .txt, .md, .csv, .json, .jsonl, .py, .js, .html, .xml, .yaml, .log, etc.
            text_content = file_path.read_text(encoding="utf-8", errors="ignore")
            parsed_doc = {
                "paper_id": doc_id,
                "num_pages": 1,
                "pages": [{"page": 1, "text": text_content}],
            }

        chunks = self.chunker.chunk_document(parsed_doc)
        if not chunks:
            raise ValueError("No text chunks could be extracted from the uploaded document.")

        # Save chunk JSON files
        chunk_file = chunk_dir / f"{doc_id}.json"
        val_chunk_file = validated_chunk_dir / f"{doc_id}.json"

        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        with open(val_chunk_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)

        # Compute & save embeddings
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_model.encode(texts)

        emb_file = embedding_dir / f"{doc_id}.npy"
        np.save(emb_file, embeddings)

        meta_file = metadata_dir / f"{doc_id}.json"
        meta_data = {
            "paper_id": doc_id,
            "filename": file_path.name,
            "num_chunks": len(chunks),
            "embedding_dim": embeddings.shape[1],
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=4)

        # Index only the single newly uploaded document into Qdrant & rebuild BM25 index
        logger.info(f"Indexing uploaded document '{doc_id}' into Qdrant...")
        self.vector_indexer.index_single_document(doc_id)
        self.bm25_index.build()

        logger.success(f"Custom document '{file_path.name}' successfully ingested ({len(chunks)} chunks).")
        return {
            "doc_id": doc_id,
            "filename": file_path.name,
            "num_chunks": len(chunks),
            "num_pages": parsed_doc.get("num_pages", 1),
        }
