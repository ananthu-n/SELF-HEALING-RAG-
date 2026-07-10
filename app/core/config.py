from pathlib import Path

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    name: str


class StorageConfig(BaseModel):
    raw_pdf_dir: str
    processed_dir: str
    metadata_dir: str
    chunks_dir: str


class ArxivConfig(BaseModel):
    queries: list[str]
    papers_per_query: int


class ChunkingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int


class EmbeddingConfig(BaseModel):
    model: str


class VectorStoreConfig(BaseModel):
    collection_name: str


class LLMConfig(BaseModel):
    model: str


class Settings(BaseModel):
    project: ProjectConfig
    storage: StorageConfig
    arxiv: ArxivConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    llm: LLMConfig


config_path = Path("configs/config.yaml")

if not config_path.exists():
    raise FileNotFoundError(config_path)

with config_path.open("r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

settings = Settings(**config_data)