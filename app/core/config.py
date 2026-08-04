from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

# Automatically load environment variables from local .env file
load_dotenv()


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
    model_name: str
    device: str
    batch_size: int
    normalize: bool
    precision: str
    embeddings_dir: Path
    metadata_dir: Path


class VectorStoreConfig(BaseModel):
    provider: str
    location: Path
    collection_name: str
    distance: str
    vector_size: int

class RerankerConfig(BaseModel):
    model_name: str
    device: str
    batch_size: int
    max_length: int
    normalize_scores: bool


class PromptConfig(BaseModel):
    max_chunks: int
    max_context_characters: int
    include_metadata: bool

class LLMConfig(BaseModel):
    provider: str
    model: str
    base_url: str
    temperature: float
    timeout: int
    grounding_model: str

class SelfHealingConfig(BaseModel):
    enabled: bool
    max_retries: int
    confidence_threshold: float
    use_langgraph: bool = False

class Settings(BaseModel):
    project: ProjectConfig
    storage: StorageConfig
    arxiv: ArxivConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    llm: LLMConfig
    reranker: RerankerConfig
    prompt: PromptConfig
    self_healing: SelfHealingConfig



config_path = Path("configs/config.yaml")

if not config_path.exists():
    raise FileNotFoundError(config_path)

with config_path.open("r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

settings = Settings(**config_data)