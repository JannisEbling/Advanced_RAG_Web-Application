import os
from functools import lru_cache
from typing import Optional, Literal
from datetime import timedelta

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from src.log_utils import logger
from src.exception.exception import RAGPipelineError


class Config:
    DEBUG = False
    LOG_LEVEL = "INFO"
    DATABASE_URL = "sqlite:///data/database.db"


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    LOG_LEVEL = "WARNING"


class LLMProviderSettings(BaseSettings):
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMProviderSettings):
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    default_model: str = "gpt-4"


class AzureSettings(LLMProviderSettings):
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_endpoint: str = os.getenv("AZURE_OPENAI_API_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    default_model: str = os.getenv("AZURE_OPENAI_MODEL", "gpt-4")
    deployment_id: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID", "Answering_35_0125")


class AnthropicSettings(LLMProviderSettings):
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    default_model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 1024


class LlamaSettings(LLMProviderSettings):
    api_key: str = "key"  # required, but not used
    default_model: str = "llama3"
    base_url: str = "http://localhost:11434/v1"


class EmbeddingProviderSettings(BaseSettings):
    max_retries: int = 3
    batch_size: int = 100


class OpenAIEmbeddingSettings(EmbeddingProviderSettings):
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    default_model: str = "text-embedding-3-large"


class AzureEmbeddingSettings(EmbeddingProviderSettings):
    api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_endpoint: str = os.getenv("AZURE_OPENAI_API_ENDPOINT", "")
    api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    default_model: str = os.getenv(
        "AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
    )
    deployment_id: str = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID", "Embedding_Small"
    )


class HuggingFaceEmbeddingSettings(EmbeddingProviderSettings):
    default_model: str = "BAAI/bge-large-en-v1.5"
    device: str = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
    normalize_embeddings: bool = True


class VectorStoreSettings(BaseSettings):
    """Configuration for vector store."""

    document_table: str = os.getenv("VECTOR_STORE_DOC_TABLE", "document_embeddings")
    figure_table: str = os.getenv("VECTOR_STORE_FIG_TABLE", "figures")
    formula_table: str = os.getenv("VECTOR_STORE_FOR_TABLE", "formulas")
    embedding_dimensions: int = 1536
    time_partition_interval: timedelta = timedelta(days=7)
    service_url: str = os.getenv(
        "TIMESCALE_DATABASE_URL",
        "postgresql://postgres:password@localhost:5433/postgres",
    )


class Settings(BaseSettings):
    app_name: str = "Advanced RAG Web Application"
    environment: Literal["development", "production"] = "development"
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")

    # Default providers
    default_llm_provider: Literal["azure", "openai", "anthropic", "llama"] = "azure"
    default_embedding_provider: Literal["azure", "openai", "huggingface"] = "azure"

    # Provider settings
    openai: OpenAISettings = OpenAISettings()
    azure: AzureSettings = AzureSettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    llama: LlamaSettings = LlamaSettings()

    # Embedding settings
    openai_embedding: OpenAIEmbeddingSettings = OpenAIEmbeddingSettings()
    azure_embedding: AzureEmbeddingSettings = AzureEmbeddingSettings()
    huggingface_embedding: HuggingFaceEmbeddingSettings = HuggingFaceEmbeddingSettings()

    # Vector store settings
    vector_store: VectorStoreSettings = VectorStoreSettings()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
