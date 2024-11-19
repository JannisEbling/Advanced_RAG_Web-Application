from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

from config.settings import get_settings
from src.exception.exception import MultiAgentRAGException
from src.logging import logger
from src.secure.secrets import secrets


class EmbeddingFactory:
    def __init__(self, provider: str):
        """
        Initialize the EmbeddingFactory with a provider.

        Args:
            provider (str): The embedding provider to use (e.g., "azure", "huggingface")
        """
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.model = self._initialize_model()

    def _initialize_model(self) -> Any:
        """Initialize and return the embedding model based on the provider."""
        model_initializers = {
            "azure": self._initialize_azure_openai,
            "huggingface": self._initialize_huggingface,
        }

        initializer = model_initializers.get(self.provider)
        if not initializer:
            raise MultiAgentRAGException(f"Unsupported embedding provider: {self.provider}")

        try:
            model = initializer()
            logger.info("Successfully initialized %s embedding model", self.provider)
            return model
        except Exception as e:
            logger.error("Failed to initialize embedding model", exc_info=True)
            raise MultiAgentRAGException(
                f"Failed to initialize {self.provider} embedding model: {str(e)}"
            ) from e

    def _initialize_azure_openai(self) -> AzureOpenAIEmbeddings:
        """Initialize Azure OpenAI embeddings."""
        api_key = secrets.get_secret("AZURE_OPENAI_API_KEY")
        api_endpoint = secrets.get_secret("AZURE_OPENAI_ENDPOINT")

        if not api_key or not api_endpoint:
            raise MultiAgentRAGException(
                "Azure OpenAI credentials not found. Please run init_secrets.py to set them up."
            )

        return AzureOpenAIEmbeddings(
            azure_deployment=self.settings.deployment_name,
            azure_endpoint=api_endpoint,
            api_key=api_key,
            api_version=self.settings.api_version,
            dimensions=self.settings.dimensions,
        )

    def _initialize_huggingface(self) -> HuggingFaceEmbeddings:
        """Initialize HuggingFace embeddings."""
        logger.info("Initializing HuggingFace Embeddings model")
        return HuggingFaceEmbeddings(
            model_name=self.settings.model_name,
            multi_process=True,
            model_kwargs={"device": self.settings.device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Create embeddings for a list of texts.

        Args:
            texts (list[str]): List of texts to create embeddings for

        Returns:
            list[list[float]]: List of embeddings
        """
        try:
            logger.debug("Creating embeddings for %d texts", len(texts))
            embeddings = self.model.embed_documents(texts)
            logger.debug("Successfully created embeddings")
            return embeddings
        except Exception as e:
            logger.error("Failed to create embeddings", exc_info=True)
            raise MultiAgentRAGException(f"Failed to create embeddings: {str(e)}") from e

    def create_embedding(self, text: str) -> list[float]:
        """
        Create embedding for a single text.

        Args:
            text (str): Text to create embedding for

        Returns:
            list[float]: Embedding vector
        """
        try:
            logger.debug("Creating embedding for text")
            embedding = self.model.embed_query(text)
            logger.debug("Successfully created embedding")
            return embedding
        except Exception as e:
            logger.error("Failed to create embedding", exc_info=True)
            raise MultiAgentRAGException(f"Failed to create embedding: {str(e)}") from e
