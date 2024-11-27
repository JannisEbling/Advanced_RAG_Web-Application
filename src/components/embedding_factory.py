from typing import Any, List
import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

from src.config.settings import get_settings
from src.exception.exception import DocumentProcessingError
from src.log_utils import logger


# Load environment variables from .env file
load_dotenv()


class EmbeddingFactory:
    def __init__(self, provider: str):
        """
        Initialize the EmbeddingFactory with a provider.

        Args:
            provider: The embedding provider to use (e.g., "azure", "huggingface")

        Raises:
            DocumentProcessingError: If initialization fails
        """
        self.provider = provider
        try:
            settings = get_settings()
            self.settings = getattr(settings, f"{provider}_embedding")
        except AttributeError:
            raise DocumentProcessingError(
                f"Invalid embedding provider configuration",
                details={"provider": provider},
            )
        self.model = self._initialize_model()

    def _initialize_model(self) -> Any:
        """
        Initialize and return the embedding model based on the provider.

        Raises:
            DocumentProcessingError: If model initialization fails
        """
        model_initializers = {
            "azure": self._initialize_azure_openai,
            "huggingface": self._initialize_huggingface,
        }

        initializer = model_initializers.get(self.provider)
        if not initializer:
            raise DocumentProcessingError(
                "Unsupported embedding provider",
                details={
                    "provider": self.provider,
                    "supported_providers": list(model_initializers.keys()),
                },
            )

        try:
            model = initializer()
            logger.info("Successfully initialized %s embedding model", self.provider)
            return model
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to initialize embedding model",
                details={
                    "provider": self.provider,
                    "error": str(e),
                },
            )

    def _initialize_azure_openai(self) -> AzureOpenAIEmbeddings:
        """
        Initialize Azure OpenAI embeddings.

        Raises:
            DocumentProcessingError: If Azure credentials are missing or invalid
        """
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        if not api_key or not api_endpoint:
            raise DocumentProcessingError(
                "Azure OpenAI credentials not found",
                details={
                    "missing_credentials": [
                        k
                        for k, v in {
                            "api_key": api_key,
                            "api_endpoint": api_endpoint,
                        }.items()
                        if not v
                    ],
                },
            )

        try:
            return AzureOpenAIEmbeddings(
                azure_deployment=self.settings.deployment_id,
                azure_endpoint=api_endpoint,
                api_key=api_key,
                api_version=self.settings.api_version,
                model=self.settings.default_model,
            )
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to initialize Azure OpenAI embeddings",
                details={
                    "deployment": self.settings.deployment_id,
                    "error": str(e),
                },
            )

    def _initialize_huggingface(self) -> HuggingFaceEmbeddings:
        """
        Initialize HuggingFace embeddings.

        Raises:
            DocumentProcessingError: If HuggingFace model initialization fails
        """
        logger.info("Initializing HuggingFace Embeddings model")
        try:
            return HuggingFaceEmbeddings(
                model_name=self.settings.model_name,
                multi_process=True,
                model_kwargs={"device": self.settings.device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to initialize HuggingFace embeddings",
                details={
                    "model": self.settings.model_name,
                    "device": self.settings.device,
                    "error": str(e),
                },
            )

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts.

        Args:
            texts: List of texts to create embeddings for

        Returns:
            List of embeddings

        Raises:
            DocumentProcessingError: If embedding creation fails
        """
        if not texts:
            raise DocumentProcessingError(
                "Empty text list provided for embedding",
                details={"num_texts": 0},
            )

        try:
            logger.debug("Creating embeddings for %d texts", len(texts))
            embeddings = self.model.embed_documents(texts)
            logger.debug("Successfully created embeddings")
            return embeddings
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to create embeddings",
                details={
                    "num_texts": len(texts),
                    "provider": self.provider,
                    "error": str(e),
                },
            )

    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text.

        Args:
            text: Text to create embedding for

        Returns:
            Embedding vector

        Raises:
            DocumentProcessingError: If embedding creation fails
        """
        if not text.strip():
            raise DocumentProcessingError(
                "Empty text provided for embedding",
                details={"text_length": 0},
            )

        try:
            logger.debug("Creating embedding for text: %s", text[:100])
            embedding = self.model.embed_query(text)
            logger.debug("Successfully created embedding")
            return embedding
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to create single embedding",
                details={
                    "text_preview": text[:100],
                    "provider": self.provider,
                    "error": str(e),
                },
            )
