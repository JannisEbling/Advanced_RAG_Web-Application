import os
from pathlib import Path
from typing import Any, Optional, Tuple

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from src import config as app_config
from src.components.embedding_factory import EmbeddingFactory
from src.exception.exception import MultiAgentRAGException
from src.logging import logger


class VectorStore:
    def __init__(self, provider: str = "azure"):
        """
        Initialize VectorStore with specified embedding provider.

        Args:
            provider (str): The embedding provider to use (e.g., "azure", "huggingface")
        """
        self.provider = provider
        self._embedding_factory = None
        self._vectorstore = None

    @property
    def embedding_factory(self) -> EmbeddingFactory:
        """Lazy loading of embedding factory."""
        if self._embedding_factory is None:
            try:
                self._embedding_factory = EmbeddingFactory(self.provider)
                logger.info("Successfully initialized embedding factory")
            except Exception as e:
                logger.error("Failed to initialize embedding factory", exc_info=True)
                raise MultiAgentRAGException(f"Failed to initialize embedding factory: {str(e)}") from e
        return self._embedding_factory

    @property
    def vectorstore(self) -> Chroma:
        """Lazy loading of vectorstore."""
        if self._vectorstore is None:
            try:
                self._vectorstore = Chroma(
                    collection_name=f"{self.provider}_embeddings",
                    embedding_function=self.embedding_factory.model,
                    persist_directory=app_config.VECTOR_DATABASE_DIR,
                )
                logger.info("Successfully created vectorstore")
            except Exception as e:
                logger.error("Failed to create vectorstore", exc_info=True)
                raise MultiAgentRAGException(f"Failed to create vectorstore: {str(e)}") from e
        return self._vectorstore

    def get_similarity_retriever(self, k: int = 10) -> Any:
        """
        Create a similarity-based retriever from the vectorstore.

        Args:
            k (int): Number of documents to retrieve

        Returns:
            Any: A similarity-based retriever
        """
        try:
            logger.info("Creating similarity retriever with k=%d", k)
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        except Exception as e:
            logger.error("Failed to create similarity retriever", exc_info=True)
            raise MultiAgentRAGException(f"Failed to create similarity retriever: {str(e)}") from e

    def get_bm25_retriever(self, docs: list, k: int = 10) -> BM25Retriever:
        """
        Create a BM25-based retriever.

        Args:
            docs (list): List of documents to index
            k (int): Number of documents to retrieve

        Returns:
            BM25Retriever: A BM25-based retriever
        """
        try:
            logger.info("Creating BM25 retriever with k=%d", k)
            return BM25Retriever.from_documents(
                docs,
                preprocess_func=lambda x: x.page_content,
                k=k
            )
        except Exception as e:
            logger.error("Failed to create BM25 retriever", exc_info=True)
            raise MultiAgentRAGException(f"Failed to create BM25 retriever: {str(e)}") from e

    def get_ensemble_retriever(
        self, 
        docs: Optional[list] = None,
        k: int = 10,
        weights: Optional[list] = None
    ) -> EnsembleRetriever:
        """
        Create an ensemble retriever combining similarity and BM25 retrievers.

        Args:
            docs (list, optional): List of documents for BM25 retriever
            k (int): Number of documents to retrieve
            weights (list, optional): Weights for each retriever. Defaults to [0.5, 0.5]

        Returns:
            EnsembleRetriever: Combined retriever
        """
        try:
            logger.info("Creating ensemble retriever")
            weights = weights or [0.5, 0.5]
            
            similarity_retriever = self.get_similarity_retriever(k=k)
            bm25_retriever = self.get_bm25_retriever(docs=docs, k=k)

            return EnsembleRetriever(
                retrievers=[similarity_retriever, bm25_retriever],
                weights=weights
            )
        except Exception as e:
            logger.error("Failed to create ensemble retriever", exc_info=True)
            raise MultiAgentRAGException(f"Failed to create ensemble retriever: {str(e)}") from e

    def add_documents(self, documents: list) -> None:
        """
        Add documents to the vectorstore.

        Args:
            documents (list): List of documents to add
        """
        try:
            logger.info("Adding %d documents to vectorstore", len(documents))
            self.vectorstore.add_documents(documents)
            logger.info("Successfully added documents to vectorstore")
        except Exception as e:
            logger.error("Failed to add documents to vectorstore", exc_info=True)
            raise MultiAgentRAGException(f"Failed to add documents: {str(e)}") from e

    def similarity_search(self, query: str, k: int = 10) -> list:
        """
        Perform similarity search in the vectorstore.

        Args:
            query (str): Query string
            k (int): Number of documents to retrieve

        Returns:
            list: List of retrieved documents
        """
        try:
            logger.info("Performing similarity search for query: %s", query)
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info("Successfully retrieved %d documents", len(results))
            return results
        except Exception as e:
            logger.error("Failed to perform similarity search", exc_info=True)
            raise MultiAgentRAGException(f"Failed to perform search: {str(e)}") from e
