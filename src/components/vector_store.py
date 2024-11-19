import os
from pathlib import Path
from typing import Any, Optional, Tuple, List

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from src import config as app_config
from src.components.embedding_factory import EmbeddingFactory
from src.exception.exception import RetrievalError
from src.logging import logger


class VectorStore:
    def __init__(self, provider: str = "azure"):
        """
        Initialize VectorStore with specified embedding provider.

        Args:
            provider: The embedding provider to use (e.g., "azure", "huggingface")

        Raises:
            RetrievalError: If vector store initialization fails
        """
        self.provider = provider
        self._embedding_factory = None
        self._vectorstore = None

    @property
    def embedding_factory(self) -> EmbeddingFactory:
        """
        Lazy loading of embedding factory.

        Raises:
            RetrievalError: If embedding factory initialization fails
        """
        if self._embedding_factory is None:
            try:
                self._embedding_factory = EmbeddingFactory(self.provider)
                logger.info("Successfully initialized embedding factory")
            except Exception as e:
                raise RetrievalError(
                    "Failed to initialize embedding factory",
                    details={
                        "provider": self.provider,
                        "error": str(e),
                    },
                )
        return self._embedding_factory

    @property
    def vectorstore(self) -> Chroma:
        """
        Lazy loading of vectorstore.

        Raises:
            RetrievalError: If vectorstore initialization fails
        """
        if self._vectorstore is None:
            try:
                self._vectorstore = Chroma(
                    collection_name=f"{self.provider}_embeddings",
                    embedding_function=self.embedding_factory.model,
                    persist_directory=app_config.VECTOR_DATABASE_DIR,
                )
                logger.info("Successfully created vectorstore")
            except Exception as e:
                raise RetrievalError(
                    "Failed to create vectorstore",
                    details={
                        "collection": f"{self.provider}_embeddings",
                        "persist_dir": app_config.VECTOR_DATABASE_DIR,
                        "error": str(e),
                    },
                )
        return self._vectorstore

    def get_similarity_retriever(self, k: int = 10) -> Any:
        """
        Create a similarity-based retriever from the vectorstore.

        Args:
            k: Number of documents to retrieve

        Returns:
            A similarity-based retriever

        Raises:
            RetrievalError: If similarity retriever creation fails
        """
        try:
            logger.info("Creating similarity retriever with k=%d", k)
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        except Exception as e:
            raise RetrievalError(
                "Failed to create similarity retriever",
                details={
                    "k": k,
                    "search_type": "similarity",
                    "error": str(e),
                },
            )

    def get_bm25_retriever(self, docs: List[Any], k: int = 10) -> BM25Retriever:
        """
        Create a BM25-based retriever.

        Args:
            docs: List of documents to index
            k: Number of documents to retrieve

        Returns:
            A BM25-based retriever

        Raises:
            RetrievalError: If BM25 retriever creation fails
        """
        if not docs:
            raise RetrievalError(
                "Empty document list provided for BM25 retriever",
                details={"num_docs": 0},
            )

        try:
            logger.info("Creating BM25 retriever with k=%d", k)
            return BM25Retriever.from_documents(
                docs,
                preprocess_func=lambda x: x.page_content,
                k=k
            )
        except Exception as e:
            raise RetrievalError(
                "Failed to create BM25 retriever",
                details={
                    "num_docs": len(docs),
                    "k": k,
                    "error": str(e),
                },
            )

    def get_ensemble_retriever(
        self, 
        docs: Optional[List[Any]] = None,
        k: int = 10,
        weights: Optional[List[float]] = None
    ) -> EnsembleRetriever:
        """
        Create an ensemble retriever combining similarity and BM25.

        Args:
            docs: List of documents for BM25 retriever
            k: Number of documents to retrieve
            weights: Weights for each retriever [similarity_weight, bm25_weight]

        Returns:
            An ensemble retriever combining similarity and BM25

        Raises:
            RetrievalError: If ensemble retriever creation fails
        """
        if weights and len(weights) != 2:
            raise RetrievalError(
                "Invalid weights for ensemble retriever",
                details={
                    "weights": weights,
                    "expected_length": 2,
                },
            )

        try:
            retrievers = [
                self.get_similarity_retriever(k=k),
            ]
            
            if docs:
                retrievers.append(self.get_bm25_retriever(docs=docs, k=k))
            
            weights = weights or [0.5] * len(retrievers)
            
            logger.info(
                "Creating ensemble retriever with %d retrievers and weights %s",
                len(retrievers),
                weights
            )
            
            return EnsembleRetriever(
                retrievers=retrievers,
                weights=weights
            )
        except Exception as e:
            raise RetrievalError(
                "Failed to create ensemble retriever",
                details={
                    "num_retrievers": len(retrievers) if 'retrievers' in locals() else 0,
                    "weights": weights,
                    "k": k,
                    "error": str(e),
                },
            )

    def add_documents(self, documents: List[Any]) -> None:
        """
        Add documents to the vectorstore.

        Args:
            documents: List of documents to add

        Raises:
            RetrievalError: If document addition fails
        """
        if not documents:
            raise RetrievalError(
                "Empty document list provided for addition",
                details={"num_docs": 0},
            )

        try:
            logger.info("Adding %d documents to vectorstore", len(documents))
            self.vectorstore.add_documents(documents)
            logger.info("Successfully added documents to vectorstore")
        except Exception as e:
            raise RetrievalError(
                "Failed to add documents to vectorstore",
                details={
                    "num_docs": len(documents),
                    "error": str(e),
                },
            )

    def similarity_search(self, query: str, k: int = 10) -> List[Any]:
        """
        Perform similarity search on the vectorstore.

        Args:
            query: Query text
            k: Number of documents to retrieve

        Returns:
            List of similar documents

        Raises:
            RetrievalError: If similarity search fails
        """
        if not query.strip():
            raise RetrievalError(
                "Empty query provided for similarity search",
                details={"query_length": 0},
            )

        try:
            logger.info("Performing similarity search with k=%d", k)
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info("Successfully retrieved %d documents", len(results))
            return results
        except Exception as e:
            raise RetrievalError(
                "Failed to perform similarity search",
                details={
                    "query_preview": query[:100],
                    "k": k,
                    "error": str(e),
                },
            )
