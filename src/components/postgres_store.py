import os
from typing import List, Dict, Any
from sqlalchemy import create_engine, Column, String, Integer, Float, JSON, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from langchain.docstore.document import Document

from src.components.embedding_factory import EmbeddingFactory
from src.exception.exception import DocumentProcessingError
from src.log_utils import logger
from src.secure.secrets import secrets

Base = declarative_base()

class DocumentEmbedding(Base):
    __tablename__ = 'document_embeddings'

    id = Column(Integer, primary_key=True)
    content = Column(String)
    embedding = Column(ARRAY(Float))
    metadata = Column(JSON)

class PostgresStore:
    def __init__(self, provider: str = "azure"):
        """
        Initialize PostgresStore with specified embedding provider.

        Args:
            provider: The embedding provider to use (e.g., "azure", "huggingface")
        """
        self.provider = provider
        self._embedding_factory = None
        self._engine = None
        self._Session = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize database connection and create tables if they don't exist."""
        try:
            db_url = secrets.get_secret("POSTGRES_URL")
            if not db_url:
                raise DocumentProcessingError(
                    "PostgreSQL connection URL not found in secrets"
                )
            
            self._engine = create_engine(db_url)
            Base.metadata.create_all(self._engine)
            self._Session = sessionmaker(bind=self._engine)
            logger.info("Successfully initialized PostgreSQL connection")
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to initialize PostgreSQL connection",
                details={"error": str(e)}
            )

    @property
    def embedding_factory(self) -> EmbeddingFactory:
        """Lazy loading of embedding factory."""
        if self._embedding_factory is None:
            try:
                self._embedding_factory = EmbeddingFactory(self.provider)
                logger.info("Successfully initialized embedding factory")
            except Exception as e:
                raise DocumentProcessingError(
                    "Failed to initialize embedding factory",
                    details={
                        "provider": self.provider,
                        "error": str(e),
                    },
                )
        return self._embedding_factory

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents and their embeddings to PostgreSQL.

        Args:
            documents: List of Langchain Document objects to add

        Raises:
            DocumentProcessingError: If document addition fails
        """
        try:
            # Generate embeddings for all documents
            texts = [doc.page_content for doc in documents]
            embeddings = self.embedding_factory.model.embed_documents(texts)

            # Create session
            session = self._Session()

            try:
                # Add each document with its embedding
                for doc, embedding in zip(documents, embeddings):
                    doc_embedding = DocumentEmbedding(
                        content=doc.page_content,
                        embedding=embedding,
                        metadata=doc.metadata
                    )
                    session.add(doc_embedding)
                
                session.commit()
                logger.info(f"Successfully added {len(documents)} documents to PostgreSQL")
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()

        except Exception as e:
            raise DocumentProcessingError(
                "Failed to add documents to PostgreSQL",
                details={"error": str(e)}
            )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query: The query text
            k: Number of documents to return

        Returns:
            List of similar documents

        Raises:
            DocumentProcessingError: If search fails
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_factory.model.embed_query(query)

            # Create session
            session = self._Session()

            try:
                # Perform similarity search using PostgreSQL
                # Note: This is a basic implementation. For production, you might want to use
                # pgvector or other optimized solutions for vector similarity search
                result = session.execute(f"""
                    SELECT content, metadata, 
                           1 - (embedding <=> ARRAY{query_embedding}::float[]) as similarity
                    FROM document_embeddings
                    ORDER BY similarity DESC
                    LIMIT {k}
                """)

                documents = []
                for row in result:
                    doc = Document(
                        page_content=row.content,
                        metadata=row.metadata
                    )
                    documents.append(doc)

                return documents
            finally:
                session.close()

        except Exception as e:
            raise DocumentProcessingError(
                "Failed to perform similarity search",
                details={"error": str(e)}
            )
