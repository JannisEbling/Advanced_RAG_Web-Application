from typing import List, Dict, Any
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from langchain.docstore.document import Document
import numpy as np

from src.components.embedding_factory import EmbeddingFactory
from src.exception.exception import DocumentProcessingError
from src.logging import logger
from src.secure.secrets import secrets

class TimescaleVectorHandler:
    def __init__(self, provider: str = "azure"):
        """
        Initialize TimescaleDB handler with vector support.

        Args:
            provider: The embedding provider to use (e.g., "azure", "huggingface")
        """
        self.provider = provider
        self._embedding_factory = None
        self._engine = None
        self._Session = None
        self._initialize_db()

    def _initialize_db(self):
        """Initialize connection to TimescaleDB."""
        try:
            db_url = os.getenv("POSTGRES_URL", "postgresql://postgres:password@localhost:5432/postgres")
            
            self._engine = create_engine(db_url)
            self._Session = sessionmaker(bind=self._engine)
            
            # Test connection and extensions
            with self._engine.connect() as conn:
                # Check if extensions are installed
                extensions = conn.execute(text("SELECT extname FROM pg_extension")).fetchall()
                ext_names = [ext[0] for ext in extensions]
                
                if 'vector' not in ext_names:
                    raise DocumentProcessingError("pgvector extension is not installed")
                if 'timescaledb' not in ext_names:
                    raise DocumentProcessingError("TimescaleDB extension is not installed")
                
            logger.info("Successfully initialized TimescaleDB connection with vector support")
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to initialize TimescaleDB connection",
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

    def store_documents(self, documents: List[Document]) -> None:
        """
        Store documents and their embeddings in TimescaleDB.

        Args:
            documents: List of Langchain Document objects to store

        Raises:
            DocumentProcessingError: If document storage fails
        """
        try:
            # Generate embeddings for all documents
            texts = [doc.page_content for doc in documents]
            embeddings = self.embedding_factory.model.embed_documents(texts)

            # Create session
            session = self._Session()

            try:
                # Prepare batch insert
                values = []
                for doc, embedding in zip(documents, embeddings):
                    values.append({
                        "content": doc.page_content,
                        "embedding": embedding,
                        "metadata": doc.metadata,
                        "created_at": datetime.utcnow()
                    })

                # Perform batch insert
                if values:
                    stmt = text("""
                        INSERT INTO document_embeddings (content, embedding, metadata, created_at)
                        VALUES (:content, :embedding::vector, :metadata::jsonb, :created_at)
                    """)
                    session.execute(stmt, values)
                    session.commit()
                    
                logger.info(f"Successfully stored {len(documents)} documents in TimescaleDB")
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()

        except Exception as e:
            raise DocumentProcessingError(
                "Failed to store documents in TimescaleDB",
                details={"error": str(e)}
            )

    def similarity_search(self, query: str, k: int = 4, time_range: Dict = None) -> List[Document]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: The query text
            k: Number of documents to return
            time_range: Optional time range filter, e.g., {"start": "2024-01-01", "end": "2024-12-31"}

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
                # Build query with optional time range filter
                query_str = """
                    SELECT content, metadata, 
                           1 - (embedding <=> :query_embedding::vector) as similarity
                    FROM document_embeddings
                    WHERE 1=1
                """
                params = {"query_embedding": query_embedding, "k": k}

                if time_range:
                    if time_range.get("start"):
                        query_str += " AND created_at >= :start_time"
                        params["start_time"] = time_range["start"]
                    if time_range.get("end"):
                        query_str += " AND created_at <= :end_time"
                        params["end_time"] = time_range["end"]

                query_str += """
                    ORDER BY embedding <=> :query_embedding::vector
                    LIMIT :k
                """

                # Execute search
                result = session.execute(text(query_str), params)

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
