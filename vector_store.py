import logging
import time
from typing import Any, List, Optional, Tuple, Union, Literal
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from langchain.docstore.document import Document
from timescale_vector.client import uuid_from_time
import psycopg2.extras
import pandas as pd
from src.config.settings import get_settings
from timescale_vector import client
from src.components.embedding_factory import EmbeddingFactory
from src.log_utils import logger
from src.exception.exception import RetrievalError


class VectorStore:
    def __init__(self, provider: str = "azure"):
        """Initialize vector store with embedding provider."""
        self.settings = get_settings()
        self.embedding_model = EmbeddingFactory(provider).model

        # Initialize vector store clients for documents and figures
        self.doc_client = client.Sync(
            self.settings.vector_store.service_url,
            self.settings.vector_store.document_table,
            self.settings.vector_store.embedding_dimensions,
            time_partition_interval=self.settings.vector_store.time_partition_interval,
        )

        # For figures, we'll use a direct database connection
        self.db_conn = psycopg2.connect(self.settings.vector_store.service_url)
        psycopg2.extras.register_uuid()

    def upsert(
        self,
        data: Union[pd.DataFrame, List[Any]],
        content_type: Literal["document", "figure", "formula"] = "document",
    ) -> None:
        """
        Insert or update records in the vector store.

        Args:
            data: Data to insert
                If DataFrame: Must have columns [id, metadata, contents, embedding] for documents
                            or [id, figure_reference, description, metadata] for figures
                If Documents: List of langchain Document objects with page_content and metadata
                If List[tuple]: For figures, list of tuples (id, figure_reference, description, metadata)
            content_type: Type of content being inserted ("document" or "figure")
        """
        if content_type == "figure":
            # For figures, use direct SQL insertion
            query = """
                INSERT INTO figures (id, image_path, caption, figure_ref)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    image_path = EXCLUDED.image_path,
                    caption = EXCLUDED.caption,
                    figure_ref = EXCLUDED.figure_ref;
            """
            # Convert metadata to JSON format
            records = [
                (
                    record[0],  # id
                    record[1],  # image_path
                    record[2]["content"],  # caption
                    record[3],  # figure_ref
                )
                for record in data
            ]

            with self.db_conn.cursor() as cur:
                cur.executemany(query, records)
            self.db_conn.commit()
            logging.info(f"Inserted {len(data)} figures into figures table")
            return

        if content_type == "formula":
            # For formulas, use direct SQL insertion
            query = """
                INSERT INTO formulas (id, formula_reference, content)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    formula_reference = EXCLUDED.formula_reference,
                    content = EXCLUDED.content;
            """
            # Convert metadata to JSON format
            records = [
                (
                    record[0],  # id
                    record[1],  # formula_reference
                    record[2],  # content
                )
                for record in data
            ]

            with self.db_conn.cursor() as cur:
                cur.executemany(query, records)
            self.db_conn.commit()
            logging.info(f"Inserted {len(data)} formulas into formulas table")
            return

        # For documents, use the standard vector store functionality
        if isinstance(data, pd.DataFrame):
            records = data.to_records(index=False)
        else:
            # Convert Documents to records format
            records = []
            for doc in data:
                # Generate embedding for the document content
                embedding = self.embedding_model.embed_query(doc.page_content)

                # Create record as tuple with required fields
                record = (
                    str(uuid_from_time(datetime.now())),
                    doc.metadata,
                    doc.page_content,
                    embedding,
                )
                records.append(record)

        self.doc_client.upsert(records)
        logging.info(
            f"Inserted {len(records)} documents into {self.settings.vector_store.document_table}"
        )

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = False,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.

        More info:
            https://github.com/timescale/docs/blob/latest/ai/python-interface-for-pgvector-and-timescale-vector.md

        Args:
            query_text: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary or list of dictionaries for equality-based metadata filtering.
            predicates: A Predicates object for complex metadata filtering.
                - Predicates objects are defined by the name of the metadata key, an operator, and a value.
                - Operators: ==, !=, >, >=, <, <=
                - & is used to combine multiple predicates with AND operator.
                - | is used to combine multiple predicates with OR operator.
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.

        Basic Examples:
            Basic search:
                vector_store.search("What are your shipping options?")
            Search with metadata filter:
                vector_store.search("Shipping options", metadata_filter={"category": "Shipping"})
        
        Predicates Examples:
            Search with predicates:
                vector_store.search("Pricing", predicates=client.Predicates("price", ">", 100))
            Search with complex combined predicates:
                complex_pred = (client.Predicates("category", "==", "Electronics") & client.Predicates("price", "<", 1000)) | \
                               (client.Predicates("category", "==", "Books") & client.Predicates("rating", ">=", 4.5))
                vector_store.search("High-quality products", predicates=complex_pred)
        
        Time-based filtering:
            Search with time range:
                vector_store.search("Recent updates", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)))
        """
        query_embedding = self.get_embedding(query_text)

        start_time = time.time()

        search_args = {
            "limit": limit,
        }

        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = self.doc_client.search(query_embedding, **search_args)
        elapsed_time = time.time() - start_time

        logging.info(f"Vector search completed in {elapsed_time:.3f} seconds")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

    def retrieve(
        self, state: Dict[str, Any], limit: int = 3
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents and associated figures/formulas based on the query.

        Args:
            state: Current workflow state
            limit: Maximum number of documents to retrieve

        Returns:
            Updated state with documents, figures, and formulas
        """
        try:
            query = state.question

            # Initialize lists in state if they don't exist
            if not hasattr(state, "figures"):
                state.figures = []
            if not hasattr(state, "formulas"):
                state.formulas = []

            # Get documents from vector search
            documents = self.search(query, limit=limit)
            state.documents = [
                Document(page_content=result[2], metadata=result[1])
                for result in documents
            ]

            # Collect unique references
            figure_references = set()
            formula_references = set()
            for doc in documents:
                metadata = doc[1]  # metadata is in the second position
                if "figure_references" in metadata:
                    # Handle both single references and lists of references
                    refs = metadata["figure_references"]
                    if isinstance(refs, list):
                        figure_references.update(refs)
                    else:
                        figure_references.add(refs)

                if "formula_references" in metadata:
                    # Handle both single references and lists of references
                    refs = metadata["formula_references"]
                    if isinstance(refs, list):
                        formula_references.update(refs)
                    else:
                        formula_references.add(refs)

            # Retrieve figures and formulas
            for figure_ref in figure_references:
                figure_data = self.get_figure(figure_ref)
                if figure_data:
                    state.figures.append(figure_data)

            for formula_ref in formula_references:
                formula_data = self.get_formula(formula_ref)
                if formula_data:
                    state.formulas.append(formula_data)

            return state

        except Exception as e:
            raise RetrievalError(
                "Failed to retrieve documents",
                details={"error": str(e)},
            )

    def create_tables(self) -> None:
        """Create the necessary tables in the database if they don't exist."""
        try:
            # Create document table with embeddings
            self.doc_client.create_tables()

            # Create figure table without embeddings
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS figures (
                        id UUID PRIMARY KEY,
                        image_path TEXT,
                        caption TEXT,
                        figure_ref TEXT UNIQUE
                    );
                """)
                self.db_conn.commit()

            # Create formula table without embeddings
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS formulas (
                        id UUID PRIMARY KEY,
                        formula_reference TEXT,
                        content TEXT
                    );
                """)
                self.db_conn.commit()

            logger.info("Vector store tables ready")
        except Exception as e:
            raise RetrievalError(
                "Failed to create vector store tables",
                details={"error": str(e)},
            )

    def create_index(self) -> None:
        """Create the StreamingDiskANN index to speed up similarity search if it doesn't exist."""
        try:
            # Check if index already exists
            try:
                self.doc_client.create_embedding_index(client.DiskAnnIndex())
                logger.info("Successfully created vector store index for documents")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info("Vector store index for documents already exists")
                else:
                    raise e
        except Exception as e:
            raise RetrievalError(
                "Failed to create vector store index",
                details={"error": str(e)},
            )

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.doc_client.drop_embedding_index()

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
        content_type: Literal["document", "figure"] = "document",
    ) -> None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.
            content_type: Type of content to delete ("document" or "figure")
        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.

        Examples:
            Delete by IDs:
                vector_store.delete(ids=["8ab544ae-766a-11ef-81cb-decf757b836d"])

            Delete by metadata filter:
                vector_store.delete(metadata_filter={"category": "Shipping"})

            Delete all records:
                vector_store.delete(delete_all=True)
        """
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )

        # Select appropriate client based on content type
        if content_type == "figure":
            # For figures, use direct SQL deletion
            if delete_all:
                query = "DELETE FROM figures;"
            elif ids:
                query = "DELETE FROM figures WHERE id IN %s;"
                ids = tuple(ids)
            elif metadata_filter:
                query = "DELETE FROM figures WHERE metadata @> %s;"
                metadata_filter = psycopg2.extras.Json(metadata_filter)

            with self.db_conn.cursor() as cur:
                cur.execute(query, (ids, metadata_filter))
            self.db_conn.commit()
            logging.info(f"Deleted records from figures table")
            return

        vec_client = self.doc_client

        if delete_all:
            vec_client.delete_all()
            logging.info(f"Deleted all records from {vec_client.table}")
        elif ids:
            vec_client.delete_by_ids(ids)
            logging.info(f"Deleted {len(ids)} records from {vec_client.table}")
        elif metadata_filter:
            vec_client.delete_by_metadata(metadata_filter)
            logging.info(
                f"Deleted records matching metadata filter from {vec_client.table}"
            )

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using the embedding factory.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.

        Raises:
            RetrievalError: If embedding generation fails
        """
        text = text.replace("\n", " ")
        start_time = time.time()

        try:
            embedding = self.embedding_model.embed_query(text)
            elapsed_time = time.time() - start_time
            logger.info(f"Embedding generated in {elapsed_time:.3f} seconds")
            return embedding
        except Exception as e:
            raise RetrievalError(
                "Failed to generate embedding",
                details={
                    "text_length": len(text),
                    "error": str(e),
                },
            )

    def get_figure(self, figure_reference: str) -> Dict[str, Any]:
        """
        Retrieve figure data from the database.

        Args:
            figure_reference: Reference ID of the figure

        Returns:
            Dictionary containing figure data and metadata
        """
        try:
            query = """
                SELECT image_path, caption, figure_ref
                FROM figures
                WHERE figure_ref = %s
            """
            with self.db_conn.cursor() as cur:
                cur.execute(query, (figure_reference,))
                result = cur.fetchone()

                if result:
                    return {
                        "reference": result[2],  # figure_ref
                        "path": result[0],  # image_path
                        "caption": result[1],  # caption
                    }
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve figure {figure_reference}: {str(e)}")
            return None

    def get_formula(self, formula_reference: str) -> Dict[str, Any]:
        """
        Retrieve formula data from the database.

        Args:
            formula_reference: Reference ID of the formula

        Returns:
            Dictionary containing formula data and metadata
        """
        try:
            query = """
                SELECT formula_reference, content
                FROM formulas
                WHERE formula_reference = %s
            """
            with self.db_conn.cursor() as cur:
                cur.execute(query, (formula_reference,))
                result = cur.fetchone()

                if result:
                    return {
                        "reference": formula_reference,
                        "data": result[0],
                        "metadata": result[1],
                    }
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve formula {formula_reference}: {str(e)}")
            return None

    def __del__(self):
        """Cleanup database connections"""
        if hasattr(self, "db_conn"):
            self.db_conn.close()
