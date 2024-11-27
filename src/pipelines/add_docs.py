import os
import sys
import json
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.docstore.document import Document

from src.components.chunk_rewriter import rewrite_chunk
from src.azure_response_processor import AzureResponseProcessor
from src.doc_intel import DocumentIntelligenceClientWrapper
from src.components.doc_cleaner import (
    filter_recurrent_obsolescences_with_remove,
    replace_new_line_with_space,
    replace_t_with_space,
)
from src.components.vector_store_chroma import VectorStore
from vector_store import VectorStore as TimescaleVectorStore

import logging

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".pdf"]
BATCH_SIZE = 100  # Number of documents to process in one batch for vector store


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors with detailed information."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DocumentMetadata(BaseModel):
    """Metadata for processed documents."""

    chapter: str = Field(description="Chapter title or section name")
    original_chunk: str = Field(description="Original unprocessed chunk text")
    source: str = Field(description="Source document path")
    embedding_provider: str = Field(description="Provider used for embeddings")
    chunk_index: int = Field(description="Index of chunk within chapter")
    confidence_score: Optional[float] = Field(
        None, description="Confidence in chunk quality"
    )


class ProcessingConfig(BaseModel):
    """Configuration for document processing."""

    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    min_chunk_size: int = Field(
        default=100, description="Minimum size for a chunk to be processed"
    )
    skip_chapters: List[str] = Field(
        default_factory=list, description="Chapter titles to skip"
    )
    batch_size: int = Field(
        default=BATCH_SIZE, description="Number of documents to process in one batch"
    )


def process_documents(
    doc_folder_path: str,
    provider: str = "azure",
    config: Optional[ProcessingConfig] = None,
) -> None:
    """
    Process multiple documents from a folder.

    Args:
        doc_folder_path: Path to the folder containing documents
        provider: Vector store provider
        config: Processing configuration
    """
    doc_folder = Path(doc_folder_path)
    if not doc_folder.exists():
        raise DocumentProcessingError(
            "Folder not found", details={"path": str(doc_folder)}
        )

    # Get all PDF files
    doc_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        doc_paths.extend(doc_folder.glob(f"*{ext}"))

    if not doc_paths:
        raise DocumentProcessingError(
            f"No supported documents found in folder. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}",
            details={"path": str(doc_folder)},
        )

    logger.info(f"Found {len(doc_paths)} documents to process")

    # Process documents with progress bar
    for doc_path in tqdm(doc_paths, desc="Processing documents"):
        try:
            process_document(str(doc_path), provider=provider, config=config)
        except DocumentProcessingError as e:
            logger.error(f"Failed to process {doc_path.name}: {e.message}")
            logger.debug("Error details:", exc_info=True)
            continue


def process_document(
    doc_path: Path,
    provider: str = "azure",
    config: Optional[ProcessingConfig] = None,
) -> None:
    """
    Process and add a document to the vectorstore with improved error handling and logging.

    Args:
        doc_path: Path to the document
        provider: Vector store provider
        config: Processing configuration
    """
    if config is None:
        config = ProcessingConfig()

    doc_path = Path(doc_path)
    try:
        logger.info(f"Starting document processing for: {doc_path}")

        # Validate document type
        if doc_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise DocumentProcessingError(
                f"Unsupported document type: {doc_path.suffix}",
                details={"supported_types": SUPPORTED_EXTENSIONS},
            )

        if not doc_path.exists():
            raise DocumentProcessingError(
                "Document not found", details={"path": str(doc_path)}
            )

        vector_store = TimescaleVectorStore()
        vector_store.create_tables()
        vector_store.create_index()
        processed_documents = []

        try:
            # Check Azure credentials before initializing
            endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
            key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
            if not endpoint or not key:
                raise DocumentProcessingError(
                    "Azure Document Intelligence credentials missing",
                    details={
                        "message": "Please set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables."
                    },
                )

            documentintelligence = DocumentIntelligenceClientWrapper()
            paragraphs_data = documentintelligence.analyze_documents(str(doc_path))
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to analyse with document intelligence",
                details={"path": str(doc_path), "error": str(e)},
            )

        try:
            processor = AzureResponseProcessor()
            processor.set_response(paragraphs_data)
            documents, formulas, figures = processor.process_paragraphs()
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to process Azure response",
                details={"path": str(doc_path), "error": str(e)},
            )

        # Process documents in batches
        for doc in documents:
            try:
                doc.content = _clean_chunk_text(doc.content)
                # processed_doc = rewrite_chunk(doc)
                processed_documents.append(doc)

                # Add to vector store when batch size is reached
                if len(processed_documents) >= config.batch_size:
                    vector_store.upsert(processed_documents)
                    processed_documents = []

            except Exception as e:
                logger.error(f"Failed to process document chunk: {str(e)}")
                continue

        # Add remaining documents
        if processed_documents:
            try:
                vector_store.upsert(processed_documents)
                logger.info(
                    f"Successfully added {len(processed_documents)} documents to vectorstore"
                )
            except Exception as e:
                raise DocumentProcessingError(
                    "Failed to add documents to vectorstore",
                    details={
                        "path": str(doc_path),
                        "num_documents": len(processed_documents),
                        "error": str(e),
                    },
                )
        else:
            logger.warning("No documents were processed successfully")

    except DocumentProcessingError:
        raise
    except Exception as e:
        raise DocumentProcessingError(
            "Unexpected error during document processing",
            details={"path": str(doc_path), "error": str(e)},
        )


def _clean_chunk_text(text: str) -> str:
    """Clean and normalize chunk text."""
    text = replace_new_line_with_space(text)
    text = filter_recurrent_obsolescences_with_remove(text)
    text = replace_t_with_space(text)
    return text.strip()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
