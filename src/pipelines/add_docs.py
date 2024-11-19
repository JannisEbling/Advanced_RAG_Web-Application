import os
import sys
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.components.chunk_rewriter import rewrite_chunk
from src.components.doc_cleaner import (
    filter_recurrent_obsolescences_with_remove,
    replace_new_line_with_space,
    replace_t_with_space,
    split_into_chapters,
)
from src.components.vector_store import VectorStore
from src.logging import logger
from src.exceptions import DocumentProcessingError

# Load environment variables
load_dotenv()


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

    chunk_size: int = Field(default=2000, description="Target size for text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    min_chunk_size: int = Field(
        default=100, description="Minimum chunk size to process"
    )
    skip_chapters: List[str] = Field(
        default_factory=list, description="Chapter titles to skip"
    )


def process_document(
    doc_path: str, provider: str = "azure", config: Optional[ProcessingConfig] = None
) -> None:
    """
    Process and add a document to the vectorstore with improved error handling and logging.

    Args:
        doc_path: Path to the document
        provider: The embedding provider to use (e.g., "azure", "huggingface")
        config: Optional processing configuration

    Raises:
        DocumentProcessingError: If document processing fails
    """
    if config is None:
        config = ProcessingConfig()

    try:
        logger.info(f"Starting document processing for: {doc_path}")
        if not Path(doc_path).exists():
            raise DocumentProcessingError(
                "Document not found", details={"path": doc_path}
            )

        vector_store = VectorStore(provider=provider)

        # Split document into chapters
        try:
            documents = split_into_chapters(doc_path)
            logger.info(f"Split document into {len(documents)} chapters")
        except Exception as e:
            raise DocumentProcessingError(
                "Failed to split document into chapters",
                details={"path": doc_path, "error": str(e)},
            )

        # Create text splitter with configuration
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )

        processed_documents = []
        processing_errors = []

        # Process each chapter (skip first if it's table of contents)
        for doc_index, doc in enumerate(documents[1:], 1):
            chapter_title = doc.metadata.get("chapter", f"Chapter_{doc_index}")

            # Skip specified chapters
            if chapter_title in config.skip_chapters:
                logger.info(f"Skipping chapter: {chapter_title}")
                continue

            try:
                # Split chapter into chunks
                chunks = text_splitter.split_text(doc.page_content)
                logger.debug(
                    f"Split chapter '{chapter_title}' into {len(chunks)} chunks"
                )

                # Process each chunk
                for chunk_index, chunk in enumerate(chunks):
                    try:
                        # Clean chunk text
                        cleaned_chunk = _clean_chunk_text(chunk)
                        if len(cleaned_chunk) < config.min_chunk_size:
                            logger.debug(f"Skipping small chunk in {chapter_title}")
                            continue

                        # Rewrite and process chunk
                        processed_chunk = rewrite_chunk(cleaned_chunk, chapter_title)

                        # Create document with metadata
                        metadata = DocumentMetadata(
                            chapter=chapter_title,
                            original_chunk=cleaned_chunk,
                            source=doc_path,
                            embedding_provider=provider,
                            chunk_index=chunk_index,
                        )

                        processed_doc = Document(
                            page_content=processed_chunk, metadata=metadata.dict()
                        )
                        processed_documents.append(processed_doc)

                    except Exception as e:
                        error_msg = (
                            f"Failed to process chunk {chunk_index} in {chapter_title}"
                        )
                        logger.warning(error_msg, exc_info=True)
                        processing_errors.append(
                            {
                                "chapter": chapter_title,
                                "chunk_index": chunk_index,
                                "error": str(e),
                            }
                        )

            except Exception as e:
                error_msg = f"Failed to process chapter: {chapter_title}"
                logger.error(error_msg, exc_info=True)
                processing_errors.append({"chapter": chapter_title, "error": str(e)})

        # Add processed documents to vectorstore
        if processed_documents:
            try:
                logger.info(
                    f"Adding {len(processed_documents)} processed chunks to vectorstore"
                )
                vector_store.add_documents(processed_documents)
                logger.info("Successfully added documents to vectorstore")
            except Exception as e:
                raise DocumentProcessingError(
                    "Failed to add documents to vectorstore",
                    details={
                        "path": doc_path,
                        "num_documents": len(processed_documents),
                        "error": str(e),
                    },
                )
        else:
            logger.warning("No documents were processed successfully")

        # Report processing errors if any occurred
        if processing_errors:
            logger.warning(
                "Document processing completed with errors",
                extra={
                    "num_errors": len(processing_errors),
                    "errors": processing_errors,
                },
            )

    except DocumentProcessingError:
        raise
    except Exception as e:
        raise DocumentProcessingError(
            "Unexpected error during document processing",
            details={"path": doc_path, "error": str(e)},
        )


def _clean_chunk_text(text: str) -> str:
    """Clean and normalize chunk text."""
    text = replace_new_line_with_space(text)
    text = filter_recurrent_obsolescences_with_remove(text)
    text = replace_t_with_space(text)
    return text.strip()


if __name__ == "__main__":
    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    try:
        # Example usage with custom configuration
        config = ProcessingConfig(
            chunk_size=1500,
            chunk_overlap=150,
            skip_chapters=["Table of Contents", "Index"],
        )

        process_document(
            "C:/Projekte/Data_Science/Advanced_RAG_Web-Application/data/Chapter4.pdf",
            config=config,
        )
    except DocumentProcessingError as e:
        logger.error("Document processing failed: %s", str(e))
