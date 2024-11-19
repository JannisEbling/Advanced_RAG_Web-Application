import os
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

os.environ["CURL_CA_BUNDLE"] = ""
load_dotenv()


def add_docs(doc_path: str, provider: str = "azure"):
    """
    Adds a document to the vectorstore

    Args:
        doc_path (str): Path to the document
        provider (str): The embedding provider to use (e.g., "azure", "huggingface")
    """
    try:
        logger.info("Starting document processing for: %s", doc_path)
        
        # Initialize vector store with specified provider
        vector_store = VectorStore(provider=provider)

        # Split documents by chapters
        documents = split_into_chapters(doc_path)
        logger.info("Split document into %d chapters", len(documents))

        # Create the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            add_start_index=True
        )

        processed_documents = []
        # Process each chapter (skipping the first as it's usually the table of contents)
        for doc in documents[1:]:
            chapter_title = doc.metadata.get("chapter", "")
            chapter_text = doc.page_content

            # Split the document text into chunks
            chunks = text_splitter.split_text(chapter_text)
            logger.debug("Split chapter '%s' into %d chunks", chapter_title, len(chunks))

            # Process each chunk
            for chunk in chunks:
                # Clean the chunk
                chunk_content = replace_new_line_with_space(chunk)
                chunk_content = filter_recurrent_obsolescences_with_remove(chunk_content)
                chunk_content = replace_t_with_space(chunk_content)
                original_chunk = chunk_content

                # Rewrite the chunk
                chunk_content = rewrite_chunk(chunk_content, chapter_title)

                # Create document with metadata
                processed_doc = Document(
                    page_content=chunk_content,
                    metadata={
                        "chapter": chapter_title,
                        "original_chunk": original_chunk,
                        "source": doc_path,
                        "embedding_provider": provider
                    }
                )
                processed_documents.append(processed_doc)

        # Add documents to vectorstore
        logger.info("Adding %d processed chunks to vectorstore", len(processed_documents))
        vector_store.add_documents(processed_documents)
        logger.info("Successfully added documents to vectorstore")

    except Exception as e:
        logger.error("Failed to add documents to vectorstore", exc_info=True)
        raise


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    add_docs("C:/Projekte/Data_Science/Advanced_RAG_Web-Application/data/Chapter4.pdf")
