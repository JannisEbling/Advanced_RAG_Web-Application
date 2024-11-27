import os
from pathlib import Path
from langchain.docstore.document import Document
from src.database.postgres_handler import PostgresHandler
from src.components.doc_cleaner import split_into_chapters
from src.logging import logger

def populate_database_example():
    """
    Example script showing how to populate the PostgreSQL database with documents and embeddings.
    """
    # Initialize PostgreSQL handler
    db_handler = PostgresHandler(provider="azure")  # or "huggingface"

    # Example: Process a PDF document
    doc_path = "data/sample.pdf"  # Replace with your document path
    try:
        # Split document into chapters
        documents = split_into_chapters(doc_path)
        logger.info(f"Split document into {len(documents)} chapters")

        # Process and store documents with embeddings
        db_handler.store_documents(documents)
        logger.info("Successfully stored documents with embeddings in PostgreSQL")

        # Example: Perform similarity search
        query = "What is machine learning?"
        similar_docs = db_handler.similarity_search(query, k=3)
        
        print("\nSimilar documents for query:", query)
        for i, doc in enumerate(similar_docs, 1):
            print(f"\nDocument {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

if __name__ == "__main__":
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    import sys
    sys.path.append(str(project_root))

    populate_database_example()
