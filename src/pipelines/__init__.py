"""
RAG pipeline implementations for document processing and query answering.
"""

from .add_docs import process_documents
from .get_answer import generate_response

__all__ = [
    'process_documents',
    'generate_response',
]