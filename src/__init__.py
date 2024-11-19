"""
Advanced RAG Web Application

A sophisticated Retrieval-Augmented Generation (RAG) system with web interface.
Features include document processing, semantic search, and AI-powered response generation.
"""

from . import components
from . import pipelines

# Direct imports for commonly used components
from .logging import logger, setup_logging
from .exception.exception import (
    RAGPipelineError,
    DocumentProcessingError,
    RetrievalError,
    GenerationError,
    RoutingError,
)

__version__ = "1.0.0"

# Initialize logging
setup_logging()

__all__ = [
    # Submodules
    'components',
    'pipelines',
    
    # Direct access to logger
    'logger',
    'setup_logging',
    
    # Direct access to exceptions
    'RAGPipelineError',
    'DocumentProcessingError',
    'RetrievalError',
    'GenerationError',
    'RoutingError',
]