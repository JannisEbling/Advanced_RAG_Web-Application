"""
Custom exceptions for the RAG pipeline.
"""

from .exception import (
    RAGPipelineError,
    DocumentProcessingError,
    RetrievalError,
    GenerationError,
    RoutingError,
)

__all__ = [
    'RAGPipelineError',
    'DocumentProcessingError',
    'RetrievalError',
    'GenerationError',
    'RoutingError',
]

import sys

from src.logging import logger


class CustomException(Exception):
    def __init__(self, error_message, error_details: sys):
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occured in python script name [{self.file_name}] line number [{self.lineno}] error message [{str(self.error_message)}]"


if __name__ == "__main__":
    try:
        logger.logging.info("Test exception handling")
        a = 1 / 0
    except Exception as e:
        raise CustomException(e, sys) from e
