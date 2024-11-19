from typing import Optional


class RAGPipelineError(Exception):
    """Base exception for RAG pipeline errors with improved error messages."""

    def __init__(self, message: str, component: str, details: Optional[dict] = None):
        self.component = component
        self.details = details or {}
        self.message = self._format_message(message)
        super().__init__(self.message)

    def _format_message(self, message: str) -> str:
        """Format the error message with component and details."""
        error_msg = f"[{self.component}] {message}"
        if self.details:
            error_msg += "\nDetails:"
            for key, value in self.details.items():
                error_msg += f"\n  - {key}: {value}"
        return error_msg


class DocumentProcessingError(RAGPipelineError):
    """Error during document processing."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "Document Processing", details)


class RetrievalError(RAGPipelineError):
    """Error during document retrieval."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "Retrieval", details)


class GenerationError(RAGPipelineError):
    """Error during response generation."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "Generation", details)


class RoutingError(RAGPipelineError):
    """Error during query routing."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, "Routing", details)
