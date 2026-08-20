"""Custom exceptions for the RAG assistant application."""


class RAGAssistantException(Exception):
    """Base exception for all RAG Assistant errors."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class IngestionError(RAGAssistantException):
    """Raised when document loading, chunking, or indexing fails."""
    pass


class DocumentNotFoundError(RAGAssistantException):
    """Raised when requested document or source is not found."""
    pass


class ModelProviderError(RAGAssistantException):
    """Raised when LLM or Embedding API calls fail or fail authentication."""
    pass


class GraphExecutionError(RAGAssistantException):
    """Raised when LangGraph execution fails to complete."""
    pass
