from typing import Any
from app.graph.workflow import compiled_rag_app
from app.ingestion.pipeline import IngestionPipeline, ingestion_pipeline
from app.retrieval.vectorstore import VectorStoreManager, vector_store_manager
from app.services.feedback_service import FeedbackService, feedback_service


def get_vector_store() -> VectorStoreManager:
    """Dependency provider for persistent ChromaDB manager."""
    return vector_store_manager


def get_ingestion_pipeline() -> IngestionPipeline:
    """Dependency provider for document ingestion pipeline."""
    return ingestion_pipeline


def get_feedback_service() -> FeedbackService:
    """Dependency provider for SQLite feedback service."""
    return feedback_service


def get_rag_workflow() -> Any:
    """Dependency provider for compiled LangGraph application."""
    return compiled_rag_app
