from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents an individual document chunk retrieved from the vector store."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity_score: float = 0.0
    is_relevant: Optional[bool] = None
    relevance_score: Optional[float] = None
    grading_reason: Optional[str] = None


class Citation(BaseModel):
    """Structured citation referencing a source and section."""
    source: str
    doc_title: str
    section_title: str
    chunk_id: Optional[str] = None


class ExecutionTraceStep(BaseModel):
    """Step execution log entry for end-to-end graph observability."""
    node_name: str
    step_number: int
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str


class GraphState(TypedDict):
    """
    Explicit LangGraph state schema for the self-corrective RAG pipeline.
    Tracks query lifecycle, retrieved context, grading outcomes, retry budget, and output.
    """
    # Query lifecycle
    original_query: str
    current_query: str
    query_type: str                         # 'conceptual' | 'how-to' | 'troubleshooting' | 'api_reference' | 'general'
    technical_keywords: List[str]

    # Context & Grading
    retrieved_documents: List[DocumentChunk]
    relevant_documents: List[DocumentChunk]
    filtered_out_documents: List[DocumentChunk]

    # Retry loop & Fallback control
    retry_count: int
    max_retries: int
    web_search_used: bool

    # Generation & Groundedness
    generation: str
    citations: List[Citation]
    is_grounded: bool
    groundedness_score: float
    groundedness_reason: Optional[str]

    # Execution telemetry
    status: str                             # 'success' | 'insufficient_context' | 'fallback_used' | 'error'
    error_message: Optional[str]
    execution_trace: List[ExecutionTraceStep]
