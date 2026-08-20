from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator


# ------------------------------------------------------------------------------
# Query Schemas
# ------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Payload for submitting a question to the assistant."""
    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="The user question to be answered from technical documentation.",
        examples=["How do you define path parameters and query parameters in FastAPI?"]
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional session ID to maintain conversation state."
    )
    max_retries: Optional[int] = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of self-corrective retrieval retries allowed."
    )

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("Query string cannot be empty or whitespace only.")
        return clean


class CitationItemResponse(BaseModel):
    """Structured citation item."""
    source: str = Field(description="Filename or URL of the source document.")
    doc_title: str = Field(description="Human readable title of the document.")
    section_title: str = Field(description="Heading or section name containing the evidence.")
    chunk_id: Optional[str] = Field(default=None, description="Unique chunk identifier.")


class ExecutionTraceStepResponse(BaseModel):
    """Observable step execution details from LangGraph."""
    node_name: str
    step_number: int
    details: Dict[str, Any]
    timestamp: str


class QueryResponse(BaseModel):
    """Structured response containing answer, citations, and execution telemetry."""
    query: str
    answer: str
    citations: List[CitationItemResponse]
    query_type: str
    retry_count: int
    web_search_used: bool
    is_grounded: bool
    groundedness_score: float
    groundedness_reason: Optional[str] = None
    status: str
    execution_trace: List[ExecutionTraceStepResponse] = Field(default_factory=list)
    latency_seconds: float


# ------------------------------------------------------------------------------
# Ingestion Schemas
# ------------------------------------------------------------------------------

class IngestUrlRequest(BaseModel):
    """Payload for indexing web documentation via URL."""
    url: str = Field(
        ...,
        description="Public URL of the documentation page to fetch and index.",
        examples=["https://fastapi.tiangolo.com/tutorial/first-steps/"]
    )
    doc_title: Optional[str] = Field(
        default=None,
        description="Optional title override for the indexed document."
    )

    @field_validator("url")
    @classmethod
    def validate_url_scheme(cls, v: str) -> str:
        clean = v.strip()
        if not (clean.startswith("http://") or clean.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")
        return clean


class IngestTextRequest(BaseModel):
    """Payload for indexing raw markdown or text directly."""
    text: str = Field(..., min_length=10, description="Raw text or markdown content.")
    source_name: str = Field(..., min_length=2, description="Identifier for this document.")
    doc_title: Optional[str] = Field(default=None, description="Display title for document.")


class IngestResponse(BaseModel):
    """Response returned after successful document indexing."""
    source: str
    doc_title: str
    chunks_created: int
    status: str = "indexed"


# ------------------------------------------------------------------------------
# Document Corpus Schemas
# ------------------------------------------------------------------------------

class DocumentItemResponse(BaseModel):
    """Metadata summary of an indexed document."""
    source: str
    title: str
    source_type: str
    total_chunks: int
    sections: List[str]
    last_ingested: Optional[str] = None


class DocumentListResponse(BaseModel):
    """List of all documents currently indexed in ChromaDB."""
    total_documents: int
    total_chunks: int
    documents: List[DocumentItemResponse]


# ------------------------------------------------------------------------------
# Feedback Schemas
# ------------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """User feedback submission."""
    rating: Literal["up", "down", "+1", "-1", "positive", "negative"] = Field(
        ...,
        description="Thumbs up or thumbs down rating."
    )
    query_id: Optional[str] = Field(default=None, description="Optional ID of the query.")
    query: Optional[str] = Field(default=None, description="Question asked by the user.")
    answer: Optional[str] = Field(default=None, description="Assistant answer being rated.")
    comment: Optional[str] = Field(default=None, max_length=1000, description="Optional user comment.")


class FeedbackResponse(BaseModel):
    """Feedback submission confirmation."""
    id: str
    query_id: Optional[str]
    rating: str
    comment: Optional[str]
    created_at: str
    status: str


class FeedbackSummaryResponse(BaseModel):
    """Aggregate feedback analytics."""
    total_feedback: int
    positive_count: int
    negative_count: int
    satisfaction_rate_percent: float


# ------------------------------------------------------------------------------
# System & Diagnostic Schemas
# ------------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """System health check and provider readiness."""
    status: str
    llm_provider: str
    llm_model: str
    embedding_provider: str
    indexed_chunks: int
    version: str
    timestamp: str
