import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api.dependencies import (
    get_feedback_service,
    get_ingestion_pipeline,
    get_rag_workflow,
    get_vector_store,
)
from app.api.schemas import (
    CitationItemResponse,
    DocumentItemResponse,
    DocumentListResponse,
    ExecutionTraceStepResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackSummaryResponse,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
    IngestUrlRequest,
    QueryRequest,
    QueryResponse,
)
from app.config import settings
from app.core.exceptions import IngestionError
from app.core.logging import logger
from app.graph.workflow import get_mermaid_graph_markdown
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.vectorstore import VectorStoreManager
from app.services.feedback_service import FeedbackService

router = APIRouter()


# ------------------------------------------------------------------------------
# 1. Query Endpoint
# ------------------------------------------------------------------------------

@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a natural language question",
    description="Processes question through LangGraph workflow: Query Analysis -> Retrieval -> Grading -> Self-Correction/Rewrite -> Generation -> Groundedness Check."
)
async def query_assistant(
    request: QueryRequest,
    rag_workflow=Depends(get_rag_workflow)
) -> QueryResponse:
    start_time = time.perf_counter()
    session_id = request.thread_id or str(uuid.uuid4())
    logger.info(f"Incoming /query request [Session: {session_id}]: '{request.query}'")

    initial_state = {
        "original_query": request.query,
        "current_query": request.query,
        "query_type": "general",
        "technical_keywords": [],
        "retrieved_documents": [],
        "relevant_documents": [],
        "filtered_out_documents": [],
        "retry_count": 0,
        "max_retries": request.max_retries or settings.MAX_RETRIES,
        "web_search_used": False,
        "generation": "",
        "citations": [],
        "is_grounded": True,
        "groundedness_score": 1.0,
        "groundedness_reason": None,
        "status": "in_progress",
        "error_message": None,
        "execution_trace": []
    }

    config = {"configurable": {"thread_id": session_id}}

    try:
        final_state = rag_workflow.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"LangGraph execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution error: {str(e)}"
        )

    latency = round(time.perf_counter() - start_time, 3)

    # Format structured citations
    citations_response = [
        CitationItemResponse(
            source=c.source,
            doc_title=c.doc_title,
            section_title=c.section_title,
            chunk_id=c.chunk_id
        )
        for c in final_state.get("citations", [])
    ]

    # Format trace steps
    trace_response = [
        ExecutionTraceStepResponse(
            node_name=t.node_name,
            step_number=t.step_number,
            details=t.details,
            timestamp=t.timestamp
        )
        for t in final_state.get("execution_trace", [])
    ]

    return QueryResponse(
        query=request.query,
        answer=final_state.get("generation", "No answer generated."),
        citations=citations_response,
        query_type=final_state.get("query_type", "general"),
        retry_count=final_state.get("retry_count", 0),
        web_search_used=final_state.get("web_search_used", False),
        is_grounded=final_state.get("is_grounded", True),
        groundedness_score=final_state.get("groundedness_score", 1.0),
        groundedness_reason=final_state.get("groundedness_reason"),
        status=final_state.get("status", "success"),
        execution_trace=trace_response,
        latency_seconds=latency
    )


# ------------------------------------------------------------------------------
# 2. Ingestion Endpoints (Files, URLs, Text)
# ------------------------------------------------------------------------------

@router.post(
    "/ingest/file",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest uploaded file (.md, .txt, .pdf)",
    description="Uploads a local technical document file, splits it into code-aware chunks, and indexes embeddings into ChromaDB."
)
async def ingest_file_upload(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file selected.")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["md", "markdown", "txt", "pdf"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '.{ext}'. Supported: .md, .txt, .pdf"
        )

    try:
        content = await file.read()
        result = pipeline.ingest_bytes(content, file.filename)
        return IngestResponse(
            source=result["source"],
            doc_title=result["doc_title"],
            chunks_created=result["chunks_created"],
            status="indexed"
        )
    except Exception as e:
        logger.error(f"File upload ingestion failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/ingest/url",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest documentation from a public URL",
    description="Fetches, cleans, chunks, and indexes web documentation from a given URL."
)
async def ingest_url_endpoint(
    payload: IngestUrlRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> IngestResponse:
    try:
        result = await pipeline.ingest_url(str(payload.url))
        return IngestResponse(
            source=result["source"],
            doc_title=payload.doc_title or result["doc_title"],
            chunks_created=result["chunks_created"],
            status="indexed"
        )
    except IngestionError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.message)
    except Exception as e:
        logger.error(f"URL ingestion failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/ingest/text",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest raw markdown/text snippet",
    description="Directly index raw markdown or text documentation."
)
async def ingest_text_endpoint(
    payload: IngestTextRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> IngestResponse:
    try:
        result = pipeline.ingest_text(
            text=payload.text,
            source_name=payload.source_name,
            doc_title=payload.doc_title
        )
        return IngestResponse(
            source=result["source"],
            doc_title=result["doc_title"],
            chunks_created=result["chunks_created"],
            status="indexed"
        )
    except Exception as e:
        logger.error(f"Raw text ingestion failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Unified /ingest endpoint to satisfy generic POST /ingest requirement
@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generic document ingestion",
    description="Accepts IngestUrlRequest payload to index web documentation."
)
async def generic_ingest_endpoint(
    payload: IngestUrlRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
) -> IngestResponse:
    return await ingest_url_endpoint(payload, pipeline)


# ------------------------------------------------------------------------------
# 3. Document Corpus Listing
# ------------------------------------------------------------------------------

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all indexed documents in the corpus",
    description="Returns an aggregated summary of all indexed documentation files, their section headings, and chunk counts."
)
async def list_documents(
    vector_store: VectorStoreManager = Depends(get_vector_store)
) -> DocumentListResponse:
    docs_summary = vector_store.list_indexed_documents()
    total_chunks = vector_store.get_total_chunk_count()

    doc_items = [
        DocumentItemResponse(
            source=item["source"],
            title=item["title"],
            source_type=item["source_type"],
            total_chunks=item["total_chunks"],
            sections=item["sections"],
            last_ingested=item.get("last_ingested")
        )
        for item in docs_summary
    ]

    return DocumentListResponse(
        total_documents=len(doc_items),
        total_chunks=total_chunks,
        documents=doc_items
    )


@router.delete(
    "/documents/{source_name}",
    status_code=status.HTTP_200_OK,
    summary="Delete document from corpus",
    description="Removes all vector embeddings and chunks associated with a specific document source."
)
async def delete_document(
    source_name: str,
    vector_store: VectorStoreManager = Depends(get_vector_store)
) -> Dict[str, Any]:
    deleted_count = vector_store.delete_by_source(source_name)
    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{source_name}' not found in index."
        )
    return {
        "source": source_name,
        "chunks_deleted": deleted_count,
        "status": "deleted"
    }


# ------------------------------------------------------------------------------
# 4. User Feedback Endpoints
# ------------------------------------------------------------------------------

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit user feedback on an answer",
    description="Records thumbs up/down rating and optional comments to an SQLite database."
)
async def submit_feedback(
    payload: FeedbackRequest,
    feedback_svc: FeedbackService = Depends(get_feedback_service)
) -> FeedbackResponse:
    res = feedback_svc.record_feedback(
        rating=payload.rating,
        query_id=payload.query_id,
        query=payload.query,
        answer=payload.answer,
        comment=payload.comment
    )
    return FeedbackResponse(
        id=res["id"],
        query_id=res["query_id"],
        rating=res["rating"],
        comment=res["comment"],
        created_at=res["created_at"],
        status=res["status"]
    )


@router.get(
    "/feedback",
    response_model=FeedbackSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get feedback metrics and satisfaction score",
    description="Returns total thumbs up/down counts and overall user satisfaction rate."
)
async def get_feedback_metrics(
    feedback_svc: FeedbackService = Depends(get_feedback_service)
) -> FeedbackSummaryResponse:
    summary = feedback_svc.get_summary()
    return FeedbackSummaryResponse(
        total_feedback=summary["total_feedback"],
        positive_count=summary["positive_count"],
        negative_count=summary["negative_count"],
        satisfaction_rate_percent=summary["satisfaction_rate_percent"]
    )


# ------------------------------------------------------------------------------
# 5. System Health & Visualization
# ------------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System health check",
    description="Returns runtime status, active LLM model, and indexed chunk metrics."
)
async def health_check(
    vector_store: VectorStoreManager = Depends(get_vector_store)
) -> HealthResponse:
    return HealthResponse(
        status="healthy",
        llm_provider=settings.LLM_PROVIDER.value,
        llm_model=settings.get_resolved_llm_model(),
        embedding_provider=settings.EMBEDDING_PROVIDER.value,
        indexed_chunks=vector_store.get_total_chunk_count(),
        version=settings.VERSION,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get(
    "/graph/visualize",
    response_class=PlainTextResponse,
    summary="Visualize LangGraph workflow",
    description="Returns Mermaid markdown diagram depicting nodes, conditional edges, and fallback routing."
)
async def visualize_graph() -> str:
    return get_mermaid_graph_markdown()
