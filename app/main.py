from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import settings
from app.core.exceptions import RAGAssistantException
from app.core.logging import logger
from app.ingestion.pipeline import ingestion_pipeline
from app.retrieval.vectorstore import vector_store_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown routines."""
    logger.info("==================================================================")
    logger.info(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"   LLM Provider: {settings.LLM_PROVIDER.value.upper()} ({settings.get_resolved_llm_model()})")
    logger.info(f"   Embedding Provider: {settings.EMBEDDING_PROVIDER.value.upper()} ({settings.get_resolved_embedding_model()})")
    logger.info(f"   Vector Database: ChromaDB persistent at '{settings.CHROMA_PERSIST_DIRECTORY}'")
    logger.info("==================================================================")

    # Auto-bootstrap corpus ingestion on first startup if vector collection is empty
    try:
        total_chunks = vector_store_manager.get_total_chunk_count()
        corpus_dir = Path(settings.CORPUS_DIRECTORY)
        if total_chunks == 0 and corpus_dir.exists():
            logger.info("Vector database is empty. Auto-ingesting default technical corpus from data/corpus/...")
            results = ingestion_pipeline.ingest_directory(corpus_dir)
            logger.info(f"Auto-ingested {len(results)} technical documents ({vector_store_manager.get_total_chunk_count()} chunks total).")
    except Exception as e:
        logger.warning(f"Auto-ingestion check encountered an issue: {e}")

    yield

    logger.info("🛑 Shutting down RAG Assistant application.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise-grade Self-Corrective RAG Documentation Assistant with LangGraph, Document Grading, and Grounded Generation.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Exception Handler
@app.exception_handler(RAGAssistantException)
async def rag_assistant_exception_handler(request: Request, exc: RAGAssistantException):
    logger.error(f"RAG Application Error: {exc.message} | Details: {exc.details}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_type": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details
        }
    )


# Include API v1 Routes
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["Assistant & Ingestion"])
# Include root routes for PDF compatibility (e.g. POST /query directly)
app.include_router(api_router, tags=["Root Endpoints"])


# Serve Interactive Web Dashboard
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/ui", response_class=HTMLResponse, tags=["Dashboard"])
    async def serve_ui():
        index_file = static_dir / "index.html"
        return FileResponse(index_file)

    @app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
    async def root_redirect():
        return FileResponse(static_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
