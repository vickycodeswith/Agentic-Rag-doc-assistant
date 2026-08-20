import shutil
import tempfile
import pytest
from httpx import ASGITransport, AsyncClient

from app.graph.state import DocumentChunk
from app.ingestion.chunking import TechnicalDocumentChunker
from app.main import app
from app.retrieval.vectorstore import VectorStoreManager


@pytest.fixture
def sample_markdown():
    return """# FastAPI Dependency Injection

## Overview
FastAPI uses `Depends` to declare dependency injection in route parameters.

```python
from fastapi import Depends, FastAPI

app = FastAPI()

def get_db():
    return "db_connection"

@app.get("/items")
def read_items(db: str = Depends(get_db)):
    return {"db": db}
```

## Advanced Sub-dependencies
Dependencies can be nested hierarchically.
"""


@pytest.fixture
def temp_vector_store():
    temp_dir = tempfile.mkdtemp()
    mgr = VectorStoreManager(
        persist_directory=temp_dir,
        collection_name="test_collection"
    )
    yield mgr
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def chunker():
    return TechnicalDocumentChunker(chunk_size=400, chunk_overlap=50)


@pytest.fixture
def sample_relevant_chunk():
    return DocumentChunk(
        page_content="FastAPI uses Depends for dependency injection.",
        metadata={"source": "fastapi_test.md", "section_title": "Dependency Injection", "doc_title": "FastAPI Guide"},
        similarity_score=0.92,
        is_relevant=True,
        relevance_score=0.95
    )


@pytest.fixture
def sample_irrelevant_chunk():
    return DocumentChunk(
        page_content="A recipe for vanilla sponge cake requires flour, eggs, and sugar.",
        metadata={"source": "cooking.md", "section_title": "Baking", "doc_title": "Desserts"},
        similarity_score=0.15,
        is_relevant=False,
        relevance_score=0.10
    )


@pytest.fixture
async def async_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
