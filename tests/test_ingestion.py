import pytest
from app.ingestion.chunking import TechnicalDocumentChunker
from app.retrieval.vectorstore import VectorStoreManager


def test_markdown_header_splitting(sample_markdown, chunker):
    metadata = {"source": "fastapi_di.md"}
    chunks = chunker.chunk_document(sample_markdown, metadata, is_markdown=True)
    
    assert len(chunks) >= 2
    section_titles = [c.metadata.get("section_title") for c in chunks]
    assert any("Overview" in s for s in section_titles)
    assert any("Advanced Sub-dependencies" in s for s in section_titles)


def test_code_block_preservation(sample_markdown, chunker):
    metadata = {"source": "fastapi_di.md"}
    chunks = chunker.chunk_document(sample_markdown, metadata, is_markdown=True)
    
    # Verify code fence is retained
    code_chunks = [c for c in chunks if "from fastapi import Depends" in c.page_content]
    assert len(code_chunks) > 0
    assert "def read_items" in code_chunks[0].page_content


def test_vector_store_indexing_and_search(sample_markdown, chunker, temp_vector_store):
    metadata = {"source": "fastapi_di.md", "doc_title": "FastAPI Dependency Guide"}
    chunks = chunker.chunk_document(sample_markdown, metadata, is_markdown=True)
    
    # Add to vector store
    added = temp_vector_store.add_documents(chunks)
    assert added == len(chunks)
    assert temp_vector_store.get_total_chunk_count() == len(chunks)

    # Search
    results = temp_vector_store.similarity_search_with_score("How does Depends work in FastAPI?", k=2)
    assert len(results) > 0
    doc, score = results[0]
    assert score >= 0.0
    assert "Depends" in doc.page_content


def test_document_listing_aggregation(sample_markdown, chunker, temp_vector_store):
    metadata = {"source": "fastapi_di.md", "doc_title": "FastAPI DI"}
    chunks = chunker.chunk_document(sample_markdown, metadata, is_markdown=True)
    temp_vector_store.add_documents(chunks)

    summary = temp_vector_store.list_indexed_documents()
    assert len(summary) == 1
    assert summary[0]["source"] == "fastapi_di.md"
    assert summary[0]["total_chunks"] == len(chunks)
    assert len(summary[0]["sections"]) > 0
